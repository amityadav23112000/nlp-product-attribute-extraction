"""WDC-PAVE product records -> token-level BIO sequences.

This is the cleaned version of the Phase 2 preprocessing (``build_bio_dataset.py``).
It reproduces the Phase 2 processed files exactly (verified: all 320 validation and
708 test sequences have identical tokens and gold labels).

Rules
-----
* Each product has two sources: title (pid 0) and description (pid 1).
* Gold values are matched case-insensitively as literal strings.
* If a value occurs several times, ONE occurrence is labelled (values with fewer
  matches and longer values are placed first; the first non-overlapping
  occurrence wins). Overlapping annotations are rejected and logged.
* Gold span edges are forced to become token boundaries ("gold-assisted"
  tokenisation, e.g. ``BlackALTERNATE`` -> ``Black`` + ``ALTERNATE``). Raw-text
  inference (demo) uses :func:`tokenize_raw`, which never sees annotations.
"""
import re

TOKEN_PATTERN = re.compile(r"\w+(?:[./'’-]\w+)*|[^\w\s]", flags=re.UNICODE)


def normalize_label(attribute):
    """'Product Type' -> PRODUCT_TYPE, 'Color(s)' -> COLOR_S."""
    return re.sub(r"[^A-Z0-9]+", "_", attribute.upper()).strip("_")


def tokenize_raw(text):
    """Annotation-independent tokenizer used for real inference."""
    return [dict(text=m.group(), start=m.start(), end=m.end()) for m in TOKEN_PATTERN.finditer(text)]


def _find_all(text, value):
    if not text or not value:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(value), text, flags=re.IGNORECASE)]


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def tokenize_with_boundaries(text, spans):
    boundaries = {s["start"] for s in spans} | {s["end"] for s in spans}
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):
        start, end = match.start(), match.end()
        points = [start] + sorted(b for b in boundaries if start < b < end) + [end]
        for left, right in zip(points[:-1], points[1:]):
            if text[left:right].strip():
                tokens.append(dict(text=text[left:right], start=left, end=right))
    return tokens


def assign_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for span in spans:
        name = normalize_label(span["attribute"])
        inside = [i for i, t in enumerate(tokens) if t["start"] < span["end"] and t["end"] > span["start"]]
        if inside:
            labels[inside[0]] = "B-" + name
            for i in inside[1:]:
                labels[i] = "I-" + name
    return labels


def _source_spans(record, pid):
    field, source = ("input_title", "title") if pid == 0 else ("input_description", "description")
    text = record.get(field) or ""
    annotations, unmatched = [], []
    for attribute, values in record.get("target_scores", {}).items():
        if "n/a" in values:
            continue
        for value, meta in values.items():
            if not isinstance(meta, dict) or pid not in meta.get("pid", []):
                continue
            matches = _find_all(text, value)
            if matches:
                annotations.append(dict(attribute=attribute, value=value, matches=matches))
            else:
                unmatched.append(dict(product_id=record["id"], source=source, attribute=attribute, value=value))
    annotations.sort(key=lambda a: (len(a["matches"]), -len(a["value"])))
    selected, rejected = [], []
    for ann in annotations:
        chosen = next(((s, e) for s, e in ann["matches"]
                       if not any(_overlaps((s, e), (x["start"], x["end"])) for x in selected)), None)
        span = dict(start=(chosen or ann["matches"][0])[0], end=(chosen or ann["matches"][0])[1],
                    attribute=ann["attribute"], value=ann["value"])
        (selected if chosen else rejected).append(span)
    selected.sort(key=lambda s: s["start"])
    return text, source, selected, rejected, unmatched


def convert_record(record):
    """Return (sequences, unmatched_values, rejected_overlaps) for one product."""
    sequences, unmatched, rejected = [], [], []
    for pid in (0, 1):
        text, source, spans, rej, unm = _source_spans(record, pid)
        unmatched += unm
        rejected += [dict(product_id=record["id"], source=source, **s) for s in rej]
        if not text.strip():
            continue
        tokens = tokenize_with_boundaries(text, spans)
        sequences.append(dict(id=f"{record['id']}_{source}", product_id=record["id"],
                              cluster_id=record["cluster_id"], category=record["category"],
                              source=source, text=text, tokens=[t["text"] for t in tokens],
                              labels=assign_bio(tokens, spans)))
    return sequences, unmatched, rejected
