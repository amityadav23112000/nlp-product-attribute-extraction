"""Shared evaluation.

* Entity (span) level: seqeval strict-span micro/macro F1 over BIO sequences - the
  metric used for all Phase 2 numbers (CRF 79.34, BERT 83.15, DeBERTa 85.77 on WDC).
* Product attribute/value pair level: distinct (attribute, exact value) pairs per
  product; used to compare token taggers with the LLM, which outputs pairs.
"""
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.metrics.sequence_labeling import get_entities


def entity_scores(gold, pred):
    return dict(precision=precision_score(gold, pred, zero_division=0), recall=recall_score(gold, pred, zero_division=0),
                micro_f1=f1_score(gold, pred, zero_division=0),
                macro_f1=f1_score(gold, pred, average="macro", zero_division=0))


def per_attribute(gold, pred):
    return classification_report(gold, pred, output_dict=True, zero_division=0)


def entities(labels):
    """BIO labels -> set of (attribute, start, end_inclusive), seqeval semantics."""
    return set(get_entities(list(labels)))


def entities_to_bio(length, spans):
    """Write non-overlapping spans back to BIO (earlier/longer spans win)."""
    labels = ["O"] * length
    for attr, start, end in sorted(spans, key=lambda s: (s[1], -s[2])):
        if any(l != "O" for l in labels[start:end + 1]):
            continue
        labels[start] = "B-" + attr
        for i in range(start + 1, end + 1):
            labels[i] = "I-" + attr
    return labels


def frequency_band(count):
    """Frozen Phase 2 long-tail thresholds on training span counts."""
    return "head" if count > 100 else "medium" if count >= 20 else "tail" if count >= 5 else "extreme_tail"


def bio_to_pairs(tokens, labels, text):
    """Map predicted BIO spans onto the original string and return distinct (attribute, value) pairs."""
    offsets, cursor = [], 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0 or text[cursor:start].strip():
            raise ValueError("tokens do not align with source text")
        offsets.append((start, start + len(token)))
        cursor = start + len(token)
    return {(attr, text[offsets[s][0]:offsets[e][1]]) for attr, s, e in get_entities(list(labels))}
