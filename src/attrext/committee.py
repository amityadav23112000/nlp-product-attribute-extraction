"""Novel contribution (Phase 3): Cross-Paradigm Evidence-grounded Committee (CPEC).

Idea. CRF, fine-tuned Transformers and an instruction-tuned LLM come from three eras of
NLP and make *different* mistakes: the CRF is precise but misses values (low recall),
DeBERTa/BERT over-generate spurious and boundary-shifted spans (precision is their
weak side, see Phase 2 error analysis) and the LLM extracts values in free form that
must be grounded back in the text. CPEC lets the paradigms vote on every candidate
(attribute, span) - or (attribute, value) pair when the LLM participates - and keeps a
candidate only if it gets at least ``tau[band]`` votes, where ``band`` is the
training-frequency band of the attribute (head >100 spans, medium 20-100, tail 5-19,
extreme tail <5). LLM votes count only after exact evidence verification (the value
must occur literally in the product text), so the LLM can never add a hallucinated value.

Thresholds are chosen on the VALIDATION split only (grid search, ties -> lower
thresholds = higher recall) and then frozen for the test split.
"""
from collections import Counter
from itertools import product
import statistics

from .metrics import entities, entities_to_bio, entity_scores, frequency_band

BANDS = ("head", "medium", "tail", "extreme_tail")


def vote_spans(label_sequences, freq, tau, weights=None):
    """label_sequences: list (one per system) of BIO label lists for ONE sentence."""
    weights = weights or [1.0] * len(label_sequences)
    votes = Counter()
    for w, labels in zip(weights, label_sequences):
        for e in entities(labels):
            votes[e] += w
    kept = [e for e, v in votes.items() if v >= tau[frequency_band(freq.get(e[0], 0))]]
    # higher-vote spans are written first so they win overlaps
    kept.sort(key=lambda e: -votes[e])
    return entities_to_bio(len(label_sequences[0]), kept)


def committee_predict(systems, freq, tau, weights=None):
    """systems: list of prediction lists (rows with id/tokens/gold/predicted), aligned by id."""
    index = [{r["id"]: r for r in s} for s in systems]
    out = []
    for row in systems[0]:
        seqs = [ix[row["id"]]["predicted"] for ix in index]
        out.append(dict(row, predicted=vote_spans(seqs, freq, tau, weights)))
    return out


def score_rows(rows):
    return entity_scores([r["gold"] for r in rows], [r["predicted"] for r in rows])


def select_thresholds(validation_groups, freq, candidates=(1, 2, 3), weights=None):
    """validation_groups: list of system-lists (e.g. one per seed). Returns best tau and table."""
    present = sorted({frequency_band(c) for c in freq.values()}, key=BANDS.index)
    table = []
    for combo in product(candidates, repeat=len(present)):
        tau = {b: combo[present.index(b)] if b in present else max(candidates) for b in BANDS}
        f = statistics.mean(score_rows(committee_predict(g, freq, tau, weights))["micro_f1"] for g in validation_groups)
        table.append((f, tau))
    # highest validation F1; ties broken toward lower (recall-friendlier) thresholds
    best = max(table, key=lambda x: (round(x[0], 10), -sum(x[1].values())))
    return best[1], table


# ---------- pair level (needed when the LLM, which outputs pairs, joins) ----------
def vote_pairs(pair_sets, freq, tau, weights=None):
    weights = weights or [1.0] * len(pair_sets)
    votes = Counter()
    for w, pairs in zip(weights, pair_sets):
        for p in set(pairs):
            votes[p] += w
    return {p for p, v in votes.items() if v >= tau[frequency_band(freq.get(p[0], 0))]}
