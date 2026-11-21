"""Classical baseline: linear-chain CRF (sklearn-crfsuite) with hand-crafted features.

Features per token: lowercased word, 1-3 char prefixes/suffixes, case/digit flags,
presence of - / . %, word shape (Xeon->Xxxx, 2.4GHz->d.dXxx), the source field
(title/description) and the same lexical/shape features of the previous and next token.
Phase 2 hyper-parameters (selected by grid search on validation): L-BFGS, c1=0.1,
c2=0.1, 150 iterations.
"""
import re

DEFAULT_PARAMS = dict(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=150, all_possible_transitions=True)


def word_shape(word):
    return "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in word)


def token_features(tokens, i, source):
    w = tokens[i]
    f = {"bias": 1.0, "word.lower": w.lower(),
         "prefix1": w[:1].lower(), "prefix2": w[:2].lower(), "prefix3": w[:3].lower(),
         "suffix1": w[-1:].lower(), "suffix2": w[-2:].lower(), "suffix3": w[-3:].lower(),
         "isupper": w.isupper(), "islower": w.islower(), "istitle": w.istitle(), "isdigit": w.isdigit(),
         "contains_digit": bool(re.search(r"\d", w)), "contains_alpha": bool(re.search(r"[A-Za-z]", w)),
         "contains_hyphen": "-" in w, "contains_slash": "/" in w, "contains_dot": "." in w,
         "contains_percent": "%" in w, "shape": word_shape(w), "source": source}
    if i > 0:
        p = tokens[i - 1]
        f.update({"-1:word.lower": p.lower(), "-1:shape": word_shape(p), "-1:isupper": p.isupper(), "-1:isdigit": p.isdigit()})
    else:
        f["BOS"] = True
    if i < len(tokens) - 1:
        n = tokens[i + 1]
        f.update({"+1:word.lower": n.lower(), "+1:shape": word_shape(n), "+1:isupper": n.isupper(), "+1:isdigit": n.isdigit()})
    else:
        f["EOS"] = True
    return f


def featurize(record):
    return [token_features(record["tokens"], i, record["source"]) for i in range(len(record["tokens"]))]


def train_crf(records, **params):
    from sklearn_crfsuite import CRF
    model = CRF(**{**DEFAULT_PARAMS, **params})
    model.fit([featurize(r) for r in records], [r["labels"] for r in records])
    return model


def predict_crf(model, records):
    return model.predict([featurize(r) for r in records])
