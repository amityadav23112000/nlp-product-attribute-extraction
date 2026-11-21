"""Paired bootstrap over test PRODUCTS (title+description kept together): is CPEC better than DeBERTa?
  python scripts/phase3_committee/significance.py [wdc|mave]"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.attrext.io import read_jsonl, write_json
from src.attrext.metrics import entities
ROOT = Path(__file__).resolve().parents[2]


def counts(rows):
    per = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        g, p = entities(r["gold"]), entities(r["predicted"])
        c = per[str(r["product_id"])]; c[0] += len(g & p); c[1] += len(p - g); c[2] += len(g - p)
    return per


def f1(tp, fp, fn): return 2 * tp / (2 * tp + fp + fn)


def main(ds="wdc", B=2000):
    out = {}
    for s in (42, 43, 44):
        a = counts(read_jsonl(ROOT / f"results/phase2_baselines/{ds}/predictions/deberta_test_seed{s}.jsonl"))
        b = counts(read_jsonl(ROOT / f"results/phase3_committee/{ds}/test_predictions_seed{s}.jsonl"))
        ids = sorted(a); rng = random.Random(s); wins = 0; deltas = []
        for _ in range(B):
            sample = [rng.choice(ids) for _ in ids]
            fa = f1(*[sum(a[i][k] for i in sample) for k in range(3)]); fb = f1(*[sum(b[i][k] for i in sample) for k in range(3)])
            deltas.append(fb - fa); wins += fb > fa
        deltas.sort()
        out[s] = dict(delta=f1(*[sum(b[i][k] for i in ids) for k in range(3)]) - f1(*[sum(a[i][k] for i in ids) for k in range(3)]),
                      ci95=[deltas[int(.025 * B)], deltas[int(.975 * B)]], p_value=1 - wins / B)
        print(ds, s, {k: (round(v, 4) if not isinstance(v, list) else [round(x, 4) for x in v]) for k, v in out[s].items()})
    write_json(out, ROOT / f"results/phase3_committee/{ds}/significance.json")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["wdc"]))
