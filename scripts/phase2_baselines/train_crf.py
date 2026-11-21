"""Train + evaluate the classical CRF baseline (CPU, ~1-2 min).

  python scripts/phase2_baselines/train_crf.py --mode dev    # train -> validation
  python scripts/phase2_baselines/train_crf.py --mode final  # train+validation -> test
Outputs go to results/phase2_baselines/wdc/rerun_crf_<mode>/ (predictions, metrics, model.pkl).
"""
import argparse, json, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.attrext.crf import train_crf, predict_crf, DEFAULT_PARAMS
from src.attrext.io import read_jsonl, write_jsonl, write_json
from src.attrext.metrics import entity_scores, per_attribute

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("dev", "final"), default="dev")
    ap.add_argument("--data", type=Path, default=ROOT / "datasets/wdc_pave/bio")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--c1", type=float, default=DEFAULT_PARAMS["c1"])
    ap.add_argument("--c2", type=float, default=DEFAULT_PARAMS["c2"])
    a = ap.parse_args()
    out = a.out or ROOT / f"results/phase2_baselines/wdc/rerun_crf_{a.mode}"
    train = read_jsonl(a.data / "train.jsonl")
    if a.mode == "final":
        train += read_jsonl(a.data / "val.jsonl")
    evaluation = read_jsonl(a.data / ("val.jsonl" if a.mode == "dev" else "test.jsonl"))
    t0 = time.time()
    model = train_crf(train, c1=a.c1, c2=a.c2)
    print(f"trained on {len(train)} sequences in {time.time()-t0:.1f}s")
    pred = predict_crf(model, evaluation)
    gold = [r["labels"] for r in evaluation]
    scores = entity_scores(gold, pred)
    print(json.dumps({k: round(v, 4) for k, v in scores.items()}))
    write_jsonl([dict(id=r["id"], product_id=r["product_id"], category=r["category"], source=r["source"],
                      tokens=r["tokens"], gold=r["labels"], predicted=p) for r, p in zip(evaluation, pred)],
                out / "predictions.jsonl")
    write_json(dict(mode=a.mode, params={**DEFAULT_PARAMS, "c1": a.c1, "c2": a.c2}, scores=scores,
                    per_attribute=per_attribute(gold, pred), train_sequences=len(train),
                    seconds=time.time() - t0), out / "metrics.json")
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
