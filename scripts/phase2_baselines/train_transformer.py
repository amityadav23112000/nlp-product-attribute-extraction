"""Fine-tune BERT or DeBERTa-v3 as a BIO token classifier (GPU: CUDA or Apple MPS; CPU works but is slow).

  python scripts/phase2_baselines/train_transformer.py --model deberta --mode dev   --epochs 15
  python scripts/phase2_baselines/train_transformer.py --model deberta --mode final --epochs 15 --seed 42
dev   = train -> validation (per-epoch validation F1 printed live)
final = train+validation -> test (fixed epoch budget, no early stopping; Phase 2 protocol)
Saves predictions, metrics, per-epoch history and the checkpoint under --out (default models/<model>).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.attrext.io import read_jsonl, write_jsonl, write_json
from src.attrext.metrics import entity_scores, per_attribute
from src.attrext.transformer_tagger import TransformerTagger, label_schema

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"bert": "google-bert/bert-base-uncased", "deberta": "microsoft/deberta-v3-base"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=NAMES, default="deberta")
    ap.add_argument("--mode", choices=("dev", "final"), default="dev")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=Path, default=ROOT / "datasets/wdc_pave/bio")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or ROOT / "models" / a.model
    train = read_jsonl(a.data / "train.jsonl")
    val = read_jsonl(a.data / "val.jsonl")
    evaluation = val if a.mode == "dev" else read_jsonl(a.data / "test.jsonl")
    if a.mode == "final":
        train = train + val
    tagger = TransformerTagger(NAMES[a.model], label_schema(train))
    print(f"device={tagger.device} model={NAMES[a.model]} train_sequences={len(train)} labels={len(tagger.label2id)}")
    t0 = time.time()
    history = tagger.train(train, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, seed=a.seed,
                           eval_records=val if a.mode == "dev" else None)
    pred = tagger.predict(evaluation)
    gold = [r["labels"] for r in evaluation]
    scores = entity_scores(gold, pred)
    print("FINAL", a.mode, json.dumps({k: round(v, 4) for k, v in scores.items()}))
    tagger.save(out)
    write_jsonl([dict(id=r["id"], product_id=r["product_id"], category=r["category"], source=r["source"],
                      tokens=r["tokens"], gold=r["labels"], predicted=p) for r, p in zip(evaluation, pred)],
                out / f"predictions_{a.mode}.jsonl")
    write_json(dict(args={k: str(v) for k, v in vars(a).items()}, scores=scores, history=history,
                    per_attribute=per_attribute(gold, pred), seconds=time.time() - t0), out / f"metrics_{a.mode}.json")


if __name__ == "__main__":
    main()
