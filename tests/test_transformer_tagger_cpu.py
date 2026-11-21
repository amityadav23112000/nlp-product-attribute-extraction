"""CPU test of the Transformer tagger with a tiny randomly initialised BERT (no download):
label alignment, a short training run, prediction length, save/reload, raw-text demo predictor."""
import sys, tempfile, json
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.attrext.io import read_jsonl
from src.attrext.transformer_tagger import TransformerTagger, label_schema


def tiny_model(records, path):
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast
    path.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted({t.lower() for r in records for t in r["tokens"]})[:3000]
    (path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(path)
    BertForTokenClassification(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                                          num_attention_heads=2, intermediate_size=64)).save_pretrained(path)


def main():
    train = read_jsonl(ROOT / "datasets/wdc_pave/bio/train.jsonl")[:64]
    val = read_jsonl(ROOT / "datasets/wdc_pave/bio/val.jsonl")[:16]
    tmp = Path(tempfile.mkdtemp())
    tiny_model(train + val, tmp / "base")
    tagger = TransformerTagger(str(tmp / "base"), label_schema(train), device="cpu")
    hist = tagger.train(train, epochs=2, batch_size=16, eval_records=val)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"] * 1.5
    pred = tagger.predict(val)
    assert all(len(p) == len(r["tokens"]) for p, r in zip(pred, val))
    tagger.save(tmp / "models/bert")
    re = TransformerTagger(None, checkpoint=str(tmp / "models/bert"), device="cpu")
    assert re.predict(val) == pred
    import shutil, pickle
    from src.attrext.crf import train_crf
    pickle.dump(train_crf(train, max_iterations=20), open(tmp / "models/crf.pkl", "wb"))
    from demo.predictor import Extractor
    ex = Extractor(tmp / "models")
    rows = ex.extract("Xerox TONER CARTRIDGE 106R01409", "25K PAGE YIELD COLOR: Black")
    assert {r["system"] for r in rows} <= {"CRF", "BERT", "CPEC committee"}
    print("OK: tagger trains, predicts, reloads; demo predictor runs;", len(rows), "rows")


if __name__ == "__main__":
    main()
