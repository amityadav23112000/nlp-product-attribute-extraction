"""Step 1. Convert WDC-PAVE products (datasets/wdc_pave/*.jsonl) to BIO sequences.

Usage:  python scripts/data/build_wdc_bio.py
Writes: datasets/wdc_pave/bio/{train,val,test}.jsonl  (+ *_unmatched / *_overlaps logs, stats.json)
"""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.attrext.bio import convert_record
from src.attrext.io import read_jsonl, write_jsonl, write_json

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/wdc_pave"


def main():
    stats = {}
    for split in ("train", "val", "test"):
        products = read_jsonl(DATA / f"{split}.jsonl")
        seqs, unmatched, rejected = [], [], []
        for record in products:
            s, u, r = convert_record(record)
            seqs += s; unmatched += u; rejected += r
        write_jsonl(seqs, DATA / "bio" / f"{split}.jsonl")
        write_jsonl(unmatched, DATA / "bio" / f"{split}_unmatched.jsonl")
        write_jsonl(rejected, DATA / "bio" / f"{split}_overlaps.jsonl")
        labels = Counter(l for q in seqs for l in q["labels"])
        spans = Counter(l[2:] for q in seqs for l in q["labels"] if l.startswith("B-"))
        stats[split] = dict(products=len(products), sequences=len(seqs), tokens=sum(labels.values()),
                            positive_tokens=sum(v for k, v in labels.items() if k != "O"),
                            entity_spans=sum(spans.values()), attributes=len(spans),
                            unmatched_values=len(unmatched), rejected_overlaps=len(rejected),
                            span_counts=dict(spans.most_common()))
        print(f"{split:5s} products={len(products)} sequences={len(seqs)} spans={sum(spans.values())} "
              f"attributes={len(spans)} unmatched={len(unmatched)} overlaps={len(rejected)}")
    write_json(stats, DATA / "bio" / "stats.json")


if __name__ == "__main__":
    main()
