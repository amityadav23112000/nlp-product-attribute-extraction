"""CPEC with the LLM as a 4th, evidence-grounded voter (product attribute/value PAIR level).

Voters per seed s: CRF, BERT_s, DeBERTa_s (spans mapped to exact text values) and Qwen3-4B
(validation-selected prompt; only values that passed exact evidence verification).
Thresholds per frequency band are selected on the 49 validation products the LLM study used,
then frozen and applied to all 354 test products.

  python scripts/phase3_committee/run_committee_with_llm.py [--llm-dir results/phase3_llm]
"""
import argparse, json, statistics, sys
from collections import Counter
from itertools import product
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.attrext.committee import vote_pairs, BANDS
from src.attrext.io import read_jsonl, write_json
from src.attrext.metrics import frequency_band
from src.llm_extraction.data_adapter import gold_pairs
from src.llm_extraction.evaluator import evaluate_pairs, pair_key
from scripts.phase3_llm.generate_llm_report import baseline_pairs

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "results/phase2_baselines/wdc/predictions"
SEEDS = (42, 43, 44)


def as_sets(pred):
    return {pid: {pair_key(x) for x in items} for pid, items in pred.items()}


def to_items(sets):
    return {pid: [dict(attribute=a, value=v) for a, v in sorted(s)] for pid, s in sets.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm-dir", type=Path, default=ROOT / "results/phase3_llm")
    a = ap.parse_args()
    data = ROOT / "datasets/wdc_pave"
    manifest = json.loads((data / "manifest.json").read_text())
    freq = json.loads((data / "schema_registry.json").read_text())["training_pair_counts"]
    selection = json.loads((a.llm_dir / "wdc_test/selection.json").read_text())
    mode = selection["selected"]["mode"]
    splits = {"val": (manifest["selected_validation_ids"], data / "val.jsonl", a.llm_dir / "wdc_validation" / mode / "verified_predictions.jsonl"),
              "test": (manifest["test_ids"], data / "test.jsonl", a.llm_dir / "wdc_test/selected_prompt/verified_predictions.jsonl")}
    prepared = {}
    for split, (ids, raw, llm_file) in splits.items():
        rows = {str(r["id"]): r for r in read_jsonl(raw)}
        gold = {pid: gold_pairs(rows[pid]) for pid in ids}
        cats = {pid: rows[pid]["category"] for pid in ids}
        llm = {r["product_id"]: {pair_key(x) for x in r["attributes"]} for r in read_jsonl(llm_file)}
        crf = as_sets(baseline_pairs(P / f"crf_{split}.jsonl", rows, ids))
        groups = []
        for s in SEEDS:
            groups.append(dict(DeBERTa=as_sets(baseline_pairs(P / f"deberta_{split}_seed{s}.jsonl", rows, ids)),
                               BERT=as_sets(baseline_pairs(P / f"bert_{split}_seed{s}.jsonl", rows, ids)), CRF=crf, Qwen=llm))
        prepared[split] = (ids, gold, cats, groups)

    def committee(group, ids, tau, members):
        return {pid: vote_pairs([group[m][pid] for m in members], freq, tau) for pid in ids}

    def f1(split, sets):
        ids, gold, cats, _ = prepared[split]
        return evaluate_pairs(gold, to_items(sets), cats, freq)

    present = sorted({frequency_band(c) for c in freq.values()}, key=BANDS.index)
    configs = {"CPEC (CRF+BERT+DeBERTa)": ["DeBERTa", "BERT", "CRF"], "CPEC + Qwen (4 voters)": ["DeBERTa", "BERT", "CRF", "Qwen"]}
    report = dict(selected_prompt=mode, validation_products=len(prepared["val"][0]), test_products=len(prepared["test"][0]), systems={})
    for name, members in configs.items():
        ids, *_ , groups = prepared["val"]
        best = None
        for combo in product(range(1, len(members) + 1), repeat=len(present)):
            tau = {b: combo[present.index(b)] if b in present else len(members) for b in BANDS}
            score = statistics.mean(f1("val", committee(g, ids, tau, members))["micro_f1"] for g in groups)
            if best is None or (round(score, 10), -sum(tau.values())) > (round(best[0], 10), -sum(best[1].values())):
                best = (score, tau)
        report["systems"][name] = dict(tau=best[1], validation_mean_f1=best[0], test={})
        for s, g in zip(SEEDS, prepared["test"][3]):
            report["systems"][name]["test"][s] = f1("test", committee(g, prepared["test"][0], best[1], members))
    for single in ("CRF", "BERT", "DeBERTa", "Qwen"):
        report["systems"][single] = dict(test={s: f1("test", g[single]) for s, g in zip(SEEDS, prepared["test"][3])},
                                         validation_mean_f1=statistics.mean(f1("val", g[single])["micro_f1"] for g in prepared["val"][3]))
    lines = ["# CPEC with the LLM voter - WDC-PAVE (product attribute/value pairs)", "",
             f"LLM prompt selected on validation: **{mode}**. Thresholds selected on {report['validation_products']} validation products; test = all {report['test_products']} products.", "",
             "| System | thresholds | val F1 | test P | test R | test micro-F1 (mean of seeds) | SD (pp) | test macro-F1 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in report["systems"].items():
        t = list(r["test"].values()); f = [x["micro_f1"] for x in t]
        mean = lambda k: statistics.mean(x[k] for x in t) * 100
        lines.append(f"| {name} | {json.dumps(r.get('tau', '-'))} | {r['validation_mean_f1']*100:.2f} | {mean('precision'):.2f} | {mean('recall'):.2f} | **{statistics.mean(f)*100:.2f}** | {statistics.stdev(f)*100 if len(set(f))>1 else 0:.2f} | {mean('macro_f1'):.2f} |")
    lines += ["", "Pair metric: distinct exact (attribute, value) pairs per product (Unicode NFC only). CRF and Qwen are single runs (repeated across seed columns)."]
    out = ROOT / "results/phase3_committee/wdc_with_llm"
    for r in report["systems"].values():
        for t in r["test"].values():
            t.pop("errors", None)
    write_json(report, out / "results.json")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
