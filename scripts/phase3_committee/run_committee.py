"""Phase 3 novel method: Cross-Paradigm Evidence-grounded Committee (CPEC) - span level.

Voters per seed s: CRF (classical) + BERT_s + DeBERTa_s (fine-tuned PLMs) on WDC-PAVE;
CRF + DeBERTa_s on MAVE (no BERT was trained on MAVE in Phase 2).
Per-frequency-band vote thresholds are selected on VALIDATION ONLY (mean over the three
seeds) and frozen before the test split is scored.

  python scripts/phase3_committee/run_committee.py            # both datasets
Outputs: results/phase3_committee/<dataset>/{summary.md,results.json,test_predictions_seed*.jsonl}
"""
import json, statistics, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.attrext.committee import committee_predict, select_thresholds, score_rows, BANDS
from src.attrext.io import read_jsonl, write_jsonl, write_json
from src.attrext.metrics import entities, frequency_band

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "results/phase2_baselines"
SEEDS = (42, 43, 44)


def training_span_counts(dataset):
    if dataset == "wdc":
        return Counter(l[2:] for r in read_jsonl(ROOT / "datasets/wdc_pave/bio/train.jsonl") for l in r["labels"] if l.startswith("B-"))
    report = json.loads((PRED / "mave/reports/crf_validation_comparison.json").read_text())
    return Counter(report["training_attribute_span_counts"])


def load(dataset, model, split, seed=None):
    name = f"{model}_{split}" + (f"_seed{seed}" if seed else "") + ".jsonl"
    rows = read_jsonl(PRED / dataset / "predictions" / name)
    return sorted(rows, key=lambda r: r["id"])


def error_profile(rows):
    """Span-level error types: exact match / boundary / wrong attribute / spurious / missed."""
    c = Counter()
    for r in rows:
        g, p = entities(r["gold"]), entities(r["predicted"])
        c["correct"] += len(g & p)
        for e in p - g:
            ov = [x for x in g - p if x[1] <= e[2] and e[1] <= x[2]]
            c["boundary" if any(x[0] == e[0] for x in ov) else "wrong_attribute" if ov else "spurious"] += 1
        for x in g - p:
            if not any(x[1] <= e[2] and e[1] <= x[2] for e in p - g):
                c["missed"] += 1
    return dict(c)


def band_scores(rows, freq):
    out = {}
    for band in BANDS:
        keep = lambda labels: [l if l == "O" or frequency_band(freq.get(l[2:], 0)) == band else "O" for l in labels]
        sub = [dict(gold=keep(r["gold"]), predicted=keep(r["predicted"])) for r in rows]
        support = sum(1 for r in sub for l in r["gold"] if l.startswith("B-"))
        if support:
            out[band] = dict(score_rows(sub), support=support)
    return out


def run(dataset):
    freq = training_span_counts(dataset)
    voters = ["crf", "bert", "deberta"] if dataset == "wdc" else ["crf", "deberta"]
    cands = (1, 2, 3) if len(voters) == 3 else (1, 2)

    def group(split, seed):
        return [load(dataset, "deberta", split, seed)] + ([load(dataset, "bert", split, seed)] if "bert" in voters else []) + [load(dataset, "crf", split)]

    val_groups = [group("val", s) for s in SEEDS]
    tau, table = select_thresholds(val_groups, freq, cands)
    majority = {b: 2 for b in BANDS}
    res = dict(dataset=dataset, voters=voters, selected_thresholds=tau,
               band_definition="training span counts: head >100, medium 20-100, tail 5-19, extreme_tail <5",
               band_sizes=dict(Counter(frequency_band(v) for v in freq.values())),
               validation={}, test={}, selection_table=[dict(f1=f, tau=t) for f, t in sorted(table, key=lambda x: -x[0])[:10]])
    out_dir = ROOT / "results/phase3_committee" / dataset
    for split in ("val", "test"):
        target = res["validation" if split == "val" else "test"]
        for s in SEEDS:
            g = group(split, s)
            systems = {"DeBERTa": g[0], "CRF": g[-1]}
            if "bert" in voters:
                systems["BERT"] = g[1]
            systems["Committee_majority"] = committee_predict(g, freq, majority)
            systems["CPEC_committee"] = committee_predict(g, freq, tau)
            for name, rows in systems.items():
                if name == "CRF" and s != SEEDS[0]:
                    continue
                target.setdefault(name, {})[str(s)] = score_rows(rows)
            if split == "test":
                write_jsonl(systems["CPEC_committee"], out_dir / f"test_predictions_seed{s}.jsonl")
                target.setdefault("_error_profile", {})[str(s)] = {"DeBERTa": error_profile(g[0]), "CPEC_committee": error_profile(systems["CPEC_committee"])}
                target.setdefault("_bands", {})[str(s)] = {"DeBERTa": band_scores(g[0], freq), "CPEC_committee": band_scores(systems["CPEC_committee"], freq)}
    lines = [f"# CPEC committee - {dataset.upper()}", "",
             f"Voters per seed: {', '.join(voters)} (CRF shared across seeds).",
             f"Thresholds selected on validation only: `{json.dumps(tau)}`", "",
             "| Split | System | Seed 42 | Seed 43 | Seed 44 | Mean micro-F1 | SD (pp) | Mean P | Mean R |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split, key in (("validation", "validation"), ("test", "test")):
        for name, per in res[key].items():
            if name.startswith("_"):
                continue
            f = [per[k]["micro_f1"] for k in sorted(per)]
            p = statistics.mean(per[k]["precision"] for k in per); r = statistics.mean(per[k]["recall"] for k in per)
            cells = [f"{x*100:.2f}" for x in f] if len(f) == 3 else [f"{f[0]*100:.2f}", "-", "-"]
            sd = f"{statistics.stdev(f)*100:.2f}" if len(f) > 1 else "-"
            per_mean = statistics.mean(f)
            res[key][name]["mean_micro_f1"] = per_mean
            lines.append(f"| {split} | {name} | " + " | ".join(cells) + f" | **{per_mean*100:.2f}** | {sd} | {p*100:.2f} | {r*100:.2f} |")
    lines += ["", "## Test error profile (seed 42, span level)", "", "| System | correct | spurious | boundary | wrong attribute | missed |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, prof in res["test"]["_error_profile"]["42"].items():
        lines.append(f"| {name} | " + " | ".join(str(prof.get(k, 0)) for k in ("correct", "spurious", "boundary", "wrong_attribute", "missed")) + " |")
    lines += ["", "## Test micro-F1 by attribute-frequency band (seed 42)", "", "| Band | support | DeBERTa | CPEC |", "| --- | ---: | ---: | ---: |"]
    for band, sc in res["test"]["_bands"]["42"]["DeBERTa"].items():
        lines.append(f"| {band} | {sc['support']} | {sc['micro_f1']*100:.2f} | {res['test']['_bands']['42']['CPEC_committee'][band]['micro_f1']*100:.2f} |")
    lines += ["", "Entity-level strict span F1 (seqeval), identical to the Phase 2 metric. No test-based selection."]
    write_json(res, out_dir / "results.json")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    for d in (sys.argv[1:] or ["wdc", "mave"]):
        run(d)
