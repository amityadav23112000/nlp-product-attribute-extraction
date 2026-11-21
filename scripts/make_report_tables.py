"""Regenerate every LaTeX results table in report/tables/ from the saved results files.
Run after any experiment finishes:   python scripts/make_report_tables.py"""
import json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "report/tables"; T.mkdir(parents=True, exist_ok=True)
R = ROOT / "results"


def pct(x): return f"{x*100:.2f}"


def committee_table(dataset, label, caption):
    res = json.loads((R / f"phase3_committee/{dataset}/results.json").read_text())
    rows = []
    for split, key in (("Val", "validation"), ("Test", "test")):
        for name, per in res[key].items():
            if name.startswith("_") or name == "Committee_majority":
                continue
            f = [per[s]["micro_f1"] for s in ("42", "43", "44") if s in per]
            p = statistics.mean(per[s]["precision"] for s in per if s.isdigit()); r = statistics.mean(per[s]["recall"] for s in per if s.isdigit())
            sd = f"{statistics.stdev(f)*100:.2f}" if len(f) > 1 else "--"
            nm = {"CPEC_committee": "\\textbf{CPEC (ours)}"}.get(name, name)
            mean = pct(statistics.mean(f)); mean = f"\\textbf{{{mean}}}" if name == "CPEC_committee" else mean
            rows.append(f"{split} & {nm} & {pct(p)} & {pct(r)} & {mean} & {sd} \\\\")
        rows.append("\\midrule")
    body = "\n".join(rows[:-1])
    tau = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in res["selected_thresholds"].items())
    (T / f"committee_{dataset}.tex").write_text(f"""\\begin{{table}}[t]\\centering\\small
\\begin{{tabular}}{{llrrrr}}\\toprule
Split & System & P & R & micro-F1 & SD \\\\\\midrule
{body}
\\bottomrule\\end{{tabular}}
\\caption{{{caption} Mean over seeds 42/43/44 (CRF is a single model). Vote thresholds selected on validation: {tau}.}}\\label{{{label}}}\\end{{table}}
""")
    return res


def profile_table(res_w, res_m):
    lines = []
    for ds, res in (("WDC", res_w), ("MAVE", res_m)):
        for name, prof in res["test"]["_error_profile"]["42"].items():
            nm = "CPEC" if name.startswith("CPEC") else name
            lines.append(f"{ds} & {nm} & " + " & ".join(str(prof.get(k, 0)) for k in ("correct", "spurious", "boundary", "wrong_attribute", "missed")) + " \\\\")
    (T / "error_profile.tex").write_text("""\\begin{table}[t]\\centering\\small
\\begin{tabular}{llrrrrr}\\toprule
Data & System & Correct & Spurious & Boundary & Wrong attr. & Missed \\\\\\midrule
""" + "\n".join(lines) + """
\\bottomrule\\end{tabular}
\\caption{Span-level error profile on the test sets (seed 42). Boundary = overlapping span with the right attribute but wrong extent.}\\label{tab:errors}\\end{table}
""")
    lines = []
    for band, sc in res_w["test"]["_bands"]["42"]["DeBERTa"].items():
        c = res_w["test"]["_bands"]["42"]["CPEC_committee"][band]
        lines.append(f"{band.replace('_', ' ')} & {sc['support']} & {pct(sc['micro_f1'])} & {pct(c['micro_f1'])} \\\\")
    (T / "bands_wdc.tex").write_text("""\\begin{table}[t]\\centering\\small
\\begin{tabular}{lrrr}\\toprule
Frequency band & Test spans & DeBERTa & CPEC \\\\\\midrule
""" + "\n".join(lines) + """
\\bottomrule\\end{tabular}
\\caption{WDC-PAVE test micro-F1 by training-frequency band of the attribute (head $>$100 training spans, medium 20--100, tail 5--19), seed 42.}\\label{tab:bands}\\end{table}
""")


def llm_tables():
    val = R / "phase3_llm/wdc_validation/comparison.json"
    test = R / "phase3_llm/wdc_test/test_comparison.json"
    (T / "committee_llm.tex").write_text("% CPEC+LLM table appears after run_committee_with_llm.py\n")
    if not val.exists():
        (T / "llm_wdc.tex").write_text("% LLM results not yet available\n\\begin{table}[t]\\centering\\small\\fbox{\\parbox{0.9\\linewidth}{\\centering Qwen3-4B results: run \\texttt{notebooks/01\\_phase3\\_qwen\\_llm\\_wdc.ipynb} (or \\texttt{scripts/phase3\\_llm/run\\_local.py}) and \\texttt{scripts/make\\_report\\_tables.py}.}}\\caption{LLM prompting study (pending).}\\label{tab:llm}\\end{table}\n")
        return
    v = json.loads(val.read_text())
    rows = []
    for name, m in v["models"].items():
        nm = name.replace("_", " ")
        rel = ""
        mode = name.rsplit("_", 1)[0]
        if name.endswith("_verified") and mode in v["reliability"]:
            r = v["reliability"][mode]; rel = f"{r['json_parse_rate']*100:.0f}\\% / {r['unsupported_rate']*100:.1f}\\%"
        rows.append(f"{nm} & {pct(m['precision'])} & {pct(m['recall'])} & {pct(m['micro_f1'])} & {rel} \\\\")
    t_rows = []
    if test.exists():
        t = json.loads(test.read_text())
        for name, m in t["models"].items():
            t_rows.append(f"{name.replace('_', ' ')} & {pct(m['precision'])} & {pct(m['recall'])} & {pct(m['micro_f1'])} & {pct(m['macro_f1'])} \\\\")
        sel = t["selected_mode"].replace("_", "-")
    (T / "llm_wdc.tex").write_text("""\\begin{table}[t]\\centering\\small
\\begin{tabular}{lrrrl}\\toprule
Validation (49 products) & P & R & F1 & JSON ok / unsupported \\\\\\midrule
""" + "\n".join(rows) + "\n\\midrule\nTest (354 products) & P & R & micro-F1 & macro-F1 \\\\\\midrule\n" + "\n".join(t_rows) + f"""
\\bottomrule\\end{{tabular}}
\\caption{{Qwen3-4B-Instruct prompting study on WDC-PAVE, product attribute/value pair metric. ``raw'' = parsed output, ``verified'' = after exact evidence verification. Selected prompt (validation only): {sel if t_rows else 'n/a'}.}}\\label{{tab:llm}}\\end{{table}}
""")
    c = R / "phase3_committee/wdc_with_llm/results.json"
    if c.exists():
        res = json.loads(c.read_text()); rows = []
        for name, r in res["systems"].items():
            t = list(r["test"].values()); f = [x["micro_f1"] for x in t]
            rows.append(f"{name.replace('+', '+')} & {pct(r['validation_mean_f1'])} & {pct(statistics.mean(x['precision'] for x in t))} & {pct(statistics.mean(x['recall'] for x in t))} & {pct(statistics.mean(f))} \\\\")
        (T / "committee_llm.tex").write_text("""\\begin{table}[t]\\centering\\small
\\begin{tabular}{lrrrr}\\toprule
System (pair level) & Val F1 & Test P & Test R & Test F1 \\\\\\midrule
""" + "\n".join(rows) + """
\\bottomrule\\end{tabular}
\\caption{CPEC with Qwen3-4B as a fourth, evidence-grounded voter (WDC-PAVE, product attribute/value pairs, mean over transformer seeds).}\\label{tab:cpecllm}\\end{table}
""")
    else:
        (T / "committee_llm.tex").write_text("% pending\n")


if __name__ == "__main__":
    w = committee_table("wdc", "tab:cpec_wdc", "WDC-PAVE entity-level results (strict span micro-F1, \\%).")
    m = committee_table("mave", "tab:cpec_mave", "MAVE pilot entity-level results. Only two paradigms (CRF, DeBERTa) are available as voters.")
    profile_table(w, m)
    llm_tables()
    print("tables written:", sorted(p.name for p in T.glob("*.tex")))
