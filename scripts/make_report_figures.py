"""Report figures (PDF): F1 evolution across eras, and per-attribute F1 DeBERTa vs CPEC (WDC test, seed 42)."""
import json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attrext.io import read_jsonl
from src.attrext.metrics import per_attribute

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "report/figures"; OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: evolution of test micro-F1 (mean +- SD over seeds), one panel per dataset
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9), sharey=True)
for ax, ds, title in zip(axes, ("wdc", "mave"), ("WDC-PAVE", "MAVE pilot")):
    res = json.loads((ROOT / f"results/phase3_committee/{ds}/results.json").read_text())["test"]
    names = [n for n in ("CRF", "BERT", "DeBERTa", "CPEC_committee") if n in res]
    means = [statistics.mean(v["micro_f1"] for k, v in res[n].items() if k.isdigit()) * 100 for n in names]
    sds = [statistics.stdev([v["micro_f1"] for k, v in res[n].items() if k.isdigit()]) * 100 if len(res[n]) > 2 else 0 for n in names]
    labels = [n.replace("_committee", "") for n in names]
    colors = [BLUE if n != "CPEC_committee" else ORANGE for n in names]
    x = range(len(names))
    ax.bar(x, [m - 75 for m in means], bottom=75, width=0.62, color=colors, yerr=sds, capsize=2, error_kw=dict(lw=0.7, ecolor=INK))
    for i, m in zip(x, means):
        ax.text(i, m + 0.45, f"{m:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
    ax.set_title(title, fontsize=8, color=INK); ax.set_ylim(75, 90)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("Test entity micro-F1 (%)")
fig.tight_layout(pad=0.3); fig.savefig(OUT / "f1_evolution.pdf"); plt.close(fig)

# Figure 2: per-attribute F1 for the 14 most frequent WDC test attributes
P = ROOT / "results/phase2_baselines/wdc/predictions/deberta_test_seed42.jsonl"
C = ROOT / "results/phase3_committee/wdc/test_predictions_seed42.jsonl"
d, c = read_jsonl(P), read_jsonl(C)
rd = per_attribute([r["gold"] for r in d], [r["predicted"] for r in d])
rc = per_attribute([r["gold"] for r in c], [r["predicted"] for r in c])
attrs = sorted([a for a in rd if a.isupper() or "_" in a], key=lambda a: -rd[a]["support"])[:14][::-1]
fig, ax = plt.subplots(figsize=(3.3, 2.9))
y = range(len(attrs))
for i, a in zip(y, attrs):
    ax.plot([rd[a]["f1-score"] * 100, rc[a]["f1-score"] * 100], [i, i], color=GRID, lw=1.5, zorder=1)
ax.scatter([rd[a]["f1-score"] * 100 for a in attrs], y, s=14, color=BLUE, label="DeBERTa", zorder=2, edgecolor="white", lw=0.6)
ax.scatter([rc[a]["f1-score"] * 100 for a in attrs], y, s=14, color=ORANGE, label="CPEC (ours)", zorder=3, edgecolor="white", lw=0.6)
ax.set_yticks(list(y)); ax.set_yticklabels([f"{a.replace('_', ' ').title()} ({rd[a]['support']})" for a in attrs], fontsize=6.3)
ax.set_xlabel("Test F1 (%), seed 42"); ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "per_attribute.pdf", bbox_inches="tight"); fig.savefig(OUT / "per_attribute.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures:", [p.name for p in OUT.iterdir()])
