"""Regenerate the historical neural-selection curve from saved source logs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/historical-stacking.json").read_text(encoding="utf-8"))
    history = report["epochs"]
    epochs = [r["epoch"] for r in history]
    aucs = [r["meta_validation_auc"] for r in history]
    best = max(history, key=lambda row: row["meta_validation_auc"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(11, 4.9), layout="constrained")
    fig.suptitle("Neural stacking · recorded experiment", fontsize=21, fontweight="bold")
    ax.plot(epochs, aucs, color="#2563EB", linewidth=2, label="Meta-validation macro ROC-AUC")
    ax.scatter([best["epoch"]], [best["meta_validation_auc"]], s=65, color="#F97316", zorder=3)
    ax.annotate(f"Selected epoch {best['epoch']} · {best['meta_validation_auc']:.5f}",
                xy=(best["epoch"], best["meta_validation_auc"]), xytext=(48, .804),
                arrowprops={"arrowstyle": "->", "color": "#64748B"}, fontsize=11)
    ax.set(xlabel="Epoch", ylabel="Macro ROC-AUC", xlim=(1, 100), ylim=(.789, .839))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(root / "docs/assets/stacking-history.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
