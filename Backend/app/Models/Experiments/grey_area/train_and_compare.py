"""
Grey Area Threshold Experiment -- Step 2: Training & Comparison

Calls run_dnn_adam.run() on each threshold variant prepared by prepare_data.py,
so each variant gets the full set of per-model figures (loss curve, ROC curve,
metrics table with baseline diff, threshold sweep).

The *original* threshold variant is always trained first and its test metrics
are used as the baseline for all adapted variants' metrics tables.

Usage:
    python train_and_compare.py              # train all four variants
    python train_and_compare.py narrow       # train only the 'narrow' variant
    python train_and_compare.py original narrow  # train specific variants

Outputs (per variant):
    figures/{variant}_test_metrics.json
    figures/{variant}_losses.json
    figures/{variant}_roc_data.json
    figures/{variant}_loss_curve.png
    figures/{variant}_roc_curve.png
    figures/{variant}_metrics_table.png
    figures/{variant}_threshold_sweep.png

Comparison figures (generated after all variants complete):
    figures/comparison_roc.png
    figures/comparison_metrics.png
    figures/comparison_loss.png
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")

sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "Torch"))
import run_dnn_adam

# ─── Variant config ─────────────────────────────────────────────────────────

ALL_VARIANTS = ["original", "narrow", "medium", "no_grey"]
COLOURS = {
    "original": "#4C72B0",
    "narrow":   "#DD8452",
    "medium":   "#55A868",
    "no_grey":  "#8172B2",
}

LABELS = {
    "original": "Original (5.30\u20137.00)",
    "narrow":   "Narrow (k=0.25)",
    "medium":   "Medium (k=0.50)",
    "no_grey":  "No grey (k=0)",
}


def _load_labels():
    """Try to load exact labels from dataset_summary.json."""
    summary_path = os.path.join(FIG_DIR, "dataset_summary.json")
    if not os.path.exists(summary_path):
        return
    with open(summary_path) as f:
        summary = json.load(f)
    stats = summary.get("_stats", {})
    mean = stats.get("mean")
    std  = stats.get("std")
    med  = stats.get("median")
    if mean is None or std is None:
        return
    LABELS["narrow"]  = (f"k=0.25 ({mean - 0.25*std:.2f}"
                         f"\u2013{mean + 0.25*std:.2f})")
    LABELS["medium"]  = (f"k=0.50 ({mean - 0.50*std:.2f}"
                         f"\u2013{mean + 0.50*std:.2f})")
    if med is not None:
        LABELS["no_grey"] = f"No grey (median={med:.2f})"


# ─── Comparison figures ──────────────────────────────────────────────────────

def plot_comparison_roc(results):
    fig, ax = plt.subplots(figsize=(7, 6))

    for v in ALL_VARIANTS:
        if v not in results:
            continue
        r = results[v]["roc"]
        ax.plot(r["fpr"], r["tpr"], color=COLOURS[v], lw=2,
                label=f"{LABELS[v]}  (AUC = {r['auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve \u2014 Grey Area Threshold Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "comparison_roc.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_roc.png")


def plot_comparison_metrics(results):
    metric_keys  = ["accuracy", "auc_roc", "pr_auc", "f1", "precision", "recall"]
    display_keys = ["Accuracy", "AUC-ROC", "PR-AUC", "F1", "Precision", "Recall"]

    present = [v for v in ALL_VARIANTS if v in results]
    col_labels = [LABELS[v] for v in present]

    cell_text = []
    for m in metric_keys:
        vals = [results[v]["metrics"][m] for v in present]
        cell_text.append([f"{val:.4f}" for val in vals])

    fig, ax = plt.subplots(figsize=(3.5 * len(present) + 2, 3.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=display_keys,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.8)

    # Colour headers
    for j, v in enumerate(present):
        table[0, j].set_facecolor(COLOURS[v])
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best per metric
    for i, m in enumerate(metric_keys):
        vals = [results[v]["metrics"][m] for v in present]
        best_idx = vals.index(max(vals))
        table[i + 1, best_idx].set_facecolor("#d4edda")

    ax.set_title("Test Metrics \u2014 Grey Area Threshold Comparison",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "comparison_metrics.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: comparison_metrics.png")


def plot_comparison_loss(results):
    present = [v for v in ALL_VARIANTS if v in results]
    n = len(present)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5 * rows), sharey=True)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, (ax, v) in enumerate(zip(axes, present)):
        losses = results[v]["losses"]
        epochs = range(1, len(losses["train"]) + 1)
        ax.plot(epochs, losses["train"], color=COLOURS[v], lw=2, label="Train")
        ax.plot(epochs, losses["val"],   color=COLOURS[v], lw=2,
                linestyle="--", label="Val")
        if results[v]["stop_epoch"]:
            ax.axvline(results[v]["stop_epoch"], color="crimson", lw=1.5,
                       linestyle=":", label=f"Stop ({results[v]['stop_epoch']})")
        ax.set_title(LABELS[v], fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        if i % cols == 0:
            ax.set_ylabel("BCE Loss", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(present):]:
        ax.set_visible(False)

    fig.suptitle("Loss Curves \u2014 Grey Area Threshold Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "comparison_loss.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: comparison_loss.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    _load_labels()

    # Allow selective training via CLI args
    requested = sys.argv[1:] if len(sys.argv) > 1 else ALL_VARIANTS
    for v in requested:
        if v not in ALL_VARIANTS:
            print(f"Unknown variant '{v}'. Choose from: {ALL_VARIANTS}")
            sys.exit(1)

    # Check that data exists
    for v in requested:
        data_path = os.path.join(DATA_DIR, v, "drugs", "drug_train.npy")
        if not os.path.exists(data_path):
            print(f"Data for '{v}' not found at {data_path}")
            print("Run prepare_data.py first.")
            sys.exit(1)

    # Train original first so its metrics can serve as baseline
    results = {}
    if "original" in requested:
        print(f"\n{'=' * 60}")
        print(f"  Training: {LABELS['original']}")
        print(f"{'=' * 60}")
        r = run_dnn_adam.run(
            data_root=os.path.join(DATA_DIR, "original"),
            prefix="original",
            label=LABELS["original"],
            save_dir=FIG_DIR,
            checkpoint=os.path.join(DATA_DIR, "original", "original_best.pt"),
            baseline=None,
            baseline_label="Original",
            threshold_sweep=False,
            threshold=0.3,
        )
        results["original"] = {
            "metrics":    r["test_metrics"],
            "losses":     r["losses"],
            "roc":        r["roc"],
            "stop_epoch": r["stop_epoch"],
        }

    # Load original baseline for the other variants
    original_baseline = None
    if "original" in results:
        original_baseline = results["original"]["metrics"]
    else:
        # Try loading from a previous run
        orig_path = os.path.join(FIG_DIR, "original_test_metrics.json")
        if os.path.exists(orig_path):
            with open(orig_path) as f:
                original_baseline = json.load(f)

    # Train remaining variants with original as baseline
    for v in requested:
        if v == "original":
            continue
        print(f"\n{'=' * 60}")
        print(f"  Training: {LABELS[v]}")
        print(f"{'=' * 60}")
        r = run_dnn_adam.run(
            data_root=os.path.join(DATA_DIR, v),
            prefix=v,
            label=LABELS[v],
            save_dir=FIG_DIR,
            checkpoint=os.path.join(DATA_DIR, v, f"{v}_best.pt"),
            baseline=original_baseline,
            baseline_label="Original",
            threshold_sweep=False,
            threshold=0.3,
        )
        results[v] = {
            "metrics":    r["test_metrics"],
            "losses":     r["losses"],
            "roc":        r["roc"],
            "stop_epoch": r["stop_epoch"],
        }

    # Load any previously trained variants for comparison figures
    for v in ALL_VARIANTS:
        if v not in results:
            metrics_path = os.path.join(FIG_DIR, f"{v}_test_metrics.json")
            losses_path  = os.path.join(FIG_DIR, f"{v}_losses.json")
            roc_path     = os.path.join(FIG_DIR, f"{v}_roc_data.json")
            if all(os.path.exists(p) for p in (metrics_path, losses_path, roc_path)):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                with open(losses_path) as f:
                    losses = json.load(f)
                with open(roc_path) as f:
                    roc = json.load(f)
                results[v] = {
                    "metrics": metrics, "losses": losses,
                    "roc": roc, "stop_epoch": None,
                }

    # Comparison figures
    print(f"\n{'=' * 60}")
    print("GENERATING COMPARISON FIGURES")
    print(f"{'=' * 60}")
    plot_comparison_roc(results)
    plot_comparison_metrics(results)
    plot_comparison_loss(results)

    # Final summary table
    present = [v for v in ALL_VARIANTS if v in results]
    print(f"\n{'=' * 60}")
    print("FINAL COMPARISON")
    print(f"{'=' * 60}")

    metric_keys = ["accuracy", "auc_roc", "pr_auc", "f1", "precision", "recall"]
    header = f"{'Metric':<12}" + "".join(f"  {LABELS[v]:>22}" for v in present)
    print(header)
    print("-" * len(header))
    for m in metric_keys:
        vals = [results[v]["metrics"][m] for v in present]
        best = max(vals)
        row = f"{m:<12}"
        for val in vals:
            marker = " *" if val == best else "  "
            row += f"  {val:>20.4f}{marker}"
        print(row)

    print(f"\n  * = best\n")
    print(f"All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
