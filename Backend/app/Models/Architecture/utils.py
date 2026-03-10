"""Shared figure helpers for Iteration 1 run scripts."""

import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, average_precision_score,
)

COLOUR = "#DD8452"   # muted orange, consistent across all Iteration 1 figures

METRIC_ORDER  = ["accuracy", "auc_roc", "pr_auc", "f1", "precision", "recall"]
DISPLAY_NAMES = ["Accuracy", "AUC-ROC", "PR-AUC", "F1", "Precision", "Recall"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "auc_roc":   roc_auc_score(y_true, y_prob),
        "pr_auc":    average_precision_score(y_true, y_prob),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
    }


def load_metrics(json_path, label="previous"):
    """
    Loads test metrics from a JSON file at json_path.
    Returns None (with a warning) if the file doesn't exist yet.
    label is used only in the warning message.
    """
    if not os.path.exists(json_path):
        print(f"Warning: {os.path.basename(json_path)} not found — "
              f"diff column omitted. Run {label} first.")
        return None
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_roc_curve_overlay(prev_fpr, prev_tpr, prev_auc,
                           curr_fpr, curr_tpr, curr_auc,
                           prev_label, curr_label, title, save_path):
    """
    Overlays two iterations' ROC curves on one graph.
    Previous iteration is plotted in muted grey; current in full colour.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(prev_fpr, prev_tpr, color="#aaaaaa", lw=1.5,
            label=f"{prev_label} (AUC = {prev_auc:.4f})")
    ax.plot(curr_fpr, curr_tpr, color=COLOUR,    lw=2,
            label=f"{curr_label} (AUC = {curr_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


BLUE = "#4C72B0"   # paired with COLOUR (orange) for two-iteration overlays


def save_loss_curve_overlay(prev_train, prev_val, curr_train, curr_val,
                            prev_label, curr_label, title, save_path,
                            stop_epoch=None):
    """
    Overlays two iterations' loss curves on one graph.
    Each iteration gets its own colour (blue / orange); solid = train, dashed = val.
    Final BCE values are appended to legend labels for quick comparison.
    """
    epochs_prev = range(1, len(prev_train) + 1)
    epochs_curr = range(1, len(curr_train) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs_prev, prev_train, color=BLUE,   lw=2,
            label=f"{prev_label} Train  (final {prev_train[-1]:.4f})")
    ax.plot(epochs_prev, prev_val,   color=BLUE,   lw=2, linestyle="--",
            label=f"{prev_label} Val    (final {prev_val[-1]:.4f})")

    ax.plot(epochs_curr, curr_train, color=COLOUR, lw=2,
            label=f"{curr_label} Train  (final {curr_train[-1]:.4f})")
    ax.plot(epochs_curr, curr_val,   color=COLOUR, lw=2, linestyle="--",
            label=f"{curr_label} Val    (final {curr_val[-1]:.4f})")

    if stop_epoch is not None:
        ax.axvline(x=stop_epoch, color="crimson", lw=1.5,
                   linestyle=":", label=f"Early stop (epoch {stop_epoch})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(1, max(len(prev_train), len(curr_train)))
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_loss_curve(train_losses, val_losses, title, save_path, stop_epoch=None):
    """
    Plots BCE train/val loss over epochs.
    If stop_epoch is provided (int, 1-indexed), draws a vertical dashed line
    at that epoch to mark where early stopping triggered.
    Final BCE values are appended to legend labels for quick reading.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, color=COLOUR, lw=2,
            label=f"Train BCE  (final {train_losses[-1]:.4f})")
    ax.plot(epochs, val_losses,   color=BLUE,   lw=2, linestyle="--",
            label=f"Val BCE    (final {val_losses[-1]:.4f})")

    if stop_epoch is not None:
        ax.axvline(x=stop_epoch, color="crimson", lw=1.5,
                   linestyle=":", label=f"Early stop (epoch {stop_epoch})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(1, len(train_losses))
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_threshold_plot(y_true, y_prob, thresholds, title, save_path):
    """
    Saves two separate figures:
      - <save_path>              : Metric-vs-threshold line graph
      - <prefix>_table<ext>     : Metrics summary table
    AUC-ROC and PR-AUC are excluded as they are threshold-independent.
    """
    keys    = ["accuracy", "f1", "precision", "recall"]
    display = {"accuracy": "Accuracy", "f1": "F1", "precision": "Precision", "recall": "Recall"}
    colours = {"accuracy": "#4C72B0", "f1": COLOUR, "precision": "#55A868", "recall": "#C44E52"}

    metrics_at_thresh = {m: [] for m in keys}
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=t)
        for key in keys:
            metrics_at_thresh[key].append(m[key])

    # ── Line graph ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in keys:
        ax.plot(thresholds, metrics_at_thresh[key], marker="o", lw=2,
                color=colours[key], label=display[key])

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # ── Metrics table ─────────────────────────────────────────────────────────
    base, ext  = os.path.splitext(save_path)
    table_path = f"{base}_table{ext}"

    col_labels = [str(t) for t in thresholds]
    cell_text  = [
        [f"{metrics_at_thresh[k][i]:.3f}" for i in range(len(thresholds))]
        for k in keys
    ]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=[display[k] for k in keys],
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.8)

    for j in range(len(thresholds)):
        table[0, j].set_facecolor(COLOUR)
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i, key in enumerate(keys):
        table[i + 1, -1].set_facecolor(colours[key])
        table[i + 1, -1].set_text_props(color="white", fontweight="bold")

    ax.set_title("Metrics by Threshold", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {table_path}")


def save_metrics_table(metrics_dict, title, save_path, baseline=None, baseline_label="Basic"):
    """
    Saves a metrics summary table as a PNG.
    If baseline dict is provided, appends a '+/- vs <baseline_label>' column
    on the Test split, colour-coded green (improvement) / red (regression).
    """
    splits     = list(metrics_dict.keys())
    col_labels = splits + ([f"+/- vs {baseline_label}"] if baseline else [])
    n_cols     = len(col_labels)

    cell_text = []
    for m in METRIC_ORDER:
        row = [f"{metrics_dict[s][m]:.4f}" for s in splits]
        if baseline:
            diff = metrics_dict["Test"][m] - baseline[m]
            row.append(f"{diff:+.4f}")
        cell_text.append(row)

    fig_w = 5 + (2 if baseline else 0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=DISPLAY_NAMES,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.6)

    for j in range(n_cols):
        table[0, j].set_facecolor(COLOUR)
        table[0, j].set_text_props(color="white", fontweight="bold")

    if baseline:
        diff_col = n_cols - 1
        for i, m in enumerate(METRIC_ORDER):
            diff   = metrics_dict["Test"][m] - baseline[m]
            colour = "#d4edda" if diff >= 0 else "#f8d7da"
            table[i + 1, diff_col].set_facecolor(colour)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
