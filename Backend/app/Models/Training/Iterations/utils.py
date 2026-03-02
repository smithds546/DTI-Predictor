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

def save_loss_curve(train_losses, val_losses, title, save_path, stop_epoch=None):
    """
    Plots BCE train/val loss over epochs.
    If stop_epoch is provided (int, 1-indexed), draws a vertical dashed line
    at that epoch to mark where early stopping triggered.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, color=COLOUR,    lw=2, label="Train BCE")
    ax.plot(epochs, val_losses,   color="#555555", lw=2,
            linestyle="--", label="Val BCE")

    if stop_epoch is not None:
        ax.axvline(x=stop_epoch, color="crimson", lw=1.5,
                   linestyle=":", label=f"Early stop (epoch {stop_epoch})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


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
