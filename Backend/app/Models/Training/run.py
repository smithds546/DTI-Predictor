import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files without a display

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score,
)

from basic import NeuralNetwork, binary_cross_entropy

# ── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)

MODEL_COLOUR = "#DD8452"   # muted orange — distinct from baseline blue
SAVE_DIR     = "figures"


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def save_loss_curve(train_losses, val_losses, save_path):
    """BCE loss for train and validation over epochs."""
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, color=MODEL_COLOUR, lw=2, label="Train BCE")
    ax.plot(epochs, val_losses,   color="#555555",    lw=2,
            linestyle="--", label="Val BCE")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title("Training Loss Curve — Iteration 1 (Mini-batch GD)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_roc_curve(y_true, y_prob, save_path):
    """ROC curve with AUC annotated."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc          = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=MODEL_COLOUR, lw=2,
            label=f"Iteration 1 (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Iteration 1 (Mini-batch GD)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_metrics_table(metrics_dict, save_path, baseline=None):
    """
    Saves a metrics summary table as a PNG.
    If baseline (dict of metric->value) is provided, appends a '+/- vs Basic'
    column showing the signed difference on the Test split, colour-coded
    green (improvement) or red (regression).
    """
    metric_order  = ["accuracy", "auc_roc", "pr_auc", "f1", "precision", "recall"]
    display_names = ["Accuracy", "AUC-ROC", "PR-AUC", "F1", "Precision", "Recall"]
    splits        = list(metrics_dict.keys())
    col_labels    = splits + (["+/- vs Basic"] if baseline else [])

    cell_text = []
    for m in metric_order:
        row = [f"{metrics_dict[split][m]:.4f}" for split in splits]
        if baseline:
            diff = metrics_dict["Test"][m] - baseline[m]
            row.append(f"{diff:+.4f}")
        cell_text.append(row)

    n_cols = len(col_labels)
    fig_w  = 5 + (2 if baseline else 0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=display_names,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.6)

    # Header row — model colour
    for j in range(n_cols):
        table[0, j].set_facecolor(MODEL_COLOUR)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Diff column — green / red per cell
    if baseline:
        diff_col = n_cols - 1
        for i, m in enumerate(metric_order):
            diff = metrics_dict["Test"][m] - baseline[m]
            colour = "#d4edda" if diff >= 0 else "#f8d7da"   # green / red tint
            table[i + 1, diff_col].set_facecolor(colour)

    ax.set_title("Performance Metrics — Iteration 1 (Mini-batch GD)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    DATA_ROOT = "/Users/drs/Projects/DTI/Backend/app/data/prepped"

    X_drug_train = np.load(f"{DATA_ROOT}/drugs/drug_train.npy")
    X_prot_train = np.load(f"{DATA_ROOT}/proteins/prot_train.npy")
    y_train = pd.read_csv(f"{DATA_ROOT}/bindingdb/bindingdb_train.csv")[
        "interaction"].values.reshape(-1, 1)

    X_drug_val = np.load(f"{DATA_ROOT}/drugs/drug_val.npy")
    X_prot_val = np.load(f"{DATA_ROOT}/proteins/prot_val.npy")
    y_val = pd.read_csv(f"{DATA_ROOT}/bindingdb/bindingdb_validation.csv")[
        "interaction"].values.reshape(-1, 1)

    X_drug_test = np.load(f"{DATA_ROOT}/drugs/drug_test.npy")
    X_prot_test = np.load(f"{DATA_ROOT}/proteins/prot_test.npy")
    y_test = pd.read_csv(f"{DATA_ROOT}/bindingdb/bindingdb_test.csv")[
        "interaction"].values.reshape(-1, 1)

    n_train = len(y_train)

    # ── Hyperparameters ────────────────────────────────────────────────────
    HIDDEN_SIZE   = 32
    LEARNING_RATE = 0.01
    EPOCHS        = 1000
    BATCH_SIZE    = 256
    PATIENCE      = 15

    model = NeuralNetwork(
        X_drug_train.shape[1], X_prot_train.shape[1],
        HIDDEN_SIZE, 1, LEARNING_RATE
    )

    # ── Early stopping state ───────────────────────────────────────────────
    best_val_loss     = np.inf
    epochs_no_improve = 0
    best_weights      = None

    train_losses = []
    val_losses   = []

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"Training for up to {EPOCHS} epochs  "
          f"(batch_size={BATCH_SIZE}, patience={PATIENCE})...")

    for epoch in range(EPOCHS):

        # Mini-batch gradient descent: shuffle indices each epoch
        indices = np.random.permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]
            model.forward(X_drug_train[idx], X_prot_train[idx])
            dW1, db1, dW2, db2 = model.backward(y_train[idx])
            model.update_weights(dW1, db1, dW2, db2)

        # Full-pass losses for tracking / early stopping
        # (model.y_hat after mini-batches only holds the last batch)
        train_pred = model.predict(X_drug_train, X_prot_train)
        train_loss = binary_cross_entropy(y_train, train_pred)
        val_pred   = model.predict(X_drug_val, X_prot_val)
        val_loss   = binary_cross_entropy(y_val, val_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:>4}/{EPOCHS}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"no_improve={epochs_no_improve}/{PATIENCE}")

        # ── Early stopping ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_weights = (
                model.W1.copy(), model.b1.copy(),
                model.W2.copy(), model.b2.copy(),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no improvement for {PATIENCE} epochs).")
                break

    # ── Restore best weights ───────────────────────────────────────────────
    if best_weights is not None:
        model.W1, model.b1, model.W2, model.b2 = best_weights
        print(f"Restored best weights (val_loss = {best_val_loss:.6f})")

    # ── Evaluate ───────────────────────────────────────────────────────────
    train_probs = model.predict(X_drug_train, X_prot_train).ravel()
    val_probs   = model.predict(X_drug_val,   X_prot_val).ravel()
    test_probs  = model.predict(X_drug_test,  X_prot_test).ravel()

    y_train_flat = y_train.ravel().astype(int)
    y_val_flat   = y_val.ravel().astype(int)
    y_test_flat  = y_test.ravel().astype(int)

    train_metrics = compute_metrics(y_train_flat, train_probs)
    val_metrics   = compute_metrics(y_val_flat,   val_probs)
    test_metrics  = compute_metrics(y_test_flat,  test_probs)

    print("\n" + "=" * 50)
    print("FINAL RESULTS — Iteration 1 (Mini-batch GD)")
    print("=" * 50)
    for split, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"\n  {split}:")
        for k, v in m.items():
            print(f"    {k:<12}: {v:.4f}")

    # ── Load baseline metrics for diff column (run basic/run.py first) ────
    baseline_path = os.path.join(
        os.path.dirname(__file__), "../basic/figures/basic_test_metrics.json"
    )
    baseline = None
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Loaded baseline metrics from {baseline_path}")
    else:
        print("Warning: basic_test_metrics.json not found — diff column omitted. "
              "Run basic/run.py first.")

    # ── Save figures ───────────────────────────────────────────────────────
    print("\nGenerating figures...")

    save_loss_curve(
        train_losses, val_losses,
        save_path=f"{SAVE_DIR}/iter1_loss_curve.png"
    )

    save_roc_curve(
        y_test_flat, test_probs,
        save_path=f"{SAVE_DIR}/iter1_roc_curve.png"
    )

    save_metrics_table(
        {"Train": train_metrics, "Validation": val_metrics, "Test": test_metrics},
        save_path=f"{SAVE_DIR}/iter1_metrics_table.png",
        baseline=baseline,
    )

    print(f"\nAll figures saved to ./{SAVE_DIR}/")


if __name__ == "__main__":
    main()
