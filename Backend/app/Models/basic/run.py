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

from basic import NeuralNetwork

# ── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)

# ── consistent colour for the basic model across all dissertation figures ──
MODEL_COLOUR = "#4C72B0"   # muted blue
SAVE_DIR     = "figures"   # all figures saved here


def save_roc_curve(y_true, y_prob, save_path):
    """ROC curve with AUC annotated."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc          = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=MODEL_COLOUR, lw=2,
            label=f"Basic Model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Basic NumPy Model", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_confusion_matrix(y_true, y_pred, save_path):
    """Normalised confusion matrix heatmap."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Proportion")

    classes = ["No Interaction (0)", "Interaction (1)"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks);  ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks);  ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Normalised Confusion Matrix — Basic NumPy Model", fontsize=12)

    thresh = 0.5
    for i in range(2):
        for j in range(2):
            colour = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]:,})",
                    ha="center", va="center", color=colour, fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_metrics_table(metrics_dict, save_path):
    """
    Saves a clean metrics table as a PNG.
    metrics_dict = {"Train": {...}, "Validation": {...}, "Test": {...}}
    """
    metric_order  = ["accuracy", "auc_roc", "pr_auc", "f1", "precision", "recall"]
    display_names = ["Accuracy", "AUC-ROC", "PR-AUC", "F1", "Precision", "Recall"]
    splits        = list(metrics_dict.keys())

    cell_text = []
    for m in metric_order:
        row = [f"{metrics_dict[split][m]:.4f}" for split in splits]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=display_names,
        colLabels=splits,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.6)

    for j, _ in enumerate(splits):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(len(metric_order)):
        table[i + 1, -1].set_facecolor("#f0f4f8")

    ax.set_title("Performance Metrics — Basic NumPy Model",
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


def main():
    import os
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

    # ── Hyperparameters ────────────────────────────────────────────────────
    HIDDEN_SIZE   = 32
    LEARNING_RATE = 0.5
    EPOCHS        = 1000
    PATIENCE      = 15    # stop if val_loss doesn't improve for this many epochs

    model = NeuralNetwork(
        X_drug_train.shape[1], X_prot_train.shape[1],
        HIDDEN_SIZE, 1, LEARNING_RATE
    )

    # ── Early stopping state ───────────────────────────────────────────────
    best_val_loss     = np.inf
    epochs_no_improve = 0
    best_weights      = None   # snapshot of weights at best val_loss

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"Training for up to {EPOCHS} epochs (patience={PATIENCE}, metric=val_loss)...")

    for epoch in range(EPOCHS):
        # Forward + backward + weight update (full-batch)
        model.forward(X_drug_train, X_prot_train)
        dW1, db1, dW2, db2 = model.backward(y_train)
        model.update_weights(dW1, db1, dW2, db2)

        # Compute losses every epoch for early stopping check
        train_loss = np.mean((model.y_hat - y_train) ** 2)
        val_pred   = model.predict(X_drug_val, X_prot_val)
        val_loss   = np.mean((val_pred - y_val) ** 2)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:>4}/{EPOCHS}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"no_improve={epochs_no_improve}/{PATIENCE}")

        # ── Early stopping logic ───────────────────────────────────────────
        #
        # If val_loss improves  → save a snapshot of the current weights
        #                          and reset the patience counter.
        # If val_loss stagnates → increment the counter. Once it hits
        #                          PATIENCE, training stops and the best
        #                          snapshot is restored before evaluation.
        #
        # This prevents the model being evaluated in an overfit state
        # and mirrors the early stopping used in the PyTorch DNN, making
        # the comparison fair.
        #
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
                print(f"\nEarly stopping triggered at epoch {epoch + 1} "
                      f"(val_loss did not improve for {PATIENCE} epochs).")
                print(f"Best val_loss: {best_val_loss:.6f}")
                break

    # ── Restore best weights before evaluation ─────────────────────────────
    # Ensures we report metrics for the best-generalising checkpoint,
    # not the final (potentially overfit) epoch state.
    if best_weights is not None:
        model.W1, model.b1, model.W2, model.b2 = best_weights
        print(f"\nRestored best weights (val_loss = {best_val_loss:.6f})")

    # ── Predict on all splits ──────────────────────────────────────────────
    train_probs = model.predict(X_drug_train, X_prot_train).ravel()
    val_probs   = model.predict(X_drug_val,   X_prot_val).ravel()
    test_probs  = model.predict(X_drug_test,  X_prot_test).ravel()

    y_train_flat = y_train.ravel().astype(int)
    y_val_flat   = y_val.ravel().astype(int)
    y_test_flat  = y_test.ravel().astype(int)

    test_preds = (test_probs >= 0.5).astype(int)

    # ── Compute metrics ────────────────────────────────────────────────────
    train_metrics = compute_metrics(y_train_flat, train_probs)
    val_metrics   = compute_metrics(y_val_flat,   val_probs)
    test_metrics  = compute_metrics(y_test_flat,  test_probs)

    print("\n" + "=" * 50)
    print("FINAL RESULTS — Basic NumPy Model")
    print("=" * 50)
    for split, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"\n  {split}:")
        for k, v in m.items():
            print(f"    {k:<12}: {v:.4f}")

    # ── Generate and save figures ──────────────────────────────────────────
    print("\nGenerating figures...")

    save_roc_curve(
        y_test_flat, test_probs,
        save_path=f"{SAVE_DIR}/basic_roc_curve.png"
    )

    save_confusion_matrix(
        y_test_flat, test_preds,
        save_path=f"{SAVE_DIR}/basic_confusion_matrix.png"
    )

    save_metrics_table(
        {"Train": train_metrics, "Validation": val_metrics, "Test": test_metrics},
        save_path=f"{SAVE_DIR}/basic_metrics_table.png"
    )

    print(f"\nAll figures saved to ./{SAVE_DIR}/")


if __name__ == "__main__":
    main()