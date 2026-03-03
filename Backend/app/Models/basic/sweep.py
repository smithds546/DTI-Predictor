"""
Hyperparameter sweep for the Basic NumPy model.

Produces two figures:
  1. figures/sweep_lr_loss_curve.png
       Line graph — val BCE loss over epochs for each learning rate.

  2. figures/sweep_hidden_metrics.png
       Grouped bar chart — test Accuracy / AUC-ROC / F1 for each hidden size.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from basic import NeuralNetwork, binary_cross_entropy

np.random.seed(42)

SAVE_DIR  = "figures"
DATA_ROOT = "/Users/drs/Projects/DTI/Backend/app/data/prepped"

# ── Sweep values ───────────────────────────────────────────────────────────
LR_VALUES    = [1.0, 0.5, 0.1, 0.01, 0.001]
HIDDEN_SIZES = [8, 16, 32, 64, 128]

# Fixed settings used across all sweeps
EPOCHS      = 1000
HIDDEN_SIZE = 32    # fixed for LR sweep
LR_FIXED    = 0.01  # fixed for hidden size sweep

# Colour palette — one colour per sweep value (up to 5)
PALETTE = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
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

    return (X_drug_train, X_prot_train, y_train,
            X_drug_val,   X_prot_val,   y_val,
            X_drug_test,  X_prot_test,  y_test)


def train_model(X_drug_train, X_prot_train, y_train,
                X_drug_val,   X_prot_val,   y_val,
                hidden_size, lr, epochs):
    """
    Full-batch training. Returns (val_losses_per_epoch, final_model).
    """
    model = NeuralNetwork(
        X_drug_train.shape[1], X_prot_train.shape[1],
        hidden_size, 1, lr,
    )
    val_losses = []

    for epoch in range(epochs):
        model.forward(X_drug_train, X_prot_train)
        dW1, db1, dW2, db2 = model.backward(y_train)
        model.update_weights(dW1, db1, dW2, db2)

        val_pred = model.predict(X_drug_val, X_prot_val)
        val_losses.append(binary_cross_entropy(y_val, val_pred))

    return val_losses, model


# ---------------------------------------------------------------------------
# 1. Learning rate sweep — line graph
# ---------------------------------------------------------------------------

def run_lr_sweep(data):
    X_drug_train, X_prot_train, y_train, \
    X_drug_val,   X_prot_val,   y_val, \
    X_drug_test,  X_prot_test,  y_test = data

    print("Running learning rate sweep...")
    results = {}

    for lr in LR_VALUES:
        print(f"  LR = {lr}")
        np.random.seed(42)   # same init for fair comparison
        val_losses, _ = train_model(
            X_drug_train, X_prot_train, y_train,
            X_drug_val,   X_prot_val,   y_val,
            HIDDEN_SIZE, lr, EPOCHS,
        )
        results[lr] = val_losses

    return results


def save_lr_line_graph(results, save_path):
    epochs = range(1, EPOCHS + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for (lr, val_losses), colour in zip(results.items(), PALETTE):
        ax.plot(epochs, val_losses, color=colour, lw=1.8,
                label=f"LR = {lr}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Val BCE Loss", fontsize=12)
    ax.set_title("Learning Rate Sweep — Val Loss over Epochs", fontsize=13)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 2. Hidden size sweep — grouped bar chart
# ---------------------------------------------------------------------------

def run_hidden_sweep(data):
    X_drug_train, X_prot_train, y_train, \
    X_drug_val,   X_prot_val,   y_val, \
    X_drug_test,  X_prot_test,  y_test = data

    print("Running hidden size sweep...")
    results = {}

    for h in HIDDEN_SIZES:
        print(f"  Hidden size = {h}")
        np.random.seed(42)
        _, model = train_model(
            X_drug_train, X_prot_train, y_train,
            X_drug_val,   X_prot_val,   y_val,
            h, LR_FIXED, EPOCHS,
        )
        test_probs = model.predict(X_drug_test, X_prot_test).ravel()
        y_true     = y_test.ravel().astype(int)
        y_pred     = (test_probs >= 0.5).astype(int)

        results[h] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC-ROC":  roc_auc_score(y_true, test_probs),
            "F1":       f1_score(y_true, y_pred, zero_division=0),
        }

    return results


def save_hidden_bar_chart(results, save_path):
    metrics      = ["Accuracy", "AUC-ROC", "F1"]
    hidden_sizes = list(results.keys())
    n_metrics    = len(metrics)
    n_sizes      = len(hidden_sizes)

    x       = np.arange(n_sizes)
    width   = 0.22
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(9, 5))

    bar_colours = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (metric, colour) in enumerate(zip(metrics, bar_colours)):
        values = [results[h][metric] for h in hidden_sizes]
        bars   = ax.bar(x + offsets[i], values, width,
                        label=metric, color=colour, alpha=0.85)

        # Value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.003,
                    f"{height:.3f}", ha="center", va="bottom",
                    fontsize=8, color="#333333")

    ax.set_xlabel("Hidden Layer Size", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Hidden Size Sweep — Test Metrics", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hidden_sizes], fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    data = load_data()

    # Learning rate sweep
    lr_results = run_lr_sweep(data)
    save_lr_line_graph(
        lr_results,
        save_path=f"{SAVE_DIR}/sweep_lr_loss_curve.png",
    )

    # Hidden size sweep
    hidden_results = run_hidden_sweep(data)
    save_hidden_bar_chart(
        hidden_results,
        save_path=f"{SAVE_DIR}/sweep_hidden_metrics.png",
    )

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
