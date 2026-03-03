"""
Architecture Baseline — 1 Hidden Layer [32]

Single hidden layer with ReLU activation, He init, mini-batch GD, no dropout.
This is the architecture baseline (equivalent to the best training-procedure
result). All architecture iterations build on top of this.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import roc_curve, roc_auc_score
from nn    import NeuralNetwork, binary_cross_entropy
from utils import compute_metrics, load_metrics, save_loss_curve, save_metrics_table

np.random.seed(42)

LABEL     = "Arch Basic — 1 Hidden Layer [32]"
PREFIX    = "arch_basic"
SAVE_DIR  = "figures"

DATA_ROOT     = "/Users/drs/Projects/DTI/Backend/app/data/prepped"
HIDDEN_SIZES  = [32]
LEARNING_RATE = 0.01
EPOCHS        = 1000
BATCH_SIZE    = 256


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────────
    model = NeuralNetwork(
        X_drug_train.shape[1], X_prot_train.shape[1],
        HIDDEN_SIZES, 1, LEARNING_RATE,
        dropout_rate=0.0,
    )

    train_losses, val_losses = [], []

    print(f"[{LABEL}] Training for {EPOCHS} epochs (batch_size={BATCH_SIZE})...")

    for epoch in range(EPOCHS):
        indices = np.random.permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]
            model.forward(X_drug_train[idx], X_prot_train[idx])
            dWs, dbs = model.backward(y_train[idx])
            model.update_weights(dWs, dbs)

        train_pred = model.predict(X_drug_train, X_prot_train)
        val_pred   = model.predict(X_drug_val,   X_prot_val)
        train_losses.append(binary_cross_entropy(y_train, train_pred))
        val_losses.append(binary_cross_entropy(y_val, val_pred))

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:>4}/{EPOCHS}  "
                  f"train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    train_metrics = compute_metrics(y_train.ravel().astype(int),
                                    model.predict(X_drug_train, X_prot_train).ravel())
    val_metrics   = compute_metrics(y_val.ravel().astype(int),
                                    model.predict(X_drug_val, X_prot_val).ravel())
    test_metrics  = compute_metrics(y_test.ravel().astype(int),
                                    model.predict(X_drug_test, X_prot_test).ravel())

    print(f"\n{'='*50}\nFINAL RESULTS — {LABEL}\n{'='*50}")
    for split, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"\n  {split}:")
        for k, v in m.items():
            print(f"    {k:<12}: {v:.4f}")

    # ── Persist (needed by arch1 for diff + ROC overlay) ──────────────────
    with open(f"{SAVE_DIR}/{PREFIX}_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(f"{SAVE_DIR}/{PREFIX}_losses.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)

    test_probs  = model.predict(X_drug_test, X_prot_test).ravel()
    y_true      = y_test.ravel().astype(int)
    fpr, tpr, _ = roc_curve(y_true, test_probs)
    auc         = roc_auc_score(y_true, test_probs)
    with open(f"{SAVE_DIR}/{PREFIX}_roc_data.json", "w") as f:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}, f)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    save_loss_curve(
        train_losses, val_losses,
        title=f"Loss Curve — {LABEL}",
        save_path=f"{SAVE_DIR}/{PREFIX}_loss_curve.png",
    )
    save_metrics_table(
        {"Train": train_metrics, "Validation": val_metrics, "Test": test_metrics},
        title=f"Performance Metrics — {LABEL}",
        save_path=f"{SAVE_DIR}/{PREFIX}_metrics_table.png",
    )
    print(f"\nAll figures saved to ./{SAVE_DIR}/")


if __name__ == "__main__":
    main()
