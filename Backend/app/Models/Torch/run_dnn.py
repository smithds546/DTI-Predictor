"""
Training and evaluation script for the PyTorch DTI_DNN model.

Loads the same prepped data as run.py (MACCS drug features + ProtBERT protein
features) so results are directly comparable to the NumPy baseline.

Outputs a side-by-side comparison table at the end.

Usage (from repo root or Models directory):
    python run_dnn.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, average_precision_score,
)

from dnn import DTI_DNN

# ---------------------------------------------------------------------------
# Paths  (mirror run.py)
# ---------------------------------------------------------------------------
DATA_ROOT = "/Users/drs/Projects/DTI/Backend/app/data/prepped"

DRUG_TRAIN  = os.path.join(DATA_ROOT, "drugs/drug_train.npy")
DRUG_VAL    = os.path.join(DATA_ROOT, "drugs/drug_val.npy")
DRUG_TEST   = os.path.join(DATA_ROOT, "drugs/drug_test.npy")

PROT_TRAIN  = os.path.join(DATA_ROOT, "proteins/prot_train.npy")
PROT_VAL    = os.path.join(DATA_ROOT, "proteins/prot_val.npy")
PROT_TEST   = os.path.join(DATA_ROOT, "proteins/prot_test.npy")

LABEL_TRAIN = os.path.join(DATA_ROOT, "bindingdb/bindingdb_train.csv")
LABEL_VAL   = os.path.join(DATA_ROOT, "bindingdb/bindingdb_validation.csv")
LABEL_TEST  = os.path.join(DATA_ROOT, "bindingdb/bindingdb_test.csv")

# Where to save the best checkpoint
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "dnn_best.pt")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE    = 512           # larger batches → faster epochs
EPOCHS        = 100
LR            = 1e-3
WEIGHT_DECAY  = 1e-4          # L2 regularisation
PATIENCE      = 15            # early-stopping patience (epochs without improvement)
THRESHOLD     = 0.5           # decision threshold for binary metrics


def load_data():
    """Load and convert all splits to float32 tensors."""
    X_drug_train = torch.tensor(np.load(DRUG_TRAIN), dtype=torch.float32)
    X_drug_val   = torch.tensor(np.load(DRUG_VAL),   dtype=torch.float32)
    X_drug_test  = torch.tensor(np.load(DRUG_TEST),  dtype=torch.float32)

    X_prot_train = torch.tensor(np.load(PROT_TRAIN), dtype=torch.float32)
    X_prot_val   = torch.tensor(np.load(PROT_VAL),   dtype=torch.float32)
    X_prot_test  = torch.tensor(np.load(PROT_TEST),  dtype=torch.float32)

    y_train = torch.tensor(
        pd.read_csv(LABEL_TRAIN)["interaction"].values, dtype=torch.float32
    ).unsqueeze(1)
    y_val   = torch.tensor(
        pd.read_csv(LABEL_VAL)["interaction"].values, dtype=torch.float32
    ).unsqueeze(1)
    y_test  = torch.tensor(
        pd.read_csv(LABEL_TEST)["interaction"].values, dtype=torch.float32
    ).unsqueeze(1)

    return (X_drug_train, X_prot_train, y_train,
            X_drug_val,   X_prot_val,   y_val,
            X_drug_test,  X_prot_test,  y_test)


def make_loader(X_drug, X_prot, y, shuffle: bool) -> DataLoader:
    ds = TensorDataset(X_drug, X_prot, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all classification metrics from probabilities."""
    y_pred = (y_prob >= THRESHOLD).astype(int)
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "auc_roc":   roc_auc_score(y_true, y_prob),
        "pr_auc":    average_precision_score(y_true, y_prob),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
    }
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_d, x_p, y in loader:
        x_d, x_p, y = x_d.to(device), x_p.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x_d, x_p)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns average loss, probabilities, and true labels."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for x_d, x_p, y in loader:
        x_d, x_p, y = x_d.to(device), x_p.to(device), y.to(device)
        logits = model(x_d, x_p)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    probs  = np.vstack(all_probs).ravel()
    labels = np.vstack(all_labels).ravel()
    return avg_loss, probs, labels


def print_metrics(split_name: str, metrics: dict):
    print(f"\n  {split_name} metrics:")
    for k, v in metrics.items():
        print(f"    {k:<12}: {v:.4f}")


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    (X_drug_train, X_prot_train, y_train,
     X_drug_val,   X_prot_val,   y_val,
     X_drug_test,  X_prot_test,  y_test) = load_data()

    print(f"  Train : {X_drug_train.shape[0]:,} samples  "
          f"(drug {X_drug_train.shape[1]}d, prot {X_prot_train.shape[1]}d)")
    print(f"  Val   : {X_drug_val.shape[0]:,} samples")
    print(f"  Test  : {X_drug_test.shape[0]:,} samples")

    train_loader = make_loader(X_drug_train, X_prot_train, y_train, shuffle=True)
    val_loader   = make_loader(X_drug_val,   X_prot_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_drug_test,  X_prot_test,  y_test,  shuffle=False)

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = DTI_DNN(
        drug_input_dim=X_drug_train.shape[1],
        prot_input_dim=X_prot_train.shape[1],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # Positive-class weight to handle potential class imbalance
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # CosineAnnealingLR decays LR smoothly to near-zero over T_max epochs then restarts.
    # More predictable than ReduceLROnPlateau and works well alongside AUC-based stopping.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ------------------------------------------------------------------
    # 3. Training loop — early stopping on val_AUC (higher = better)
    # ------------------------------------------------------------------
    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE}, metric=val_auc)...")
    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step()

        if (epoch % 5 == 0) or epoch == 1:
            val_acc = accuracy_score(val_labels, (val_probs >= THRESHOLD).astype(int))
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  "
                  f"val_auc={val_auc:.4f}  "
                  f"lr={current_lr:.2e}")

        # Early stopping & checkpointing on val_AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch} "
                      f"(val_auc did not improve for {PATIENCE} epochs).")
                break

    print(f"\nBest val_auc: {best_val_auc:.4f}  (checkpoint: {CHECKPOINT_PATH})")

    # ------------------------------------------------------------------
    # 4. Final evaluation on all splits using the best checkpoint
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    _, train_probs, train_labels = evaluate(model, train_loader, criterion, device)
    _, val_probs,   val_labels   = evaluate(model, val_loader,   criterion, device)
    _, test_probs,  test_labels  = evaluate(model, test_loader,  criterion, device)

    train_metrics = compute_metrics(train_labels, train_probs)
    val_metrics   = compute_metrics(val_labels,   val_probs)
    test_metrics  = compute_metrics(test_labels,  test_probs)

    print("\n" + "=" * 60)
    print("FINAL RESULTS — DTI_DNN (PyTorch)")
    print("=" * 60)
    print_metrics("Train",      train_metrics)
    print_metrics("Validation", val_metrics)
    print_metrics("Test",       test_metrics)

    # ------------------------------------------------------------------
    # 5. Comparison summary (to copy next to ReLu.py results)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<14} {'Basic (NumPy)':>16} {'DNN (PyTorch)':>16}")
    print("-" * 48)
    metric_labels = {
        "accuracy":  "Accuracy",
        "auc_roc":   "AUC-ROC",
        "pr_auc":    "PR-AUC",
        "f1":        "F1",
        "precision": "Precision",
        "recall":    "Recall",
    }
    for key, label in metric_labels.items():
        dnn_val = test_metrics[key]
        print(f"  {label:<12} {'(see run.py)':>16} {dnn_val:>16.4f}")

    print("\nNote: run run.py with the same test split to fill in the Basic column.")
    print("=" * 60)


if __name__ == "__main__":
    main()
