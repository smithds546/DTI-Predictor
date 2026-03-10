"""
Torch DNN — Iteration 1: Adam + Cosine LR Decay.

Builds on the SGD baseline (run_dnn.py) by replacing SGD+momentum with the
Adam optimiser and adding CosineAnnealingLR.  Everything else — model
architecture, data, dropout, loss function — is identical to the SGD run.

Key changes vs. SGD baseline (run_dnn.py):
  - Adam (lr=1e-3, β1=0.9, β2=0.999, weight_decay=1e-4)
  - CosineAnnealingLR (T_max=100 epochs, eta_min=1e-5)
  - EPOCHS=100  (Adam converges much faster than SGD)
  - BATCH_SIZE=512  (larger batches benefit Adam's variance reduction)

Everything inherited from SGD baseline:
  - Dual-branch DTI_DNN architecture
  - Dropout=0.4, BCEWithLogitsLoss + pos_weight
  - Same data splits (drug_train/val/test, prot_train/val/test, bindingdb CSV)
  - Early stopping on val AUC, patience=15

Outputs: figures/{PREFIX}_test_metrics.json, _losses.json, _roc_data.json,
         _loss_curve.png, _roc_curve.png, _metrics_table.png
         (comparing against the SGD baseline)

Usage:
    python run_dnn_adam.py
    (run run_dnn.py first so the SGD baseline figures exist for comparison)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Architecture"))
from utils import (compute_metrics, load_metrics,
                   save_loss_curve, save_loss_curve_overlay,
                   save_roc_curve_overlay, save_metrics_table,
                   save_threshold_plot)

from dnn import DTI_DNN

# ─── Labels & Paths ───────────────────────────────────────────────────────────
LABEL    = "Torch DNN — Adam + Cosine LR"
PREFIX   = "dnn_adam"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"{PREFIX}_best.pt")

DATA_ROOT = "/Users/drs/Projects/DTI/Backend/app/data/prepped"

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 512
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 15      # early stopping on val AUC
THRESHOLD            = 0.5
THRESHOLD_SWEEP      = [0.1, 0.3, 0.5, 0.7, 0.9]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load and convert all splits to float32 tensors."""
    X_drug_train = torch.tensor(np.load(f"{DATA_ROOT}/drugs/drug_train.npy"),    dtype=torch.float32)
    X_drug_val   = torch.tensor(np.load(f"{DATA_ROOT}/drugs/drug_val.npy"),      dtype=torch.float32)
    X_drug_test  = torch.tensor(np.load(f"{DATA_ROOT}/drugs/drug_test.npy"),     dtype=torch.float32)

    X_prot_train = torch.tensor(np.load(f"{DATA_ROOT}/proteins/prot_train.npy"), dtype=torch.float32)
    X_prot_val   = torch.tensor(np.load(f"{DATA_ROOT}/proteins/prot_val.npy"),   dtype=torch.float32)
    X_prot_test  = torch.tensor(np.load(f"{DATA_ROOT}/proteins/prot_test.npy"),  dtype=torch.float32)

    y_train = torch.tensor(
        pd.read_csv(f"{DATA_ROOT}/bindingdb/bindingdb_train.csv")["interaction"].values,
        dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(
        pd.read_csv(f"{DATA_ROOT}/bindingdb/bindingdb_validation.csv")["interaction"].values,
        dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(
        pd.read_csv(f"{DATA_ROOT}/bindingdb/bindingdb_test.csv")["interaction"].values,
        dtype=torch.float32).unsqueeze(1)

    return (X_drug_train, X_prot_train, y_train,
            X_drug_val,   X_prot_val,   y_val,
            X_drug_test,  X_prot_test,  y_test)


def make_loader(X_drug, X_prot, y, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X_drug, X_prot, y),
                      batch_size=BATCH_SIZE, shuffle=shuffle)


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_d, x_p, y in loader:
        x_d, x_p, y = x_d.to(device), x_p.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_d, x_p), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for x_d, x_p, y in loader:
        x_d, x_p, y = x_d.to(device), x_p.to(device), y.to(device)
        logits = model(x_d, x_p)
        total_loss += criterion(logits, y).item() * len(y)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return (total_loss / len(loader.dataset),
            np.vstack(all_probs).ravel(),
            np.vstack(all_labels).ravel())


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    (X_drug_train, X_prot_train, y_train,
     X_drug_val,   X_prot_val,   y_val,
     X_drug_test,  X_prot_test,  y_test) = load_data()

    print(f"  Train : {X_drug_train.shape[0]:,}  "
          f"(drug {X_drug_train.shape[1]}d, prot {X_prot_train.shape[1]}d)")
    print(f"  Val   : {X_drug_val.shape[0]:,}")
    print(f"  Test  : {X_drug_test.shape[0]:,}")

    train_loader = make_loader(X_drug_train, X_prot_train, y_train, shuffle=True)
    val_loader   = make_loader(X_drug_val,   X_prot_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_drug_test,  X_prot_test,  y_test,  shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = DTI_DNN(
        drug_input_dim=X_drug_train.shape[1],
        prot_input_dim=X_prot_train.shape[1],
        encoder_drop=0.4,       # Arch 3 best dropout (same as SGD baseline)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    n_pos      = y_train.sum().item()
    n_neg      = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[{LABEL}] Training up to {EPOCHS} epochs "
          f"(batch={BATCH_SIZE}, lr={LR}, wd={WEIGHT_DECAY}, patience={PATIENCE})...")

    train_losses, val_losses = [], []
    best_val_auc      = 0.0
    epochs_no_improve = 0
    stop_epoch        = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"val_auc={val_auc:.4f}  lr={lr_now:.2e}")

        if val_auc > best_val_auc:
            best_val_auc      = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                stop_epoch = epoch
                print(f"\nEarly stopping at epoch {stop_epoch} "
                      f"(no improvement for {PATIENCE} epochs).")
                break

    print(f"\nBest val_auc: {best_val_auc:.4f}  (checkpoint: {CHECKPOINT})")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

    # ── Evaluate ──────────────────────────────────────────────────────────
    _, train_probs, train_labels = evaluate(model, train_loader, criterion, device)
    _, val_probs,   val_labels   = evaluate(model, val_loader,   criterion, device)
    _, test_probs,  test_labels  = evaluate(model, test_loader,  criterion, device)

    train_metrics = compute_metrics(train_labels.astype(int), train_probs, THRESHOLD)
    val_metrics   = compute_metrics(val_labels.astype(int),   val_probs,   THRESHOLD)
    test_metrics  = compute_metrics(test_labels.astype(int),  test_probs,  THRESHOLD)

    print(f"\n{'='*50}\nFINAL RESULTS — {LABEL}\n{'='*50}")
    for split, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"\n  {split}:")
        for k, v in m.items():
            print(f"    {k:<12}: {v:.4f}")

    # ── Persist ───────────────────────────────────────────────────────────
    with open(f"{SAVE_DIR}/{PREFIX}_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(f"{SAVE_DIR}/{PREFIX}_losses.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)

    fpr, tpr, _ = roc_curve(test_labels.astype(int), test_probs)
    auc         = roc_auc_score(test_labels.astype(int), test_probs)
    with open(f"{SAVE_DIR}/{PREFIX}_roc_data.json", "w") as f:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}, f)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    baseline = load_metrics(f"{SAVE_DIR}/dnn_sgd_test_metrics.json", label="run_dnn.py")

    prev_losses = load_metrics(f"{SAVE_DIR}/dnn_sgd_losses.json", label="run_dnn.py")
    if prev_losses:
        save_loss_curve_overlay(
            prev_losses["train"], prev_losses["val"], train_losses, val_losses,
            prev_label="Torch SGD", curr_label="Torch Adam",
            title=f"Loss Curve — Torch SGD vs {LABEL}",
            save_path=f"{SAVE_DIR}/{PREFIX}_loss_curve.png",
            stop_epoch=stop_epoch,
        )
    else:
        save_loss_curve(
            train_losses, val_losses,
            title=f"Loss Curve — {LABEL}",
            save_path=f"{SAVE_DIR}/{PREFIX}_loss_curve.png",
            stop_epoch=stop_epoch,
        )

    prev_roc = load_metrics(f"{SAVE_DIR}/dnn_sgd_roc_data.json", label="run_dnn.py")
    if prev_roc:
        save_roc_curve_overlay(
            prev_roc["fpr"], prev_roc["tpr"], prev_roc["auc"],
            fpr.tolist(), tpr.tolist(), auc,
            prev_label="Torch SGD", curr_label="Torch Adam",
            title=f"ROC Curve — Torch SGD vs {LABEL}",
            save_path=f"{SAVE_DIR}/{PREFIX}_roc_curve.png",
        )

    save_metrics_table(
        {"Train": train_metrics, "Validation": val_metrics, "Test": test_metrics},
        title=f"Performance Metrics — {LABEL}",
        save_path=f"{SAVE_DIR}/{PREFIX}_metrics_table.png",
        baseline=baseline,
        baseline_label="Torch SGD",
    )
    save_threshold_plot(
        test_labels.astype(int), test_probs,
        thresholds=THRESHOLD_SWEEP,
        title=f"Metric vs Threshold — {LABEL}",
        save_path=f"{SAVE_DIR}/{PREFIX}_threshold_sweep.png",
    )

    print(f"\nAll figures saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
