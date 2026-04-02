"""
Torch DNN — SGD + Momentum baseline.

Builds directly on Arch 3 (the best NumPy architecture: dropout=0.4,
L2 λ=0.01, hidden=[64,32]).  The dual-branch PyTorch encoder replaces
Arch 3's flat-concatenation input, while keeping the same optimiser
family (mini-batch SGD) so results sit naturally in the arch series.

Key design choices inherited from the architecture iterations:
  - Same data files and splits as all preceding models
  - Dropout = 0.4          (arch3 best)
  - Weight decay = 0.01    (arch3 best L2 λ)
  - SGD with momentum=0.9  (continuation of the arch series)
  - Early stopping on val AUC, patience=15  (from Iter 4)

New in this model (vs. Arch 3):
  - Dual-branch encoder: separate drug / protein sub-networks
  - BatchNorm after every linear layer
  - BCEWithLogitsLoss with pos_weight class balancing

Outputs: figures/{PREFIX}_test_metrics.json, _losses.json, _roc_data.json,
         _loss_curve.png, _roc_curve.png, _metrics_table.png
         (comparing against Arch 3 as the baseline)

Usage:
    python run_dnn.py
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
                   save_roc_curve_overlay, save_metrics_table)

from dnn import DTI_DNN

# ─── Labels & Paths ───────────────────────────────────────────────────────────
LABEL    = "Torch DNN — SGD + Momentum"
PREFIX   = "dnn_sgd"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
ARCH_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Architecture", "figures")
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"{PREFIX}_best.pt")

DATA_ROOT = "/Users/drs/Projects/DTI/Backend/app/data/prepped"

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 256
EPOCHS       = 1000
LR           = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.01    # L2 regularisation — matches Arch 3 λ=0.01
PATIENCE     = 75      # early stopping on val AUC (from Iter 4)
THRESHOLD    = 0.5


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

def run(data_root=None, prefix=None, label=None, save_dir=None,
        checkpoint=None, baseline=None, baseline_label="Baseline"):
    """
    Train DTI_DNN with SGD + Momentum and generate all figures.

    All parameters are optional — when omitted they fall back to the
    module-level defaults so the script still works standalone.

    Returns:
        dict with keys: test_metrics, losses, roc, stop_epoch
    """
    data_root  = data_root  or DATA_ROOT
    prefix     = prefix     or PREFIX
    label      = label      or LABEL
    save_dir   = save_dir   or SAVE_DIR
    checkpoint = checkpoint or CHECKPOINT

    os.makedirs(save_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    X_drug_train = torch.tensor(np.load(f"{data_root}/drugs/drug_train.npy"),    dtype=torch.float32)
    X_drug_val   = torch.tensor(np.load(f"{data_root}/drugs/drug_val.npy"),      dtype=torch.float32)
    X_drug_test  = torch.tensor(np.load(f"{data_root}/drugs/drug_test.npy"),     dtype=torch.float32)

    X_prot_train = torch.tensor(np.load(f"{data_root}/proteins/prot_train.npy"), dtype=torch.float32)
    X_prot_val   = torch.tensor(np.load(f"{data_root}/proteins/prot_val.npy"),   dtype=torch.float32)
    X_prot_test  = torch.tensor(np.load(f"{data_root}/proteins/prot_test.npy"),  dtype=torch.float32)

    y_train = torch.tensor(
        pd.read_csv(f"{data_root}/bindingdb/bindingdb_train.csv")["interaction"].values,
        dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(
        pd.read_csv(f"{data_root}/bindingdb/bindingdb_validation.csv")["interaction"].values,
        dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(
        pd.read_csv(f"{data_root}/bindingdb/bindingdb_test.csv")["interaction"].values,
        dtype=torch.float32).unsqueeze(1)

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
        encoder_drop=0.4,       # Arch 3 best dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    n_pos      = y_train.sum().item()
    n_neg      = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[{label}] Training up to {EPOCHS} epochs "
          f"(batch={BATCH_SIZE}, lr={LR}, momentum={MOMENTUM}, "
          f"wd={WEIGHT_DECAY}, patience={PATIENCE})...")

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

        if epoch % 100 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{EPOCHS}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc      = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                stop_epoch = epoch
                print(f"\nEarly stopping at epoch {stop_epoch} "
                      f"(no improvement for {PATIENCE} epochs).")
                break

    print(f"\nBest val_auc: {best_val_auc:.4f}  (checkpoint: {checkpoint})")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    # ── Evaluate ──────────────────────────────────────────────────────────
    _, train_probs, train_labels = evaluate(model, train_loader, criterion, device)
    _, val_probs,   val_labels   = evaluate(model, val_loader,   criterion, device)
    _, test_probs,  test_labels  = evaluate(model, test_loader,  criterion, device)

    train_metrics = compute_metrics(train_labels.astype(int), train_probs)
    val_metrics   = compute_metrics(val_labels.astype(int),   val_probs)
    test_metrics  = compute_metrics(test_labels.astype(int),  test_probs)

    print(f"\n{'='*50}\nFINAL RESULTS — {label}\n{'='*50}")
    for split, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"\n  {split}:")
        for k, v in m.items():
            print(f"    {k:<12}: {v:.4f}")

    # ── Persist ───────────────────────────────────────────────────────────
    with open(f"{save_dir}/{prefix}_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(f"{save_dir}/{prefix}_losses.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)

    fpr, tpr, _ = roc_curve(test_labels.astype(int), test_probs)
    auc         = roc_auc_score(test_labels.astype(int), test_probs)
    with open(f"{save_dir}/{prefix}_roc_data.json", "w") as f:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}, f)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    # Use provided baseline, or fall back to arch3 baseline for standalone use
    if baseline is None:
        baseline = load_metrics(f"{ARCH_FIG}/arch3_test_metrics.json", label="arch3.py")
        baseline_label = "Arch 3"

    prev_losses = load_metrics(f"{ARCH_FIG}/arch3_losses.json", label="arch3.py")
    if prev_losses:
        save_loss_curve_overlay(
            prev_losses["train"], prev_losses["val"], train_losses, val_losses,
            prev_label="Arch 3", curr_label=label,
            title=f"Loss Curve — Arch 3 vs {label}",
            save_path=f"{save_dir}/{prefix}_loss_curve.png",
            stop_epoch=stop_epoch,
        )
    else:
        save_loss_curve(
            train_losses, val_losses,
            title=f"Loss Curve — {label}",
            save_path=f"{save_dir}/{prefix}_loss_curve.png",
            stop_epoch=stop_epoch,
        )

    prev_roc = load_metrics(f"{ARCH_FIG}/arch3_roc_data.json", label="arch3.py")
    if prev_roc:
        save_roc_curve_overlay(
            prev_roc["fpr"], prev_roc["tpr"], prev_roc["auc"],
            fpr.tolist(), tpr.tolist(), auc,
            prev_label="Arch 3", curr_label=label,
            title=f"ROC Curve — Arch 3 vs {label}",
            save_path=f"{save_dir}/{prefix}_roc_curve.png",
        )

    save_metrics_table(
        {"Train": train_metrics, "Validation": val_metrics, "Test": test_metrics},
        title=f"Performance Metrics — {label}",
        save_path=f"{save_dir}/{prefix}_metrics_table.png",
        baseline=baseline,
        baseline_label=baseline_label,
    )
    print(f"\nAll figures saved to {save_dir}/")

    return {
        "test_metrics": test_metrics,
        "losses":       {"train": train_losses, "val": val_losses},
        "roc":          {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc},
        "stop_epoch":   stop_epoch,
    }


def main():
    run()


if __name__ == "__main__":
    main()
