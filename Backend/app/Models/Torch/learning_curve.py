"""
Learning Curve — Sample Size Sensitivity Analysis.

Trains the DTI_DNN (Adam + Cosine LR) on 25 %, 50 %, 75 %, and 100 % of the
training data, evaluating each run on the same held-out test set.  Produces:

  figures/learning_curve_auc.json   — fraction, n_samples, test_auc_roc per run
  figures/learning_curve_auc.png    — line plot for the Evaluation chapter

Hyperparameters are identical to run_dnn_adam.py so results are comparable.
Val / test splits are never subsampled.

Usage:
    python learning_curve.py
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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Architecture"))

from dnn import DTI_DNN

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = "/Users/drs/Projects/DTI/Backend/app/data/prepped"
SAVE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# ─── Hyperparameters (identical to run_dnn_adam.py) ───────────────────────────
BATCH_SIZE   = 512
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 15
FRACTIONS    = [0.25, 0.50, 0.75, 1.00]

COLOUR = "#DD8452"   # consistent with rest of project figures


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_split(split: str):
    """Load drug features, protein features, and labels for one split."""
    X_drug = torch.tensor(np.load(f"{DATA_ROOT}/drugs/drug_{split}.npy"),    dtype=torch.float32)
    X_prot = torch.tensor(np.load(f"{DATA_ROOT}/proteins/prot_{split}.npy"), dtype=torch.float32)
    name   = "validation" if split == "val" else split
    y      = torch.tensor(
        pd.read_csv(f"{DATA_ROOT}/bindingdb/bindingdb_{name}.csv")["interaction"].values,
        dtype=torch.float32,
    ).unsqueeze(1)
    return X_drug, X_prot, y


def subsample(X_drug, X_prot, y, fraction: float, seed: int = 42):
    """Stratified subsample of the training set at the given fraction."""
    if fraction >= 1.0:
        return X_drug, X_prot, y
    labels = y.squeeze().numpy().astype(int)
    idx, _ = train_test_split(
        np.arange(len(labels)),
        train_size=fraction,
        stratify=labels,
        random_state=seed,
    )
    idx = torch.tensor(idx)
    return X_drug[idx], X_prot[idx], y[idx]


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
def get_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x_d, x_p, y in loader:
        x_d, x_p = x_d.to(device), x_p.to(device)
        all_probs.append(torch.sigmoid(model(x_d, x_p)).cpu().numpy())
        all_labels.append(y.numpy())
    return np.vstack(all_probs).ravel(), np.vstack(all_labels).ravel()


@torch.no_grad()
def val_auc(model, loader, device) -> float:
    probs, labels = get_probs(model, loader, device)
    return roc_auc_score(labels.astype(int), probs)


# ─── Single training run ──────────────────────────────────────────────────────

def train_and_eval(X_drug_tr, X_prot_tr, y_tr,
                   X_drug_val, X_prot_val, y_val,
                   X_drug_te,  X_prot_te,  y_te,
                   device, fraction: float) -> float:
    """Train a fresh model on the given (possibly subsampled) training split and
    return the test AUC-ROC of the best checkpoint."""

    train_loader = make_loader(X_drug_tr, X_prot_tr, y_tr, shuffle=True)
    val_loader   = make_loader(X_drug_val, X_prot_val, y_val, shuffle=False)
    test_loader  = make_loader(X_drug_te,  X_prot_te,  y_te,  shuffle=False)

    model = DTI_DNN(
        drug_input_dim=X_drug_tr.shape[1],
        prot_input_dim=X_prot_tr.shape[1],
        encoder_drop=0.4,
    ).to(device)

    n_pos      = y_tr.sum().item()
    n_neg      = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_auc_score = 0.0
    epochs_no_improve  = 0
    best_state         = None

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        current_val_auc = val_auc(model, val_loader, device)
        scheduler.step()

        if current_val_auc > best_val_auc_score:
            best_val_auc_score = current_val_auc
            best_state         = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve  = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_probs, test_labels = get_probs(model, test_loader, device)
    test_auc = roc_auc_score(test_labels.astype(int), test_probs)
    print(f"    fraction={fraction:.0%}  n={len(y_tr):,}  "
          f"best_val_auc={best_val_auc_score:.4f}  test_auc={test_auc:.4f}")
    return test_auc


# ─── Figure ───────────────────────────────────────────────────────────────────

def _power_law(n, a, b, c):
    """AUC = a - b * n^(-c)  — standard learning curve model."""
    return a - b * np.power(n, -c)


def _project(results: list, proj_fractions: list):
    """Fit a power-law to observed results and extrapolate to proj_fractions."""
    n_full   = results[-1]["n_samples"]   # samples at 100 %
    samples  = np.array([r["n_samples"]   for r in results], dtype=float)
    aucs     = np.array([r["test_auc_roc"] for r in results], dtype=float)

    # Initial guess: asymptote just above current max, decay params
    p0 = (min(aucs[-1] + 0.05, 0.99), 1.0, 0.5)
    bounds = ([aucs[-1], 0, 0], [1.0, np.inf, 2.0])
    try:
        popt, _ = curve_fit(_power_law, samples, aucs, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        print("  Warning: curve_fit did not converge; using linear extrapolation fallback.")
        # simple linear slope from last two points
        slope = (aucs[-1] - aucs[-2]) / (samples[-1] - samples[-2])
        projections = []
        for frac in proj_fractions:
            n = int(n_full * frac)
            auc = min(aucs[-1] + slope * (n - samples[-1]), 0.999)
            projections.append({"fraction": frac, "n_samples": n, "projected_auc_roc": round(float(auc), 6)})
        return projections, None

    projections = []
    for frac in proj_fractions:
        n   = int(n_full * frac)
        auc = min(float(_power_law(n, *popt)), 0.999)
        projections.append({"fraction": frac, "n_samples": n, "projected_auc_roc": round(auc, 6)})

    return projections, popt


def save_learning_curve(results: list, save_path: str,
                        proj_fractions: list | None = None):
    """Plot test AUC-ROC vs number of training samples, with optional projections."""
    fracs   = [r["fraction"]  for r in results]
    samples = [r["n_samples"] for r in results]
    aucs    = [r["test_auc_roc"] for r in results]

    projections, popt = _project(results, proj_fractions or [])

    # ── smooth fitted curve across the full observed + projected range ─────────
    n_full    = results[-1]["n_samples"]
    n_max_proj = projections[-1]["n_samples"] if projections else n_full
    n_smooth  = np.linspace(samples[0], n_max_proj, 300)
    if popt is not None:
        auc_smooth = np.clip(_power_law(n_smooth, *popt), 0, 1)
    else:
        auc_smooth = None

    fig, ax = plt.subplots(figsize=(9, 5))

    # fitted curve (full range, light)
    if auc_smooth is not None:
        # observed region
        mask_obs  = n_smooth <= n_full
        mask_proj = n_smooth >= n_full
        ax.plot(n_smooth[mask_obs],  auc_smooth[mask_obs],
                color=COLOUR, lw=1.2, alpha=0.4, zorder=1)
        ax.plot(n_smooth[mask_proj], auc_smooth[mask_proj],
                color="#888888", lw=1.2, alpha=0.4, linestyle="--", zorder=1)

    # ── observed points ────────────────────────────────────────────────────────
    ax.plot(samples, aucs, marker="o", color=COLOUR, lw=2, markersize=9,
            label="Observed", zorder=3)
    for x, y in zip(samples, aucs):
        ax.annotate(f"{y:.4f}", xy=(x, y),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color=COLOUR, fontweight="bold")

    # ── projected points ───────────────────────────────────────────────────────
    if projections:
        proj_n   = [p["n_samples"]         for p in projections]
        proj_auc = [p["projected_auc_roc"] for p in projections]
        proj_f   = [p["fraction"]          for p in projections]

        # dashed connector from last observed → first projected
        ax.plot([samples[-1]] + proj_n, [aucs[-1]] + proj_auc,
                marker="o", color="#888888", lw=2, markersize=8,
                linestyle="--", label="Projected (power-law fit)", zorder=3,
                markerfacecolor="white", markeredgecolor="#888888", markeredgewidth=2)

        for x, y, f in zip(proj_n, proj_auc, proj_f):
            ax.annotate(f"{y:.4f}", xy=(x, y),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, color="#555555")

        # vertical separator
        ax.axvline(x=n_full, color="steelblue", lw=1.2, linestyle=":",
                   alpha=0.7, label="100 % boundary")

    ax.set_xlabel("Training samples", fontsize=12)
    ax.set_ylabel("Test AUC-ROC", fontsize=12)
    ax.set_title("Learning Curve — Sample Size vs. Test AUC-ROC\n(DTI DNN, Adam + Cosine LR)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")

    # ── secondary x-axis: percentage labels ───────────────────────────────────
    all_samples = samples + ([p["n_samples"] for p in projections] if projections else [])
    all_fracs   = fracs   + ([p["fraction"]  for p in projections] if projections else [])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(all_samples)
    ax2.set_xticklabels([f"{int(f*100)}%" for f in all_fracs], fontsize=9)
    ax2.set_xlabel("Fraction of training data", fontsize=11)

    all_aucs = aucs + ([p["projected_auc_roc"] for p in projections] if projections else [])
    ax.set_ylim(max(0, min(all_aucs) - 0.05), min(1.0, max(all_aucs) + 0.06))
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


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

    print("\nLoading data...")
    X_drug_train_full, X_prot_train_full, y_train_full = load_split("train")
    X_drug_val,        X_prot_val,        y_val        = load_split("val")
    X_drug_test,       X_prot_test,       y_test       = load_split("test")

    print(f"  Full train : {X_drug_train_full.shape[0]:,}")
    print(f"  Val        : {X_drug_val.shape[0]:,}")
    print(f"  Test       : {X_drug_test.shape[0]:,}")

    results = []

    for frac in FRACTIONS:
        print(f"\n{'─'*50}")
        print(f"  Training on {frac:.0%} of training data...")

        X_drug_tr, X_prot_tr, y_tr = subsample(
            X_drug_train_full, X_prot_train_full, y_train_full, frac
        )

        test_auc = train_and_eval(
            X_drug_tr, X_prot_tr, y_tr,
            X_drug_val, X_prot_val, y_val,
            X_drug_test, X_prot_test, y_test,
            device, frac,
        )
        results.append({
            "fraction":    frac,
            "n_samples":   int(len(y_tr)),
            "test_auc_roc": round(test_auc, 6),
        })

    print(f"\n{'='*50}")
    print("LEARNING CURVE RESULTS")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['fraction']:.0%}  ({r['n_samples']:>6,} samples)  →  test AUC-ROC = {r['test_auc_roc']:.4f}")

    PROJ_FRACTIONS = [1.25, 1.50, 1.75, 2.00]
    projections, _ = _project(results, PROJ_FRACTIONS)

    print(f"\n{'='*50}")
    print("PROJECTED AUC-ROC (power-law extrapolation)")
    print(f"{'='*50}")
    for p in projections:
        print(f"  {p['fraction']:.0%}  ({p['n_samples']:>6,} samples)  →  "
              f"projected AUC-ROC = {p['projected_auc_roc']:.4f}")

    output = {"observed": results, "projected": projections}
    json_path = os.path.join(SAVE_DIR, "learning_curve_auc.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    png_path = os.path.join(SAVE_DIR, "learning_curve_auc.png")
    save_learning_curve(results, png_path, proj_fractions=PROJ_FRACTIONS)


if __name__ == "__main__":
    main()