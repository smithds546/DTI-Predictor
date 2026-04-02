"""
Grey Area Threshold Experiment -- Step 1: Data Preparation

Processes the raw BindingDB dataset and creates four dataset variants
with grey area thresholds derived from standard deviations around the
mean pAffinity, then generates class distribution figures for the report.

Threshold selection rationale:
  The pAffinity distribution has mean ~6.53 and std ~1.52.  Defining the
  grey area as mean +/- k*std produces symmetric, statistically motivated
  thresholds that yield near-perfect class balance at every k.

Variants:
  - Original:       binder > 7.00, non-binder < 5.30  (asymmetric, current system)
  - Narrow (k=0.25): binder > mean+0.25*std, non-binder < mean-0.25*std
  - Medium (k=0.50): binder > mean+0.50*std, non-binder < mean-0.50*std
  - No grey (k=0):   binder >= median, non-binder < median  (all data retained)

Usage:
    python prepare_data.py

Outputs:
    data/{variant}/drugs/drug_{train,val,test}.npy
    data/{variant}/proteins/prot_{train,val,test}.npy
    data/{variant}/bindingdb/bindingdb_{train,validation,test}.csv
    figures/paffinity_distribution.png   (plain histogram with median)
    figures/class_distribution.png
    figures/paffinity_thresholds.png
    figures/dataset_summary.json
"""

import os
import sys
import re
import json
import hashlib
import zipfile
import io
import gc
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
RDLogger.DisableLog("rdApp.*")

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "data"))
RAW_DIR    = os.path.join(DATA_BASE, "raw")
PROT_CACHE = os.path.join(DATA_BASE, "preprocessed", "protein_cache")
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")
FULL_CACHE = os.path.join(DATA_DIR, "full_dataset.csv")

TSV_ZIP    = "BindingDB_All_202511_tsv.zip"
FASTA_FILE = "BindingDBTargetSequences.fasta"
CHUNK_SIZE = 200_000

# Fraction of the full dataset to use (1.0 = all data).
# Reduce to 0.1 or 0.02 if you run into memory issues.
SAMPLE_FRAC = 1.0

# Variants are built dynamically in main() from the data's mean and std.
# These are populated after loading the full dataset.
VARIANTS = {}   # filled by build_variants()
LABELS   = {}
COLOURS  = {
    "original": "#4C72B0",
    "narrow":   "#DD8452",
    "medium":   "#55A868",
    "no_grey":  "#8172B2",
}
VARIANT_ORDER = ["original", "narrow", "medium", "no_grey"]


def build_variants(pa: pd.Series):
    """Compute threshold variants from the pAffinity distribution.

    - original: the current system's asymmetric thresholds
    - narrow (k=0.25): mean +/- 0.25*std  (~51/49 balance)
    - medium (k=0.50): mean +/- 0.50*std  (~50/50 balance)
    - no_grey (k=0):   median split, all data retained
    """
    mean = pa.mean()
    std  = pa.std()
    med  = pa.median()

    global VARIANTS, LABELS

    VARIANTS = {
        "original": {"binder": 7.00, "nonbinder": 5.30, "k": None},
        "narrow":   {"binder": round(mean + 0.25 * std, 2),
                     "nonbinder": round(mean - 0.25 * std, 2),
                     "k": 0.25},
        "medium":   {"binder": round(mean + 0.50 * std, 2),
                     "nonbinder": round(mean - 0.50 * std, 2),
                     "k": 0.50},
        "no_grey":  {"binder": round(med, 4),
                     "nonbinder": round(med, 4),
                     "k": 0},
    }

    LABELS = {
        "original": "Original (5.30\u20137.00)",
        "narrow":   f"k=0.25 ({VARIANTS['narrow']['nonbinder']:.2f}"
                    f"\u2013{VARIANTS['narrow']['binder']:.2f})",
        "medium":   f"k=0.50 ({VARIANTS['medium']['nonbinder']:.2f}"
                    f"\u2013{VARIANTS['medium']['binder']:.2f})",
        "no_grey":  f"No grey (median={med:.2f})",
    }

    print(f"\n  pAffinity:  mean={mean:.4f}  std={std:.4f}  median={med:.4f}")
    for v in VARIANT_ORDER:
        cfg = VARIANTS[v]
        print(f"  {LABELS[v]:<36}  binder >{cfg['binder']:.2f}  "
              f"non-binder <{cfg['nonbinder']:.2f}")

    return mean, std, med


# ─── Helpers (mirrored from dataLoader.py) ────────────────────────────────────

def parse_fasta(path):
    sequences, cur_id, cur_seq = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id and cur_seq:
                    sequences[cur_id] = "".join(cur_seq)
                    cur_seq = []
                header = re.sub(r"mol:protein length:\d+\s*", "", line[1:].strip())
                cur_id = re.sub(r"^p\d+\s*", "", header).strip()
            elif cur_id:
                cur_seq.append(line)
    if cur_id and cur_seq:
        sequences[cur_id] = "".join(cur_seq)
    return sequences


def clean_affinity(val):
    if pd.isna(val):
        return None
    val = str(val).strip().replace(">", "").replace("<", "").replace("=", "").replace("~", "")
    try:
        return float(val)
    except ValueError:
        return None


def normalize_drug_name(name):
    if name is None or pd.isna(name):
        return None
    name = str(name).strip()
    if not name:
        return None
    parts = sorted([p.strip() for p in name.split("::") if p.strip()], key=len)
    for p in parts:
        if not re.search(r"[=#\[\]\(\)]", p):
            return p
    return parts[0] if parts else None


def seq_hash(seq):
    return hashlib.md5(seq.encode("utf-8")).hexdigest()[:12]


def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(list(fp.ToBitString()), dtype=np.int8)
    return np.zeros(167, dtype=np.int8)


# ─── Dataset loading ─────────────────────────────────────────────────────────

def load_full_dataset():
    """Load all BindingDB interactions (no grey area removal).

    Caches the result (without protein sequences) so subsequent runs
    skip the slow zip processing.  Protein sequences are re-joined
    from the FASTA each time (fast).
    """
    fasta_path = os.path.join(RAW_DIR, FASTA_FILE)
    seq_dict = parse_fasta(fasta_path)
    print(f"FASTA: {len(seq_dict):,} protein sequences")

    if os.path.exists(FULL_CACHE):
        print(f"Loading cached dataset from {FULL_CACHE}...")
        df = pd.read_csv(FULL_CACHE)
        df["protein_sequence"] = df["Target_name"].map(seq_dict)
        df.dropna(subset=["protein_sequence"], inplace=True)
        print(f"  {len(df):,} rows")
        return df

    print("Processing raw BindingDB zip (this takes ~5-10 minutes)...")
    t0 = time.time()
    zip_path = os.path.join(RAW_DIR, TSV_ZIP)
    chunks = []

    with zipfile.ZipFile(zip_path) as z:
        tsv_name = [n for n in z.namelist() if n.endswith(".tsv")][0]
        with z.open(tsv_name) as f:
            reader = pd.read_csv(
                io.TextIOWrapper(f, encoding="utf-8"),
                sep="\t", on_bad_lines="skip", low_memory=False,
                chunksize=CHUNK_SIZE,
            )
            for i, chunk in enumerate(reader):
                print(f"  Chunk {i + 1}...", flush=True)

                # Species filter
                sp_cols = [
                    "Target Source Organism According to Curator or DataSource",
                    "Target Source Organism", "Target Species",
                    "Target Organism", "Target Species Name", "Organism",
                ]
                sp_col = next((c for c in sp_cols if c in chunk.columns), None)
                if sp_col:
                    chunk = chunk[
                        chunk[sp_col].str.contains("Homo sapiens", case=False, na=False)
                    ].copy()

                # Affinity extraction
                aff_srcs = ["Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)"]
                existing = [s for s in aff_srcs if s in chunk.columns]
                if not existing:
                    continue
                df_c = chunk.dropna(how="all", subset=existing).copy()
                df_c["aff_raw"] = None
                for src in existing:
                    df_c["aff_raw"] = df_c["aff_raw"].fillna(df_c[src])

                # Target & drug columns
                tgt_cols = [
                    "Target Name Assigned by Curator or DataSource", "Target Name",
                    "UniProt (Primary ID of Target)", "Target UniProt ID",
                ]
                tgt_col = next((c for c in tgt_cols if c in df_c.columns), None)
                dn_col = ("BindingDB Ligand Name"
                          if "BindingDB Ligand Name" in df_c.columns else None)

                if not tgt_col or "Ligand SMILES" not in df_c.columns:
                    continue

                keep = ["Ligand SMILES", tgt_col, "aff_raw"]
                if dn_col:
                    keep.insert(1, dn_col)
                df_c = df_c[keep].copy()
                if dn_col:
                    df_c.columns = ["drug_smiles", "drug_name", "Target_name", "aff_raw"]
                else:
                    df_c.columns = ["drug_smiles", "Target_name", "aff_raw"]
                    df_c["drug_name"] = None

                # Clean affinity values
                df_c["affinity_nm"] = df_c["aff_raw"].apply(clean_affinity)
                df_c.dropna(subset=["affinity_nm", "drug_smiles"], inplace=True)
                df_c = df_c[df_c["affinity_nm"] > 0].copy()
                df_c["p_affinity"] = -np.log10(df_c["affinity_nm"] * 1e-9)

                # Normalize drug name
                df_c["drug_name"] = df_c["drug_name"].apply(normalize_drug_name)
                df_c["drug_name"] = df_c["drug_name"].fillna(df_c["drug_smiles"])

                df_c = df_c[["drug_name", "drug_smiles", "Target_name", "p_affinity"]]
                if not df_c.empty:
                    chunks.append(df_c)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Raw rows (species-filtered, valid affinity): {len(df):,}")

    # Map protein sequences
    df["protein_sequence"] = df["Target_name"].map(seq_dict)
    df.dropna(subset=["protein_sequence"], inplace=True)
    print(f"  With valid sequences: {len(df):,}")

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  After dedup: {len(df):,}")

    # Keep only proteins with cached ProtBERT embeddings
    cached = {
        os.path.splitext(f)[0]
        for f in os.listdir(PROT_CACHE)
        if f.endswith(".npy")
    }
    mask = df["protein_sequence"].apply(lambda s: seq_hash(s) in cached)
    n_before = len(df)
    df = df[mask].reset_index(drop=True)
    print(f"  With cached ProtBERT: {len(df):,}  "
          f"(filtered {n_before - len(df):,} rows without embeddings)")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed / 60:.1f} min")

    # Cache without protein_sequence (re-joined from FASTA on load)
    os.makedirs(DATA_DIR, exist_ok=True)
    df[["drug_name", "drug_smiles", "Target_name", "p_affinity"]].to_csv(
        FULL_CACHE, index=False
    )
    print(f"  Cached to {FULL_CACHE}")
    return df


# ─── Feature caches ──────────────────────────────────────────────────────────

def build_maccs_cache(unique_smiles):
    """Compute MACCS fingerprints for all unique SMILES strings."""
    print(f"Computing MACCS for {len(unique_smiles):,} unique drugs...")
    cache = {}
    n_fail = 0
    for i, smi in enumerate(unique_smiles):
        if (i + 1) % 100_000 == 0:
            print(f"  {i + 1:,} / {len(unique_smiles):,}", flush=True)
        fp = smiles_to_maccs(smi)
        if fp.sum() == 0:
            n_fail += 1
        cache[smi] = fp
    if n_fail:
        print(f"  Warning: {n_fail:,} SMILES failed to parse (zeros used)")
    return cache


def build_prot_cache(df):
    """Load ProtBERT embeddings from disk cache for all unique proteins."""
    unique = df[["Target_name", "protein_sequence"]].drop_duplicates()
    print(f"Loading ProtBERT embeddings for {len(unique):,} unique proteins...")
    cache = {}
    for _, row in unique.iterrows():
        h = seq_hash(row["protein_sequence"])
        emb = np.load(os.path.join(PROT_CACHE, f"{h}.npy")).astype(np.float32)
        if emb.ndim == 2:
            emb = emb.squeeze(0)
        cache[row["Target_name"]] = emb
    return cache


# ─── Per-variant preparation ─────────────────────────────────────────────────

def prepare_variant(df_full, maccs_cache, prot_cache, name, config):
    binder_t = config["binder"]
    nonbinder_t = config["nonbinder"]

    df = df_full.copy()
    n_total = len(df)

    if name == "no_grey":
        # Median split: everything >= median is binder, < median is non-binder
        median_val = binder_t  # binder_t == nonbinder_t == median
        df["interaction"] = (df["p_affinity"] >= median_val).astype(int)
        n_binder = int((df["interaction"] == 1).sum())
        n_nonbinder = int((df["interaction"] == 0).sum())
        n_grey = 0
    else:
        df["interaction"] = -1
        df.loc[df["p_affinity"] > binder_t, "interaction"] = 1
        df.loc[df["p_affinity"] < nonbinder_t, "interaction"] = 0
        n_binder = int((df["interaction"] == 1).sum())
        n_nonbinder = int((df["interaction"] == 0).sum())
        n_grey = int((df["interaction"] == -1).sum())
        # Remove grey area
        df = df[df["interaction"].isin([0, 1])].reset_index(drop=True)

    n_retained = len(df)
    binder_pct = n_binder / n_retained * 100

    print(f"\n  {LABELS[name]}")
    print(f"    Binders (>{binder_t}):      {n_binder:>9,}  ({n_binder/n_total*100:5.1f}%)")
    print(f"    Non-binders (<{nonbinder_t}): {n_nonbinder:>9,}  ({n_nonbinder/n_total*100:5.1f}%)")
    print(f"    Grey area (removed):  {n_grey:>9,}  ({n_grey/n_total*100:5.1f}%)")
    print(f"    Retained:             {n_retained:>9,}  "
          f"({binder_pct:.1f}% binder / {100-binder_pct:.1f}% non-binder)")

    # Split
    train_df, temp_df = train_test_split(
        df, test_size=0.35, random_state=42, shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=15 / 35, random_state=42, shuffle=True
    )

    # Save per split
    variant_dir = os.path.join(DATA_DIR, name)
    for sub in ("drugs", "proteins", "bindingdb"):
        os.makedirs(os.path.join(variant_dir, sub), exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X_drug = np.array(
            [maccs_cache[s] for s in split_df["drug_smiles"]], dtype=np.int8
        )
        X_prot = np.array(
            [prot_cache[t] for t in split_df["Target_name"]], dtype=np.float32
        )

        np.save(os.path.join(variant_dir, "drugs", f"drug_{split_name}.npy"), X_drug)
        np.save(os.path.join(variant_dir, "proteins", f"prot_{split_name}.npy"), X_prot)

        csv_name = f"bindingdb_{'validation' if split_name == 'val' else split_name}.csv"
        split_df.to_csv(
            os.path.join(variant_dir, "bindingdb", csv_name), index=False
        )

        del X_drug, X_prot
        gc.collect()

    print(f"    Saved to {variant_dir}/  "
          f"(train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,})")

    return {
        "total": n_total,
        "binder": n_binder,
        "nonbinder": n_nonbinder,
        "grey": n_grey,
        "retained": n_retained,
        "binder_pct": round(binder_pct, 2),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }


# ─── Figures ─────────────────────────────────────────────────────────────────

def plot_class_distribution(summary, save_path):
    variants = [v for v in VARIANT_ORDER if v in summary]
    n = len(variants)
    x = np.arange(n)
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # (a) Absolute counts
    binders    = [summary[v]["binder"] for v in variants]
    nonbinders = [summary[v]["nonbinder"] for v in variants]
    greys      = [summary[v]["grey"] for v in variants]

    b1 = ax1.bar(x - w, binders,    w, label="Binders",     color="#55A868", edgecolor="white")
    b2 = ax1.bar(x,     nonbinders, w, label="Non-binders", color="#C44E52", edgecolor="white")
    b3 = ax1.bar(x + w, greys,      w, label="Grey area (removed)", color="#AAAAAA", edgecolor="white")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, h,
                         f"{h / 1000:.0f}K", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[v] for v in variants], fontsize=9, rotation=15, ha="right")
    ax1.set_ylabel("Number of Pairs", fontsize=11)
    ax1.set_title("(a) Class Counts by Threshold", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # (b) Class balance
    pcts = [summary[v]["binder_pct"] for v in variants]
    bars = ax2.bar(x, pcts, 0.5,
                   color=[COLOURS[v] for v in variants], edgecolor="white")
    ax2.axhline(50, color="black", ls="--", lw=1, alpha=0.5, label="Perfect balance (50%)")
    for bar, pct in zip(bars, pcts):
        ax2.text(bar.get_x() + bar.get_width() / 2, pct + 1,
                 f"{pct:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[v] for v in variants], fontsize=9, rotation=15, ha="right")
    ax2.set_ylabel("Binder Percentage (%)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.set_title("(b) Class Balance (Retained Data)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_paffinity_plain(df, median_val, mean_val, std_val, save_path):
    """Plain pAffinity histogram with median line — no threshold overlays."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df["p_affinity"].clip(0, 14), bins=120,
            color="#8DA0CB", edgecolor="white", alpha=0.85)

    # Median
    ax.axvline(median_val, color="#E41A1C", ls="-", lw=2, zorder=5,
               label=f"Median = {median_val:.2f}")
    # Mean +/- 1 std reference
    ax.axvline(mean_val, color="#333333", ls=":", lw=1.5, zorder=4,
               label=f"Mean = {mean_val:.2f}")
    ax.axvspan(mean_val - std_val, mean_val + std_val,
               alpha=0.08, color="#333333", zorder=1,
               label=f"\u00b11 std ({mean_val - std_val:.2f}\u2013{mean_val + std_val:.2f})")

    ax.set_xlabel(r"pAffinity ($-\log_{10}$ $K_d$/$K_i$ in M)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("pAffinity Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_paffinity_thresholds(df, save_path):
    """pAffinity histogram with grey area threshold bands overlaid."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df["p_affinity"].clip(0, 14), bins=120,
            color="#CCCCCC", edgecolor="white", alpha=0.9, zorder=1)

    # Only show variants that have a grey area (skip no_grey)
    for name in VARIANT_ORDER:
        if name not in VARIANTS or name == "no_grey":
            continue
        cfg = VARIANTS[name]
        c = COLOURS[name]
        lo, hi = cfg["nonbinder"], cfg["binder"]
        ax.axvspan(lo, hi, alpha=0.12, color=c, zorder=2)
        ax.axvline(lo, color=c, ls="--", lw=1.5, zorder=3)
        ax.axvline(hi, color=c, ls="--", lw=1.5, zorder=3,
                   label=f"{LABELS[name]}")

    ax.set_xlabel(r"pAffinity ($-\log_{10}$ $K_d$/$K_i$ in M)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("pAffinity Distribution with Grey Area Thresholds",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load full dataset
    df_full = load_full_dataset()

    if SAMPLE_FRAC < 1.0:
        df_full = df_full.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
        print(f"\nSampled {SAMPLE_FRAC * 100:.0f}%: {len(df_full):,} rows")

    # 2. Compute variants from data statistics
    pa_mean, pa_std, pa_median = build_variants(df_full["p_affinity"])

    # 3. Pre-compute feature caches
    maccs_cache = build_maccs_cache(df_full["drug_smiles"].unique())
    prot_cache  = build_prot_cache(df_full)

    # 4. Prepare each variant
    print(f"\n{'=' * 60}")
    print("PREPARING DATASET VARIANTS")
    print(f"{'=' * 60}")

    summary = {}
    for name in VARIANT_ORDER:
        summary[name] = prepare_variant(
            df_full, maccs_cache, prot_cache, name, VARIANTS[name]
        )

    # Store distribution stats in summary for reproducibility
    summary["_stats"] = {
        "mean": round(pa_mean, 4),
        "std": round(pa_std, 4),
        "median": round(pa_median, 4),
    }

    # 5. Generate figures
    print(f"\n{'=' * 60}")
    print("GENERATING FIGURES")
    print(f"{'=' * 60}")

    plot_paffinity_plain(
        df_full, pa_median, pa_mean, pa_std,
        os.path.join(FIG_DIR, "paffinity_distribution.png"),
    )
    plot_class_distribution(
        summary, os.path.join(FIG_DIR, "class_distribution.png")
    )
    plot_paffinity_thresholds(
        df_full, os.path.join(FIG_DIR, "paffinity_thresholds.png")
    )

    with open(os.path.join(FIG_DIR, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {os.path.join(FIG_DIR, 'dataset_summary.json')}")

    # 6. Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  pAffinity:  mean={pa_mean:.4f}  std={pa_std:.4f}  median={pa_median:.4f}")
    print()
    print(f"{'Variant':<36} {'Total':>10} {'Binder':>10} {'Non-bind':>10} "
          f"{'Grey':>10} {'Retained':>10} {'Bal%':>7}")
    print("-" * 97)
    for v in VARIANT_ORDER:
        s = summary[v]
        print(f"{LABELS[v]:<36} {s['total']:>10,} {s['binder']:>10,} "
              f"{s['nonbinder']:>10,} {s['grey']:>10,} {s['retained']:>10,} "
              f"{s['binder_pct']:>6.1f}%")

    print(f"\nNext step: python train_and_compare.py")


if __name__ == "__main__":
    main()
