import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# === CONFIG ===
# Point this to your ALREADY PROCESSED file if you have it,
# or run this after running your dataLoader.
INPUT_CSV = "/Users/drs/Projects/DTI/Backend/app/data/loaded/bindingdb_offline_processed_2025-11-17_20-05.csv"
RAW_INPUT_CSV = "/Users/drs/Projects/DTI/Backend/app/data/raw/BindingDB_All_202511_tsv.zip"


def load_raw_bindingdb(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            tsv_name = [f for f in z.namelist() if f.endswith(".tsv")][0]
            print("Extracting", tsv_name)
            return pd.read_csv(
                z.open(tsv_name),
                sep="\t",
                engine="python",
                on_bad_lines="skip",
                nrows=50000
            )
    else:
        return pd.read_csv(
            path,
            sep="\t",
            engine="python",
            on_bad_lines="skip"
        )


def generate_stats():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("Error: processed CSV not found. Run your dataLoader first.")
        return

    print("-" * 30)
    print("FOR TABLE 1 (Attrition) - Note: You might need to estimate RAW count from file size or logs")
    print(f"Final Count (after grey area removal): {len(df)}")

    print("-" * 30)
    print("FOR TABLE 2 (Class Distribution)")
    pos_count = len(df[df['interaction'] == 1])
    neg_count = len(df[df['interaction'] == 0])
    total = len(df)

    print(f"Binders (1): {pos_count} ({pos_count / total * 100:.2f}%)")
    print(f"Non-Binders (0): {neg_count} ({neg_count / total * 100:.2f}%)")
    print(f"Total: {total}")

    print("-" * 30)
    print("FOR TABLE 4 (Splitting - 65/20/15)")
    print(f"Train (65%): {int(total * 0.65)}")
    print(f"Validation (20%): {int(total * 0.20)}")
    print(f"Test (15%): {int(total * 0.15)}")

    # === GENERATE PLOT FOR FIGURE 1 ===
    plt.figure(figsize=(10, 6))

    # We plot the p_affinity.
    # Note: Your processed CSV might only have 0 and 1 interactions.
    # If you want the full histogram including the gap, you technically need the data
    # BEFORE you dropped the grey area.
    # If you only have the final file, the histogram will show two distinct islands.

    sns.histplot(data=df, x='p_affinity', bins=50, kde=True, color='blue')

    # Add lines for thresholds
    plt.axvline(x=5.3, color='red', linestyle='--', linewidth=2, label='Non-Binder Threshold (5.3)')
    plt.axvline(x=7.0, color='green', linestyle='--', linewidth=2, label='Binder Threshold (7.0)')

    # Shade the grey area (visually representing what was removed/separated)
    plt.axvspan(5.3, 7.0, color='red', alpha=0.1, label='Grey Area (Excluded)')

    plt.title("Distribution of Binding Affinities (pAffinity)")
    plt.xlabel("pAffinity (-log10)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("affinity_distribution.png")
    print("\nGraph saved as 'affinity_distribution.png'")


def to_paffinity(value_nm):
    try:
        value_nm = float(value_nm)
        return -np.log10(value_nm * 1e-9)
    except:
        return np.nan


def corr_heatmap():
    try:
        df = load_raw_bindingdb(RAW_INPUT_CSV)
    except Exception as e:
        print("Error reading raw TSV:", e)
        return

    # === GENERATE CORRELATION HEATMAP FOR AFFINITY TYPES (Ki, Kd, IC50, EC50) ===

    affinity_candidates = {
        "Ki": ["Ki (nM)", "KI (nM)", "Ki", "Ki_nM"],
        "Kd": ["Kd (nM)", "KD (nM)", "Kd", "Kd_nM"],
        "IC50": ["IC50 (nM)", "IC50", "IC50_nM"],
        "EC50": ["EC50 (nM)", "EC50", "EC50_nM"]
    }

    col_map = {}
    for key, candidates in affinity_candidates.items():
        for c in candidates:
            if c in df.columns:
                col_map[key] = c
                break

    if not col_map:
        print("\nNo affinity columns found for correlation heatmap.")
        return

    print("\nGenerating correlation heatmap for affinity types:", col_map)

    # Convert nM → pAffinity
    sub = pd.DataFrame()
    for aff_type, col in col_map.items():
        sub[aff_type] = df[col].apply(to_paffinity)

    corr = sub.corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        linewidths=0.5,
        square=True,
        vmin=-1,
        vmax=1
    )
    plt.title("Correlation Between Affinity Types (pAffinity Scale)")
    plt.tight_layout()
    plt.savefig("affinity_correlation_heatmap.png", dpi=300)
    print("Heatmap saved as 'affinity_correlation_heatmap.png'")


if __name__ == "__main__":
    generate_stats()
    corr_heatmap()