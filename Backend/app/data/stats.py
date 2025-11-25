import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
# Point this to your ALREADY PROCESSED file if you have it,
# or run this after running your dataLoader.
INPUT_CSV = "/Users/drs/Projects/DTI/Backend/app/data/loaded/bindingdb_offline_processed_2025-11-17_20-05.csv"


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


if __name__ == "__main__":
    generate_stats()