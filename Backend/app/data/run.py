"""
Simple script to download and prepare BindingDB datasets
Run this file to get started with your DTI project!
"""

import sys
from pathlib import Path

# Ensure proper imports
sys.path.append(str(Path(__file__).parent.parent))

from data.prepare_data import prepare_complete_pipeline

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║   Drug-Target Interaction Predictor - Data Preparation        ║
    ║   BindingDB Kd + Ki Datasets                                   ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    print("\nThis script will:")
    print("  1. Download Davis (Kd) and KIBA (Ki) benchmark datasets")
    print("  2. Preprocess drug molecules (SMILES -> fingerprints)")
    print("  3. Preprocess protein sequences (sequences -> encodings)")
    print("  4. Split data into train/validation/test sets")
    print("  5. Save processed data for model training")

    response = input("\nProceed? (y/n): ").strip().lower()

    if response == 'y':
        try:
            prepare_complete_pipeline()
            print("\n✅ Success! Your data is ready for model training.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTroubleshooting:")
            print("  1. Check your internet connection")
            print("  2. Ensure you have installed all requirements: pip install -r requirements.txt")
            print("  3. Make sure you have write permissions in the data directory")
    else:
        print("\nCancelled.")