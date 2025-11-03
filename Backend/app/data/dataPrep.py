"""
Complete data preparation pipeline
Downloads BindingDB datasets and preprocesses them for model training
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataLoader import BindingDBLoader, explore_dataset
from data.preprocess import DTIDataset


def prepare_complete_pipeline(
        data_dir: str = "Backend/app/data",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
):
    """
    Complete data preparation pipeline

    Args:
        data_dir: Directory for data storage
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility
    """

    print("=" * 70)
    print(" DTI PREDICTION - DATA PREPARATION PIPELINE")
    print("=" * 70)

    # Step 1: Download and load datasets
    print("\n[STEP 1/4] Downloading BindingDB datasets...")
    loader = BindingDBLoader(data_dir=f"{data_dir}/raw")
    combined_df = loader.load_both_datasets()

    # Step 2: Explore dataset
    print("\n[STEP 2/4] Exploring dataset...")
    explore_dataset(combined_df)

    # Save raw combined dataset
    raw_csv_path = loader.save_processed_dataset(combined_df, 'bindingdb_combined.csv')

    # Step 3: Preprocess features
    print("\n[STEP 3/4] Preprocessing features...")
    dataset = DTIDataset(data_dir=f"{data_dir}/raw")

    drug_features, protein_features, labels = dataset.preprocess_dataset(
        combined_df,
        save_path=None  # We'll save after splitting
    )

    # Step 4: Split into train/val/test sets
    print(f"\n[STEP 4/4] Splitting data (test={test_size}, val={val_size})...")

    # First split: separate test set
    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Second split: separate validation from training
    train_labels = labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_state,
        stratify=train_labels
    )

    # Create splits
    splits = {
        'train': {
            'drug_features': drug_features[train_idx],
            'protein_features': protein_features[train_idx],
            'labels': labels[train_idx]
        },
        'val': {
            'drug_features': drug_features[val_idx],
            'protein_features': protein_features[val_idx],
            'labels': labels[val_idx]
        },
        'test': {
            'drug_features': drug_features[test_idx],
            'protein_features': protein_features[test_idx],
            'labels': labels[test_idx]
        }
    }

    # Save processed data
    processed_dir = Path(f"{data_dir}/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        save_path = processed_dir / f"{split_name}_data.npz"
        np.savez(
            save_path,
            drug_features=split_data['drug_features'],
            protein_features=split_data['protein_features'],
            labels=split_data['labels']
        )
        print(f"✓ Saved {split_name} split to {save_path}")
        print(f"  - Samples: {len(split_data['labels'])}")
        print(f"  - Positive rate: {split_data['labels'].mean() * 100:.2f}%")

    # Save full dataset as well
    full_save_path = processed_dir / "full_data.npz"
    np.savez(
        full_save_path,
        drug_features=drug_features,
        protein_features=protein_features,
        labels=labels
    )
    print(f"\n✓ Saved full dataset to {full_save_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(" DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(labels)}")
    print(f"   Training samples: {len(splits['train']['labels'])}")
    print(f"   Validation samples: {len(splits['val']['labels'])}")
    print(f"   Test samples: {len(splits['test']['labels'])}")
    print(f"\n📁 Files saved in: {processed_dir}")
    print(f"\n🎯 Next step: Build and train your neural network!")
    print("=" * 70)

    return splits


def load_processed_data(data_dir: str = "Backend/app/data/processed", split: str = "train"):
    """
    Load preprocessed data splits

    Args:
        data_dir: Directory containing processed data
        split: Which split to load ('train', 'val', 'test', or 'full')

    Returns:
        Tuple of (drug_features, protein_features, labels)
    """
    filepath = Path(data_dir) / f"{split}_data.npz"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. "
            f"Please run prepare_complete_pipeline() first."
        )

    data = np.load(filepath)
    return data['drug_features'], data['protein_features'], data['labels']


if __name__ == "__main__":
    # Run the complete pipeline
    splits = prepare_complete_pipeline()

    # Example: Load training data
    print("\n" + "=" * 70)
    print("Testing data loading...")
    drug_feat, protein_feat, labels = load_processed_data(split='train')
    print(f"✓ Successfully loaded training data")
    print(f"  Drug features shape: {drug_feat.shape}")
    print(f"  Protein features shape: {protein_feat.shape}")
    print(f"  Labels shape: {labels.shape}")
    print("=" * 70)