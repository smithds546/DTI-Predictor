"""
BindingDB dataset loader for DTI prediction
Downloads and preprocesses BindingDB KD and KI datasets
"""

import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
from typing import Tuple, Optional
import zipfile
import io


class BindingDBLoader:
    """Download and load BindingDB datasets"""

    # BindingDB benchmark datasets (these are commonly used benchmark datasets)
    DATASETS = {
        'davis': {
            'url': 'https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/',
            'files': ['ligands_can.txt', 'proteins.txt', 'Y', 'folds/train_fold_setting1.txt']
        },
        'kiba': {
            'url': 'https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/',
            'files': ['ligands_can.txt', 'proteins.txt', 'Y', 'folds/train_fold_setting1.txt']
        }
    }

    def __init__(self, data_dir: str = "Backend/app/data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, filepath: Path) -> bool:
        """Download file from URL"""
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded to {filepath}")
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def download_davis_dataset(self) -> bool:
        """Download Davis Kd dataset"""
        print("\n=== Downloading Davis Dataset (Kd values) ===")
        dataset_dir = self.data_dir / 'davis'
        dataset_dir.mkdir(parents=True, exist_ok=True)

        base_url = self.DATASETS['davis']['url']

        # Download required files
        files = {
            'ligands': 'ligands_can.txt',
            'proteins': 'proteins.txt',
            'affinities': 'Y'
        }

        success = True
        for key, filename in files.items():
            url = base_url + filename
            filepath = dataset_dir / filename
            if not filepath.exists():
                if not self.download_file(url, filepath):
                    success = False
            else:
                print(f"{filepath} already exists, skipping download")

        return success

    def download_kiba_dataset(self) -> bool:
        """Download KIBA Ki dataset"""
        print("\n=== Downloading KIBA Dataset (Ki values) ===")
        dataset_dir = self.data_dir / 'kiba'
        dataset_dir.mkdir(parents=True, exist_ok=True)

        base_url = self.DATASETS['kiba']['url']

        files = {
            'ligands': 'ligands_can.txt',
            'proteins': 'proteins.txt',
            'affinities': 'Y'
        }

        success = True
        for key, filename in files.items():
            url = base_url + filename
            filepath = dataset_dir / filename
            if not filepath.exists():
                if not self.download_file(url, filepath):
                    success = False
            else:
                print(f"{filepath} already exists, skipping download")

        return success

    def load_davis_dataset(self) -> pd.DataFrame:
        """
        Load Davis Kd dataset into DataFrame

        Returns:
            DataFrame with columns: drug_id, drug_smiles, protein_id, protein_sequence, affinity, interaction
        """
        dataset_dir = self.data_dir / 'davis'

        if not dataset_dir.exists():
            print("Davis dataset not found. Downloading...")
            self.download_davis_dataset()

        # Load ligands (drugs)
        with open(dataset_dir / 'ligands_can.txt', 'r') as f:
            ligands = [line.strip() for line in f.readlines()]

        # Load proteins
        with open(dataset_dir / 'proteins.txt', 'r') as f:
            proteins = [line.strip() for line in f.readlines()]

        # Load affinity matrix
        affinities = []
        with open(dataset_dir / 'Y', 'r') as f:
            for line in f:
                affinities.append([float(x) for x in line.strip().split()])
        affinity_matrix = np.array(affinities)

        # Create DataFrame
        data = []
        for i, ligand in enumerate(ligands):
            for j, protein in enumerate(proteins):
                affinity = affinity_matrix[i, j]

                # Skip missing values (usually represented as very large numbers or NaN)
                if affinity > 10 or np.isnan(affinity):  # Davis uses pKd values (0-10 range)
                    continue

                # Convert to binary interaction (pKd > 7.0 is considered active binding)
                interaction = 1 if affinity > 7.0 else 0

                data.append({
                    'drug_id': f'drug_{i}',
                    'drug_smiles': ligand,
                    'protein_id': f'protein_{j}',
                    'protein_sequence': protein,
                    'affinity': affinity,
                    'interaction': interaction,
                    'dataset': 'davis_kd'
                })

        df = pd.DataFrame(data)
        print(f"\nLoaded Davis dataset: {len(df)} interactions")
        print(f"Positive interactions: {df['interaction'].sum()} ({df['interaction'].mean() * 100:.2f}%)")
        print(f"Unique drugs: {df['drug_id'].nunique()}")
        print(f"Unique proteins: {df['protein_id'].nunique()}")

        return df

    def load_kiba_dataset(self) -> pd.DataFrame:
        """
        Load KIBA Ki dataset into DataFrame

        Returns:
            DataFrame with columns: drug_id, drug_smiles, protein_id, protein_sequence, affinity, interaction
        """
        dataset_dir = self.data_dir / 'kiba'

        if not dataset_dir.exists():
            print("KIBA dataset not found. Downloading...")
            self.download_kiba_dataset()

        # Load ligands (drugs)
        with open(dataset_dir / 'ligands_can.txt', 'r') as f:
            ligands = [line.strip() for line in f.readlines()]

        # Load proteins
        with open(dataset_dir / 'proteins.txt', 'r') as f:
            proteins = [line.strip() for line in f.readlines()]

        # Load affinity matrix
        affinities = []
        with open(dataset_dir / 'Y', 'r') as f:
            for line in f:
                affinities.append([float(x) for x in line.strip().split()])
        affinity_matrix = np.array(affinities)

        # Create DataFrame
        data = []
        for i, ligand in enumerate(ligands):
            for j, protein in enumerate(proteins):
                affinity = affinity_matrix[i, j]

                # Skip missing values (KIBA scores typically range 0-17)
                if affinity > 17 or np.isnan(affinity):
                    continue

                # Convert to binary interaction (KIBA < 12.1 is considered active)
                interaction = 1 if affinity < 12.1 else 0

                data.append({
                    'drug_id': f'drug_{i}',
                    'drug_smiles': ligand,
                    'protein_id': f'protein_{j}',
                    'protein_sequence': protein,
                    'affinity': affinity,
                    'interaction': interaction,
                    'dataset': 'kiba_ki'
                })

        df = pd.DataFrame(data)
        print(f"\nLoaded KIBA dataset: {len(df)} interactions")
        print(f"Positive interactions: {df['interaction'].sum()} ({df['interaction'].mean() * 100:.2f}%)")
        print(f"Unique drugs: {df['drug_id'].nunique()}")
        print(f"Unique proteins: {df['protein_id'].nunique()}")

        return df

    def load_both_datasets(self) -> pd.DataFrame:
        """Load both Davis (Kd) and KIBA (Ki) datasets and combine them"""
        print("\n" + "=" * 60)
        print("Loading BindingDB benchmark datasets (Davis Kd + KIBA Ki)")
        print("=" * 60)

        davis_df = self.load_davis_dataset()
        kiba_df = self.load_kiba_dataset()

        # Combine datasets
        combined_df = pd.concat([davis_df, kiba_df], ignore_index=True)

        print(f"\n=== Combined Dataset Statistics ===")
        print(f"Total interactions: {len(combined_df)}")
        print(f"Davis (Kd) interactions: {len(davis_df)}")
        print(f"KIBA (Ki) interactions: {len(kiba_df)}")
        print(f"Total unique drugs: {combined_df['drug_smiles'].nunique()}")
        print(f"Total unique proteins: {combined_df['protein_sequence'].nunique()}")
        print(f"Overall positive rate: {combined_df['interaction'].mean() * 100:.2f}%")

        return combined_df

    def save_processed_dataset(self, df: pd.DataFrame, filename: str = 'bindingdb_combined.csv'):
        """Save processed dataset to CSV"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"\nSaved combined dataset to {filepath}")
        return filepath


def explore_dataset(df: pd.DataFrame):
    """Explore dataset statistics"""
    print("\n" + "=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    print("\n--- Basic Statistics ---")
    print(f"Total samples: {len(df)}")
    print(f"Positive interactions: {df['interaction'].sum()} ({df['interaction'].mean() * 100:.2f}%)")
    print(f"Negative interactions: {(1 - df['interaction']).sum()} ({(1 - df['interaction'].mean()) * 100:.2f}%)")

    print("\n--- Drug Statistics ---")
    print(f"Unique drugs: {df['drug_smiles'].nunique()}")
    drug_lengths = df['drug_smiles'].str.len()
    print(f"SMILES length - Min: {drug_lengths.min()}, Max: {drug_lengths.max()}, Mean: {drug_lengths.mean():.1f}")

    print("\n--- Protein Statistics ---")
    print(f"Unique proteins: {df['protein_sequence'].nunique()}")
    protein_lengths = df['protein_sequence'].str.len()
    print(
        f"Sequence length - Min: {protein_lengths.min()}, Max: {protein_lengths.max()}, Mean: {protein_lengths.mean():.1f}")

    print("\n--- Affinity Statistics ---")
    print(df['affinity'].describe())

    print("\n--- Dataset Distribution ---")
    if 'dataset' in df.columns:
        print(df['dataset'].value_counts())

    # Check for missing values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    return df


# Main execution script
if __name__ == "__main__":
    # Initialize loader
    loader = BindingDBLoader()

    # Option 1: Load both datasets
    print("\nLoading BindingDB datasets...")
    combined_df = loader.load_both_datasets()

    # Option 2: Load individual datasets (uncomment if needed)
    # davis_df = loader.load_davis_dataset()
    # kiba_df = loader.load_kiba_dataset()

    # Explore the data
    explore_dataset(combined_df)

    # Save the combined dataset
    saved_path = loader.save_processed_dataset(combined_df)

    print("\n" + "=" * 60)
    print("✓ Data loading complete!")
    print(f"✓ Dataset saved to: {saved_path}")
    print("\nNext steps:")
    print("1. Run preprocess.py to generate features")
    print("2. Split data into train/val/test sets")
    print("3. Build and train your neural network")
    print("=" * 60)