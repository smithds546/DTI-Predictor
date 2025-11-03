"""
Data preprocessing module for DTI prediction
Handles drug and protein feature extraction and dataset preparation
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Tuple, List, Optional
import pickle
import os


class DrugPreprocessor:
    """Handle drug molecule preprocessing"""

    def __init__(self, fingerprint_size: int = 2048):
        self.fingerprint_size = fingerprint_size

    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert SMILES string to Morgan fingerprint

        Args:
            smiles: SMILES representation of molecule

        Returns:
            Numpy array of molecular fingerprint or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=self.fingerprint_size
            )
            return np.array(fp)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None

    def calculate_descriptors(self, smiles: str) -> Optional[dict]:
        """Calculate molecular descriptors"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            descriptors = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_h_acceptors': Descriptors.NumHAcceptors(mol),
                'num_h_donors': Descriptors.NumHDonors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol)
            }
            return descriptors
        except Exception as e:
            print(f"Error calculating descriptors: {e}")
            return None


class ProteinPreprocessor:
    """Handle protein sequence preprocessing"""

    def __init__(self, max_seq_length: int = 1000):
        self.max_seq_length = max_seq_length
        # Standard amino acid encoding
        self.aa_dict = {aa: i + 1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.aa_dict['X'] = 0  # Unknown amino acid

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode protein sequence as integers

        Args:
            sequence: Protein amino acid sequence

        Returns:
            Encoded and padded sequence
        """
        # Convert to uppercase and limit length
        sequence = sequence.upper()[:self.max_seq_length]

        # Encode amino acids
        encoded = [self.aa_dict.get(aa, 0) for aa in sequence]

        # Pad sequence
        if len(encoded) < self.max_seq_length:
            encoded.extend([0] * (self.max_seq_length - len(encoded)))

        return np.array(encoded)

    def calculate_protein_features(self, sequence: str) -> dict:
        """Calculate basic protein features"""
        return {
            'length': len(sequence),
            'molecular_weight': sum([self._aa_weights.get(aa, 0) for aa in sequence]),
            'charge': sum([self._aa_charges.get(aa, 0) for aa in sequence])
        }

    # Simplified amino acid properties
    _aa_weights = {
        'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
        'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
        'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
        'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
    }

    _aa_charges = {
        'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5
    }


class DTIDataset:
    """Main dataset handler for DTI prediction"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.drug_processor = DrugPreprocessor()
        self.protein_processor = ProteinPreprocessor()

    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw DTI data from CSV

        Expected columns: drug_id, drug_smiles, protein_id, protein_sequence, interaction
        """
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} interactions")
        return df

    def preprocess_dataset(
            self,
            df: pd.DataFrame,
            save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess full dataset

        Returns:
            drug_features, protein_features, labels
        """
        drug_features = []
        protein_features = []
        labels = []

        print("Processing dataset...")
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)} samples")

            # Process drug
            drug_fp = self.drug_processor.smiles_to_fingerprint(row['drug_smiles'])
            if drug_fp is None:
                continue

            # Process protein
            protein_enc = self.protein_processor.encode_sequence(row['protein_sequence'])

            drug_features.append(drug_fp)
            protein_features.append(protein_enc)
            labels.append(row['interaction'])

        # Convert to numpy arrays
        drug_features = np.array(drug_features)
        protein_features = np.array(protein_features)
        labels = np.array(labels)

        print(f"\nFinal dataset: {len(labels)} samples")
        print(f"Drug features shape: {drug_features.shape}")
        print(f"Protein features shape: {protein_features.shape}")

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(
                save_path,
                drug_features=drug_features,
                protein_features=protein_features,
                labels=labels
            )
            print(f"Saved processed data to {save_path}")

        return drug_features, protein_features, labels


# Example usage
if __name__ == "__main__":
    # Example: Test preprocessing
    dataset = DTIDataset()

    # Test drug processing
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    drug_fp = dataset.drug_processor.smiles_to_fingerprint(test_smiles)
    print(f"Drug fingerprint shape: {drug_fp.shape if drug_fp is not None else 'Invalid'}")

    # Test protein processing
    test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    protein_enc = dataset.protein_processor.encode_sequence(test_sequence)
    print(f"Protein encoding shape: {protein_enc.shape}")