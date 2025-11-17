import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
import hashlib

def main():
    print("Starting preprocessing...")

    loaded_dir = "/Users/drs/Projects/DTI/Backend/app/data/loaded/"
    csv_files = [os.path.join(loaded_dir, f) for f in os.listdir(loaded_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {loaded_dir}")
    input_path = max(csv_files, key=os.path.getmtime)
    print(f"Using most recent CSV: {os.path.basename(input_path)}")
    output_dir = "/Users/drs/Projects/DTI/Backend/app/data/preprocessed/"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)

    # --- TEMPORARY: use a small subset to test ---
    df = df.sample(frac=0.01, random_state=42)  # use 1% of the data randomly
    # OR, for a fixed number of rows:
    # df = df.head(500)

    print(f"Using a test subset: {len(df)} rows")

    # --- Drug features (MACCS) ---
    def smiles_to_maccs(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(list(fp.ToBitString()), dtype=int)
        else:
            return np.zeros(167, dtype=int)

    maccs_path = os.path.join(output_dir, "drug_maccs.npy")
    if os.path.exists(maccs_path):
        print("Loading MACCS features from cache...")
        maccs_features = np.load(maccs_path)
    else:
        print("Computing MACCS features...")
        maccs_features = np.vstack([smiles_to_maccs(s) for s in tqdm(df['drug_smiles'], desc="MACCS keys")])
        np.save(maccs_path, maccs_features)

    # --- Protein features (ProtBERT) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert").to(device).eval()

    def sequence_hash(seq):
        """Create a stable short hash for the protein sequence."""
        return hashlib.md5(seq.encode('utf-8')).hexdigest()[:12]

    protein_cache_dir = os.path.join(output_dir, "protein_cache")
    os.makedirs(protein_cache_dir, exist_ok=True)

    def protbert_embedding_cached(seq):
        """Compute ProtBERT embedding if not already cached."""
        seq_hash = sequence_hash(seq)
        cache_path = os.path.join(protein_cache_dir, f"{seq_hash}.npy")

        if os.path.exists(cache_path):
            return np.load(cache_path)

        seq_spaced = ' '.join(list(seq))
        inputs = tokenizer(seq_spaced, return_tensors='pt', truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        np.save(cache_path, emb)
        return emb


    protein_features_path = os.path.join(output_dir, "protein_protbert.npy")
    if os.path.exists(protein_features_path):
        print("Loading ProtBERT features from cache...")
        protein_features = np.load(protein_features_path)
    else:
        print("Computing ProtBERT features...")
        protein_features = np.vstack([protbert_embedding_cached(seq) for seq in tqdm(df['protein_sequence'], desc="ProtBERT")])
        np.save(protein_features_path, protein_features)

    # --- Save features ---
    df.to_csv(os.path.join(output_dir, "bindingdb_featured.csv"), index=False)

    print("✅ Feature extraction complete.")
    print("Preprocessing finished.")

if __name__ == "__main__":
    main()