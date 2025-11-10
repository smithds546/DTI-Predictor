import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

def main():

    # === File paths ===
    input_csv = "/Users/drs/Projects/DTI/Backend/app/data/preprocessed/bindingdb_featured.csv"
    drug_npy = "/Users/drs/Projects/DTI/Backend/app/data/preprocessed/drug_maccs.npy"
    prot_npy = "/Users/drs/Projects/DTI/Backend/app/data/preprocessed/protein_protbert.npy"
    output_dir = "/Users/drs/Projects/DTI/Backend/app/data/prepped/"

    # Create subdirectories
    bindingdb_dir = os.path.join(output_dir, "bindingdb")
    drugs_dir = os.path.join(output_dir, "drugs")
    proteins_dir = os.path.join(output_dir, "proteins")

    os.makedirs(bindingdb_dir, exist_ok=True)
    os.makedirs(drugs_dir, exist_ok=True)
    os.makedirs(proteins_dir, exist_ok=True)

    # === Load dataset ===
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    X_drug = np.load(drug_npy)
    X_prot = np.load(prot_npy)

    print(f"Dataset loaded: {len(df)} rows")

    # === Split indices ===
    train_df, temp_df = train_test_split(df, test_size=0.35, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=(15 / 35), random_state=42, shuffle=True)

    # === Use the same indices for .npy arrays ===
    train_idx = train_df.index
    val_idx = val_df.index
    test_idx = test_df.index

    np.save(os.path.join(drugs_dir, "drug_train.npy"), X_drug[train_idx])
    np.save(os.path.join(drugs_dir, "drug_val.npy"), X_drug[val_idx])
    np.save(os.path.join(drugs_dir, "drug_test.npy"), X_drug[test_idx])

    np.save(os.path.join(proteins_dir, "prot_train.npy"), X_prot[train_idx])
    np.save(os.path.join(proteins_dir, "prot_val.npy"), X_prot[val_idx])
    np.save(os.path.join(proteins_dir, "prot_test.npy"), X_prot[test_idx])

    # === Save CSVs (metadata + labels) ===
    train_df.to_csv(os.path.join(bindingdb_dir, "bindingdb_train.csv"), index=False)
    val_df.to_csv(os.path.join(bindingdb_dir, "bindingdb_validation.csv"), index=False)
    test_df.to_csv(os.path.join(bindingdb_dir, "bindingdb_test.csv"), index=False)

    print("\n✅ Data successfully split and saved:")
    print(f" - Train CSV: {os.path.join(bindingdb_dir, 'bindingdb_train.csv')}")
    print(f" - Validation CSV: {os.path.join(bindingdb_dir, 'bindingdb_validation.csv')}")
    print(f" - Test CSV: {os.path.join(bindingdb_dir, 'bindingdb_test.csv')}")
    print(f" - Drug features: {drugs_dir}")
    print(f" - Protein features: {proteins_dir}")

if __name__ == "__main__":
    main()