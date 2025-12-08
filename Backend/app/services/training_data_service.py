import pandas as pd
from pathlib import Path

TRAINING_DATA_PATH = Path(__file__).resolve().parent.parent / "data/loaded/bindingdb_offline_processed_2025-11-17_20-05.csv"

# Load once at startup
df = pd.read_csv(TRAINING_DATA_PATH)

# Determine correct target column
possible_target_cols = ['target_name', 'Target_name', 'Target Name', 'protein', 'Protein']
for col in possible_target_cols:
    if col in df.columns:
        target_col = col
        break
else:
    target_col = None
    print("⚠️ No target/protein column found in dataset.")

# Unique lists
DRUG_SMILES_LIST = df['drug_smiles'].dropna().unique().tolist()
TARGET_LIST = df[target_col].dropna().astype(str).str.strip().unique().tolist()

def autocomplete_drugs(query: str):
    q = query.lower()
    return [d for d in DRUG_SMILES_LIST if q in d.lower()][:10]

def autocomplete_proteins(query: str):
    if not TARGET_LIST:
        return []
    q = query.lower()
    return [t for t in TARGET_LIST if q in t.lower()][:10]