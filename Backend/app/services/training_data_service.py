import pandas as pd
from pathlib import Path

TRAINING_DATA_PATH = Path(__file__).resolve().parent.parent / "data/loaded/bindingdb_offline_processed_2025-12-16_13-15.csv"

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

# Prefer drug_name for search/display; keep drug_smiles for actual submission
HAS_DRUG_NAME = 'drug_name' in df.columns
HAS_DRUG_SMILES = 'drug_smiles' in df.columns

if not HAS_DRUG_SMILES:
    raise ValueError("Expected column 'drug_smiles' in processed dataset, but it was not found.")

DRUG_ROWS = df[['drug_name', 'drug_smiles']].copy() if HAS_DRUG_NAME else df[['drug_smiles']].assign(drug_name=None)
DRUG_ROWS['drug_name'] = DRUG_ROWS['drug_name'].where(DRUG_ROWS['drug_name'].notna(), None)
DRUG_ROWS['drug_smiles'] = DRUG_ROWS['drug_smiles'].astype(str)

# Deduplicate by (name, smiles)
DRUG_ROWS = DRUG_ROWS.drop_duplicates()

# Unique lists
DRUG_SMILES_LIST = df['drug_smiles'].dropna().unique().tolist()
TARGET_LIST = df[target_col].dropna().astype(str).str.strip().unique().tolist()

def autocomplete_drugs(query: str):
    """
    Returns up to 10 matches as objects:
      { "name": <drug_name or None>, "smiles": <drug_smiles> }

    Search order:
      1) name contains query (case-insensitive)
      2) fallback: smiles contains query
    """
    q = (query or "").strip().lower()
    if not q:
        return []

    results = []

    # 1) Name matches first (best UX)
    name_matches = DRUG_ROWS[
        DRUG_ROWS['drug_name'].notna() &
        DRUG_ROWS['drug_name'].astype(str).str.lower().str.contains(q, na=False)
    ]
    for _, row in name_matches.head(10).iterrows():
        results.append({"name": row["drug_name"], "smiles": row["drug_smiles"]})

    # 2) Fill remainder with SMILES matches (useful if user pastes SMILES)
    if len(results) < 10:
        remaining = 10 - len(results)
        smiles_matches = DRUG_ROWS[
            DRUG_ROWS['drug_smiles'].astype(str).str.lower().str.contains(q, na=False)
        ]
        for _, row in smiles_matches.head(remaining).iterrows():
            item = {"name": row["drug_name"], "smiles": row["drug_smiles"]}
            if item not in results:
                results.append(item)

    return results[:10]


def autocomplete_proteins(query: str):
    if not TARGET_LIST:
        return []
    q = query.lower()
    return [t for t in TARGET_LIST if q in t.lower()][:10]


def get_random_drug_sample(n: int = 20):
    """Return n random unique drugs (name + smiles) for quick screening."""
    sample = DRUG_ROWS.dropna(subset=["drug_name"]).drop_duplicates(subset=["drug_smiles"]).sample(
        n=min(n, len(DRUG_ROWS)), random_state=None
    )
    return [{"name": row["drug_name"], "smiles": row["drug_smiles"]} for _, row in sample.iterrows()]