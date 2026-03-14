"""
Precompute and cache molecular descriptors for all unique drugs in the
training set.  Descriptors are computed via RDKit from SMILES strings and
persisted to a Parquet file so subsequent startups are fast.

Provides filter functions used by the /drug-filter endpoint.
"""

import os
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, MACCSkeys
from rdkit import DataStructs

from app.services.training_data_service import df as _training_df, target_col, DRUG_ROWS

# ─── Paths ────────────────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "preprocessed")
_CACHE_PATH = os.path.join(_CACHE_DIR, "drug_descriptors.pkl")


# ─── Descriptor computation ──────────────────────────────────────────────────

def _compute_descriptors(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "rings": Descriptors.RingCount(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
    }


def _build_descriptor_table() -> pd.DataFrame:
    """Compute descriptors for every unique (drug_name, drug_smiles) pair."""
    unique = DRUG_ROWS.drop_duplicates(subset=["drug_smiles"]).copy()
    print(f"Computing molecular descriptors for {len(unique)} unique SMILES …")
    t0 = time.time()

    records = []
    for _, row in unique.iterrows():
        smiles = str(row["drug_smiles"])
        desc = _compute_descriptors(smiles)
        if desc is None:
            continue
        desc["drug_smiles"] = smiles
        desc["drug_name"] = row.get("drug_name")
        records.append(desc)

    result = pd.DataFrame(records)
    elapsed = time.time() - t0
    print(f"  → {len(result)} compounds in {elapsed:.1f}s ({len(unique) - len(result)} invalid SMILES skipped)")
    return result


def _load_or_build() -> pd.DataFrame:
    if os.path.exists(_CACHE_PATH):
        cached = pd.read_pickle(_CACHE_PATH)
        # Check cache is still roughly in sync with training data
        n_unique = DRUG_ROWS["drug_smiles"].nunique()
        if abs(len(cached) - n_unique) < n_unique * 0.05:
            print(f"Loaded cached descriptors ({len(cached)} compounds)")
            return cached
        print("Descriptor cache is stale, rebuilding …")

    table = _build_descriptor_table()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    table.to_pickle(_CACHE_PATH)
    print(f"Saved descriptor cache to {_CACHE_PATH}")
    return table


# ─── Module-level cache ──────────────────────────────────────────────────────

DESC_TABLE: pd.DataFrame = _load_or_build()


# ─── Known-binder lookup (interaction == 1 per target) ───────────────────────

def _build_binder_index() -> dict[str, set[str]]:
    """Map target name → set of SMILES that are known binders (interaction=1)."""
    if "interaction" not in _training_df.columns or target_col is None:
        return {}
    binders = _training_df[_training_df["interaction"] == 1]
    index: dict[str, set[str]] = {}
    for tname, group in binders.groupby(target_col):
        tname = str(tname).strip()
        index[tname] = set(group["drug_smiles"].dropna().astype(str).unique())
    return index


_BINDER_INDEX: dict[str, set[str]] = _build_binder_index()


# ─── Filter API ──────────────────────────────────────────────────────────────

def get_available_filters() -> list[dict]:
    """Return the list of filter definitions the frontend can render."""
    return [
        {
            "id": "lipinski",
            "label": "Lipinski's Rule of Five",
            "description": "MW \u2264 500, LogP \u2264 5, HBA \u2264 10, HBD \u2264 5",
            "type": "toggle",
        },
        {
            "id": "mw_range",
            "label": "Molecular weight (Da)",
            "type": "preset_range",
        },
        {
            "id": "rings",
            "label": "Ring count",
            "type": "preset_range",
        },
        {
            "id": "known_binders",
            "label": "Known binders of a target",
            "description": "Compounds with interaction = 1 for a chosen target",
            "type": "target_select",
            "target_count": len(_BINDER_INDEX),
        },
    ]


def _apply_mask(
    *,
    lipinski: bool = False,
    mw_min: float | None = None,
    mw_max: float | None = None,
    rings_min: int | None = None,
    rings_max: int | None = None,
    known_binders_target: str | None = None,
) -> pd.Series:
    """Build a boolean mask over DESC_TABLE for the given filters."""
    mask = pd.Series(True, index=DESC_TABLE.index)

    if lipinski:
        mask &= (
            (DESC_TABLE["mw"] <= 500)
            & (DESC_TABLE["logp"] <= 5)
            & (DESC_TABLE["hba"] <= 10)
            & (DESC_TABLE["hbd"] <= 5)
        )

    if mw_min is not None:
        mask &= DESC_TABLE["mw"] >= mw_min
    if mw_max is not None:
        mask &= DESC_TABLE["mw"] <= mw_max

    if rings_min is not None:
        mask &= DESC_TABLE["rings"] >= rings_min
    if rings_max is not None:
        mask &= DESC_TABLE["rings"] <= rings_max

    if known_binders_target:
        binder_smiles = _BINDER_INDEX.get(known_binders_target.strip(), set())
        if not binder_smiles:
            return pd.Series(False, index=DESC_TABLE.index)
        mask &= DESC_TABLE["drug_smiles"].isin(binder_smiles)

    return mask


def count_drugs(**kwargs) -> int:
    """Return the number of compounds matching the given filters."""
    return int(_apply_mask(**kwargs).sum())


def _diverse_pick(smiles_list: list[str], n: int) -> list[int]:
    """MaxMin diversity picking using MACCS Tanimoto distance.

    Selects *n* compounds from *smiles_list* that are maximally spread
    across chemical space.  This mirrors how real HTS libraries are
    curated — you want broad structural coverage, not 50 close analogues.

    Returns indices into *smiles_list*.
    """
    # Pre-screen: if the pool is only moderately larger than n, random
    # is nearly as diverse and much faster.
    if len(smiles_list) <= n:
        return list(range(len(smiles_list)))
    if len(smiles_list) <= n * 3:
        rng = np.random.default_rng()
        return rng.choice(len(smiles_list), size=n, replace=False).tolist()

    # Compute MACCS fingerprints
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(MACCSkeys.GenMACCSKeys(mol))
            valid_idx.append(i)

    if len(fps) <= n:
        return valid_idx

    # MaxMin algorithm: greedily pick the compound most distant from
    # all already-picked compounds.
    picked = [0]  # seed with first compound
    min_dists = np.array([
        1.0 - DataStructs.TanimotoSimilarity(fps[0], fp) for fp in fps
    ])

    for _ in range(n - 1):
        # Pick the compound with the largest minimum distance to the
        # already-selected set.
        next_idx = int(np.argmax(min_dists))
        picked.append(next_idx)
        # Update min distances
        new_dists = np.array([
            1.0 - DataStructs.TanimotoSimilarity(fps[next_idx], fp) for fp in fps
        ])
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[picked] = -1  # exclude already picked

    return [valid_idx[i] for i in picked]


def filter_drugs(*, limit: int = 50, **kwargs) -> list[dict]:
    """
    Apply filters to the descriptor table and return matching drugs.

    When the pool exceeds *limit*, uses MaxMin diversity picking on MACCS
    fingerprints so the returned subset covers as much chemical space as
    possible rather than being a random slice.

    Returns up to `limit` drugs as [{name, smiles, mw, logp, rings, ...}].
    """
    mask = _apply_mask(**kwargs)
    filtered = DESC_TABLE[mask]

    if len(filtered) == 0:
        return []

    # Diversity-pick if pool > limit
    if len(filtered) > limit:
        # Cap the candidate pool for tractable diversity picking.
        # Pre-sample to ~10x the limit so MaxMin has variety but stays fast.
        max_pool = limit * 10
        if len(filtered) > max_pool:
            filtered = filtered.sample(n=max_pool, random_state=None)
        smiles_list = filtered["drug_smiles"].tolist()
        picked_idx = _diverse_pick(smiles_list, limit)
        filtered = filtered.iloc[picked_idx]

    return [
        {
            "name": row.get("drug_name"),
            "smiles": row["drug_smiles"],
            "mw": round(row["mw"], 1),
            "logp": round(row["logp"], 2),
            "hba": int(row["hba"]),
            "hbd": int(row["hbd"]),
            "rings": int(row["rings"]),
        }
        for _, row in filtered.iterrows()
    ]
