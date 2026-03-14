import hashlib
import os
import sys
import glob as _glob

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys

# ─── Paths ────────────────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_APP_DIR, "Models", "Torch", "dnn_adam_best.pt")
_PROT_CACHE_DIR = os.path.join(_APP_DIR, "data", "preprocessed", "protein_cache")
_DATA_LOADED_DIR = os.path.join(_APP_DIR, "data", "loaded")

# ─── Load model ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(_APP_DIR, "Models", "Torch"))
from dnn import DTI_DNN  # noqa: E402 (inserted path above)

if torch.backends.mps.is_available():
    _device = torch.device("mps")
elif torch.cuda.is_available():
    _device = torch.device("cuda")
else:
    _device = torch.device("cpu")

_model = DTI_DNN()
_model.load_state_dict(torch.load(_MODEL_PATH, map_location=_device, weights_only=True))
_model.to(_device)
_model.eval()

# ─── Protein name → sequence lookup (built once at startup) ──────────────────
def _build_protein_lookup() -> dict:
    csv_files = _glob.glob(os.path.join(_DATA_LOADED_DIR, "*.csv"))
    if not csv_files:
        return {}
    df = pd.read_csv(max(csv_files, key=os.path.getmtime))
    if "Target_name" in df.columns and "protein_sequence" in df.columns:
        return dict(
            zip(
                df["Target_name"].astype(str).str.strip(),
                df["protein_sequence"].astype(str),
            )
        )
    return {}

_protein_lookup: dict = _build_protein_lookup()

# ─── Featurizers ──────────────────────────────────────────────────────────────

def _smiles_to_maccs(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(list(fp.ToBitString()), dtype=np.float32)


def _sequence_hash(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()[:12]


def _get_protein_embedding(protein_name: str) -> np.ndarray:
    seq = _protein_lookup.get(protein_name.strip())
    if seq is None:
        raise ValueError(
            f"Unknown protein '{protein_name}'. Please select from the autocomplete list."
        )
    cache_path = os.path.join(_PROT_CACHE_DIR, f"{_sequence_hash(seq)}.npy")
    if not os.path.exists(cache_path):
        raise ValueError(
            f"No cached ProtBERT embedding for '{protein_name}'. Re-run preprocessing."
        )
    return np.load(cache_path).astype(np.float32)


# ─── Inference ────────────────────────────────────────────────────────────────

def simple_score(drug: str, protein: str) -> float:
    """Return a binding probability in [0, 1] using the Adam-trained DTI_DNN."""
    drug_feat = _smiles_to_maccs(drug)
    prot_feat = _get_protein_embedding(protein)

    x_drug = torch.tensor(drug_feat).unsqueeze(0).to(_device)  # [1, 167]
    x_prot = torch.tensor(prot_feat).unsqueeze(0).to(_device)  # [1, 1024]

    prob = _model.predict_proba(x_drug, x_prot)  # [1, 1]
    return float(prob.squeeze().item())


def batch_score(drugs: list[str], protein: str) -> list[float]:
    """Score multiple drugs against a single protein in one batched forward pass."""
    prot_feat = _get_protein_embedding(protein)

    drug_feats = []
    for smiles in drugs:
        drug_feats.append(_smiles_to_maccs(smiles))

    x_drug = torch.tensor(np.stack(drug_feats)).to(_device)        # [N, 167]
    x_prot = torch.tensor(prot_feat).unsqueeze(0).expand(len(drugs), -1).to(_device)  # [N, 1024]

    probs = _model.predict_proba(x_drug, x_prot)  # [N, 1]
    flat = probs.squeeze(-1)  # [N]
    values = flat.tolist()
    # tolist() returns a scalar when N=1, wrap it
    if isinstance(values, float):
        values = [values]
    return [float(p) for p in values]