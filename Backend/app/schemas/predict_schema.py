from pydantic import BaseModel
from typing import Optional, List

class PredictRequest(BaseModel):
    drug: str                       # SMILES string
    drug_name: Optional[str] = None # human-readable name (resolved server-side if omitted)
    protein: str

class PredictResponse(BaseModel):
    id: Optional[str]
    drug: str                       # SMILES
    drug_name: Optional[str]        # human-readable name
    protein: str
    score: float
    binder: bool
    timestamp: str


class DrugEntry(BaseModel):
    smiles: str
    name: Optional[str] = None

class ScreenRequest(BaseModel):
    drugs: List[DrugEntry]  # list of {smiles, name} objects
    protein: str            # single protein target name

class ScreenHit(BaseModel):
    drug: str
    drug_name: Optional[str]
    score: float
    binder: bool
    rank: int
    mw: Optional[float] = None
    logp: Optional[float] = None
    rings: Optional[int] = None

class ScreenResponse(BaseModel):
    protein: str
    hits: List[ScreenHit]
    timestamp: str


class HistoryEntry(BaseModel):
    """Unified history entry for both single predictions and virtual screens."""
    id: Optional[str]
    type: str  # "single" | "screen"
    timestamp: str
    # Single prediction fields
    drug: Optional[str] = None
    drug_name: Optional[str] = None
    protein: Optional[str] = None
    score: Optional[float] = None
    binder: Optional[bool] = None
    # Screen fields
    screen_data: Optional[ScreenResponse] = None