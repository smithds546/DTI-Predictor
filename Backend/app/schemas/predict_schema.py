from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    drug: str
    protein: str

class PredictResponse(BaseModel):
    id: Optional[str]
    drug: str
    protein: str
    score: float
    binder: bool
    timestamp: str