from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import os
import math

# Simple, lightweight featurizer and scorer placeholder
# This can be replaced with a real model loaded from disk.


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def simple_featurize(text: str) -> float:
    # Map string to a stable numeric feature: normalized sum of char codes
    if not text:
        return 0.0
    total = sum((i + 1) * ord(c) for i, c in enumerate(text))
    return (total % 10000) / 10000.0


def simple_score(drug: str, protein: str) -> float:
    # Simple interaction proxy combining features with sine/cosine mixing
    fd = simple_featurize(drug)
    fp = simple_featurize(protein)
    raw = (fd - 0.5) * (fp - 0.5) * 8.0 + math.sin(fd * 10.0) * 0.3 + math.cos(fp * 10.0) * 0.3
    return _sigmoid(raw)


HISTORY_PATH = os.path.join(os.path.dirname(__file__), "data", "history.json")
os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)


def load_history() -> List[dict]:
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            return []
    return []


def save_history(history: List[dict]):
    try:
        with open(HISTORY_PATH, "w") as f:
            json.dump(history[-100:], f, indent=2)
    except Exception:
        pass


class PredictRequest(BaseModel):
    drug: str
    protein: str


class PredictResponse(BaseModel):
    binder: bool
    score: float
    timestamp: str
    drug: str
    protein: str
    id: Optional[str] = None


app = FastAPI(title="DTI FastAPI", version="0.1.0")

# Allow Gatsby dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_history_cache: List[dict] = load_history()


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.drug or not req.protein:
        raise HTTPException(status_code=400, detail="Both 'drug' and 'protein' must be provided")
    score = simple_score(req.drug, req.protein)
    binder = bool(score >= 0.5)
    ts = datetime.utcnow().isoformat() + "Z"
    item = {
        "id": f"{hash((req.drug, req.protein, ts))}",
        "drug": req.drug,
        "protein": req.protein,
        "score": round(float(score), 6),
        "binder": binder,
        "timestamp": ts,
    }
    _history_cache.append(item)
    save_history(_history_cache)
    return item


@app.get("/history", response_model=List[PredictResponse])
def history():
    # Return most recent first
    return list(reversed(_history_cache[-100:]))


@app.delete("/history")
def clear_history():
    _history_cache.clear()
    save_history(_history_cache)
    return {"ok": True}
