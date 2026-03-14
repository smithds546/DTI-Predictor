from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.schemas.predict_schema import PredictRequest, PredictResponse, ScreenRequest, ScreenResponse, ScreenHit
from app.services.predict_service import simple_score, batch_score
from app.services.training_data_service import DRUG_ROWS
from app.services.drug_descriptor_service import DESC_TABLE
from app.services.history_service import _history_cache, save_history
import uuid

router = APIRouter()

# Build a reverse lookup: SMILES -> drug name (for labelling screen results)
_smiles_to_name: dict[str, str] = {}
for _, row in DRUG_ROWS.iterrows():
    s = str(row["drug_smiles"]).strip()
    n = row.get("drug_name")
    if s and n and s not in _smiles_to_name:
        _smiles_to_name[s] = str(n)

# Build SMILES -> descriptor lookup for enriching screen results
_smiles_to_desc: dict[str, dict] = {}
for _, row in DESC_TABLE.iterrows():
    _smiles_to_desc[str(row["drug_smiles"]).strip()] = {
        "mw": round(float(row["mw"]), 1),
        "logp": round(float(row["logp"]), 2),
        "rings": int(row["rings"]),
    }


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        score = simple_score(req.drug, req.protein)
        binder = score > 0.5

        # Resolve drug name: use client-provided name, fall back to SMILES lookup
        drug_name = req.drug_name or _smiles_to_name.get(req.drug.strip())

        item = {
            "id": str(uuid.uuid4()),
            "type": "single",
            "drug": req.drug,
            "drug_name": drug_name,
            "protein": req.protein,
            "score": score,
            "binder": binder,
            "timestamp": datetime.utcnow().isoformat()
        }

        _history_cache.append(item)
        save_history(_history_cache)

        return item

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screen", response_model=ScreenResponse)
def screen(req: ScreenRequest):
    """Screen multiple drugs against a single protein target (virtual screening)."""
    if not req.drugs:
        raise HTTPException(status_code=400, detail="At least one drug is required.")
    if len(req.drugs) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 drugs per screen.")

    smiles_list = [d.smiles for d in req.drugs]

    try:
        scores = batch_score(smiles_list, req.protein)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    hits = []
    for entry, score in zip(req.drugs, scores):
        # Use client-provided name, fall back to server-side SMILES lookup
        name = entry.name or _smiles_to_name.get(entry.smiles.strip())
        desc = _smiles_to_desc.get(entry.smiles.strip(), {})
        hits.append(ScreenHit(
            drug=entry.smiles,
            drug_name=name,
            score=score,
            binder=score > 0.5,
            rank=0,
            mw=desc.get("mw"),
            logp=desc.get("logp"),
            rings=desc.get("rings"),
        ))

    # Sort by score descending (best binders first)
    hits.sort(key=lambda h: h.score, reverse=True)
    for i, h in enumerate(hits):
        h.rank = i + 1

    response = ScreenResponse(
        protein=req.protein,
        hits=hits,
        timestamp=datetime.utcnow().isoformat(),
    )

    # Save to history
    screen_item = {
        "id": str(uuid.uuid4()),
        "type": "screen",
        "timestamp": response.timestamp,
        "screen_data": response.model_dump(),
    }
    _history_cache.append(screen_item)
    save_history(_history_cache)

    return response