from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.schemas.predict_schema import PredictRequest, PredictResponse
from app.services.predict_service import simple_score
from app.services.history_service import _history_cache, save_history
import uuid

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        score = simple_score(req.drug, req.protein)
        binder = score > 0.5

        item = {
            "id": str(uuid.uuid4()),
            "drug": req.drug,
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