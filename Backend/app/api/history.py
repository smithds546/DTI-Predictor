from fastapi import APIRouter
from typing import List
from app.schemas.predict_schema import PredictResponse
from app.services.history_service import _history_cache, save_history

router = APIRouter()

@router.get("/history", response_model=List[PredictResponse])
def history():
    # Return latest 100 in reverse order (most recent first)
    return list(reversed(_history_cache[-100:]))

@router.delete("/history")
def clear_history():
    _history_cache.clear()
    save_history(_history_cache)
    return {"ok": True}