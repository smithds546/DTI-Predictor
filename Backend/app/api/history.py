from fastapi import APIRouter
from typing import List
from app.schemas.predict_schema import HistoryEntry
from app.services.history_service import _history_cache, save_history

router = APIRouter()

@router.get("/history", response_model=List[HistoryEntry])
def history():
    # Backfill type for legacy entries that lack it
    entries = []
    for h in _history_cache[-100:]:
        if "type" not in h:
            h["type"] = "single"
        entries.append(h)
    return list(reversed(entries))

@router.delete("/history")
def clear_history():
    _history_cache.clear()
    save_history(_history_cache)
    return {"ok": True}
