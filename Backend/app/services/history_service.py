import json
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "history.json"

# Ensure file exists
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
if not HISTORY_PATH.exists():
    HISTORY_PATH.write_text("[]")

def load_history():
    return json.loads(HISTORY_PATH.read_text())

def save_history(history):
    HISTORY_PATH.write_text(json.dumps(history, indent=2))

# In-memory cache
_history_cache = load_history()