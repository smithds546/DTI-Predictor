# DTI-Predictor
This is my final year project where I use machine learning to predict the binding success rate between drug molecules and protein targets. I will code the frontend off of the react js framework and use a REST API to connect it to the backend - Neural network, Database

For Data:

# 1. Install dependencies (if not already done)
cd Backend
pip install -r requirements.txt

# 2. Run the data preparation script
python -m app.data.download_and_prepare

https://www.kaggle.com/datasets/christang0002/davis-and-kiba/data

for protein sequences:
	1.	Pull the UniProt ID from BindingDB.
	2.	Query UniProt for the amino acid sequence.
	3.	Combine the two into your dataset.

for drug SMILES:
    1.  rest api json output

Use ProtBERT for Feature Extraction of protein sequences

Use RdKIT to convert SMILES to MACCS keys


Next Steps:
    change data to ingest drug name as well as the SMILES so its more usable for frontend.
    add code snippets into paper
    For evalaution:
        could show difference between using logged binding affinity and not
        difference between using random negative sampling and not
        difference between using grey area and not
    potentially look at training on chembl as well


How to run frontend:

gatsby develop

How to run backend:

cd /Users/drs/Projects/DTI/Backend
uvicorn app.main:app --reload


what Junie did:

### What I changed
- Implemented a FastAPI backend with endpoints to predict binder/non-binder and to manage a history of predictions.
- Replaced the default Gatsby starter homepage with a React-based DTI form that calls the FastAPI backend, shows the latest result, and lists previous searches.

### Backend (FastAPI)
- File added/updated: `Backend/app/main.py`
  - Added endpoints:
    - `GET /health` — health check
    - `POST /predict` — takes `drug` and `protein`, returns `{binder, score, timestamp, ...}` and stores in history
    - `GET /history` — returns the latest 100 predictions (most recent first)
    - `DELETE /history` — clears stored history
  - Enabled CORS for Gatsby dev (`http://localhost:8000`).
  - Implemented lightweight, simple placeholder scoring (`simple_score`) with deterministic features to make the API functional while you wire in the real model.
  - Implemented persistent JSON-backed history at `Backend/app/data/history.json` (auto-created). 

Note: This backend scaffolding is designed so you can later replace `simple_score(drug, protein)` with a call to your trained model (e.g., load weights at startup and compute the real binder/non-binder probability and threshold at inference).

### Frontend (Gatsby/React)
- File updated: `Frontend/src/pages/index.js`
  - Replaced the default starter page with a DTI Checker UI:
    - Input form for `drug` and `protein`.
    - On submit, calls `POST /predict`.
    - Displays prediction (binder/non-binder) and score.
    - Fetches and displays history from `GET /history`.
    - Button to clear history via `DELETE /history`.
  - API base URL is configurable: `GATSBY_API_BASE_URL` (defaults to `http://localhost:8001`).

### How to run (local dev)
1) Backend (FastAPI)
- From project root:
```
cd Backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```
- Check health: http://localhost:8001/health
2) Frontend (Gatsby)
- In another terminal:
```
cd Frontend
# optionally: echo 'GATSBY_API_BASE_URL=http://localhost:8001' > .env.development
npm install
npm run develop
```
- Open UI: http://localhost:8000

### Wiring the real model later
- Replace the placeholder in `Backend/app/main.py`:
  - Load your trained model on startup (e.g., in module scope or inside a FastAPI `startup` event), then implement `predict()` to compute a real probability `score` from the model using your drug/protein featurization.
  - Keep returning `{binder: score >= threshold, score, ...}` to avoid frontend changes.

### Notes
- The history is limited to the latest 100 entries and persists to JSON. You can swap to SQLite/SQLModel later without changing the frontend.
- CORS currently allows `http://localhost:8000` (Gatsby dev). Add production origin(s) as needed.

