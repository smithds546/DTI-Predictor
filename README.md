# DTI-Predictor
This is my final year project where I use machine learning to predict the binding success rate between drug molecules and protein targets. I will code the frontend off of the react js framework and use a REST API to connect it to the backend - Neural network, Database

For Data:

# 1. Install dependencies (if not already done)
cd Backend
pip install -r requirements.txt

# 2. Run the data preparation script
python -m app.data.download_and_prepare


Next Steps:
    - put data table in data chapter - statistics of the data, distribution of binders/non-binders, class imbalance etc.
    - get citations for Fastapi, 
    - not sure what to write in \subsection{Model Inference Pipeline}
    - use code examples


How to run frontend:

gatsby develop or gatsby build

How to run backend:

cd /Users/drs/Projects/DTI/Backend
uvicorn app.main:app --reload

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

from the Report/ directory:
latexmk -pdf Main.tex 
latexmk -pdf -bibtex Main.tex
latexmk -C Main.tex && latexmk -pdf -bibtex Main.tex

for claude code opus 4.6:
claude --model opus


Our dataset exhibited a slight class imbalance, with binders (60.25\%) outnumbering non-binders (39.75\%) after removing interactions in the "grey area" (pAffinity between 5.3 and 7.0).
