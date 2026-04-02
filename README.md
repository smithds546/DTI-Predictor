# DTI-Predictor
This is my final year project where I use machine learning to predict the binding success rate between drug molecules and protein targets. I will code the frontend off of the react js framework and use a REST API to connect it to the backend - Neural network, Database

For Data:

# 1. Install dependencies (if not already done)
cd Backend
pip install -r requirements.txt

# 2. Run the data preparation script
python -m app.data.download_and_prepare


Next Steps:
    - in evaluation - try a narrower grey area 5.5-6.5 and 5.0-8.0, compare class balance and performance
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



Backend/app/Models/Experiments/grey_area/prepare_data.py — Data preparation + distribution     
  figures:                                                                                       
  1. Processes the raw BindingDB zip once (caches result for re-runs)                            
  2. Filters to proteins with cached ProtBERT embeddings                                         
  3. Computes MACCS fingerprints for all unique drugs                                            
  4. For each variant (original 5.3-7.0, narrow 5.5-6.5, wide 4.5-8.0): applies thresholds,      
  counts binder/non-binder/grey, splits 65/20/15, saves .npy + .csv                              
  5. Generates:                                                                                  
    - figures/class_distribution.png — Two-panel: (a) binder/non-binder/grey counts, (b) binder  
  percentage showing class balance                                                               
    - figures/paffinity_thresholds.png — pAffinity histogram with the three grey bands overlaid  
    - figures/dataset_summary.json — All counts                                                
                                                                                                 
  Backend/app/Models/Experiments/grey_area/train_and_compare.py — Training + comparison figures: 
  1. Trains DTI_DNN with Adam + CosineAnnealing for each variant (same hyperparameters as your   
  existing run_dnn_adam.py)                                                                      
  2. Supports selective training: python train_and_compare.py narrow to run one at a time        
  3. Generates:                                                                                  
    - figures/comparison_roc.png — 3-way ROC overlay                                             
    - figures/comparison_metrics.png — Side-by-side metrics table with best values highlighted in
   green                                                                                         
    - figures/comparison_loss.png — 3-panel loss curves with early stopping markers              
   
  To run:                                                                                        
  cd Backend/app/Models/Experiments/grey_area               
  python prepare_data.py           # ~10-15 min (zip processing + MACCS computation)             
  python train_and_compare.py      # ~30-90 min depending on hardware (3 training runs)          
                                                                                                 
  If memory is tight, set SAMPLE_FRAC = 0.1 at the top of prepare_data.py to use 10% of data.    
                   