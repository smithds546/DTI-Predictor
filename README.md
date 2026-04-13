# DTI-Predictor
This is my final year project where I use machine learning to predict the binding success rate between drug molecules and protein targets. I will code the frontend off of the react js framework and use a REST API to connect it to the backend - Neural network, Database

Next Steps:
    - cnage stuff in interface to include ethicalm message
    - use code examples


How to run backend:

cd Backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

How to run frontend:

gatsby develop -p 8001

click link:
http://localhost:8001/

### Notes
- The history is limited to the latest 100 entries and persists to JSON. You can swap to SQLite/SQLModel later without changing the frontend.
- CORS currently allows `http://localhost:8000` (Gatsby dev). Add production origin(s) as needed.

from the Report/ directory:
latexmk -pdf Main.tex 
latexmk -pdf -bibtex Main.tex
latexmk -C Main.tex && latexmk -pdf -bibtex Main.tex

cd Report && latexmk -C Main.tex && latexmk -pdf -bibtex Main.tex                              

for claude code opus 4.6:
claude --model opus





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
                   

Model design:
    to include in report:
    he forward pass returns raw logits (no in‑network sigmoid) and BCEWithLogitsLoss applies sigmoid implicitly during training.
    
    Fix:
    The hidden=32 / dropout=0.2 / no‑BN / no‑scheduler row looks copy‑pasted from the arch3 NumPyiteration, not the final PyTorch model.

interface:
