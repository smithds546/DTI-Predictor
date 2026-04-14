# Drug–Target Interaction Prediction Using Deep Neural Networks

**Author:** Daniel R. Smith (F225694)
**Module:** 25COC251 — Computer Science Final Year Project
**Supervisor:** Mohamad Saada
**Institution:** Loughborough University, Department of Computer Science

This repository accompanies the final year dissertation *Drug–Target Interaction Prediction Using Deep Neural Networks*. It contains the full training pipeline, the trained PyTorch model, a FastAPI backend that serves predictions, and a Gatsby/React frontend for single-pair prediction and batch virtual screening. The compiled dissertation is submitted separately via Learn.

## Contents

```
.
├── Backend/                FastAPI service, training code, cached embeddings
│   ├── app/
│   │   ├── main.py               FastAPI entrypoint
│   │   ├── api/                  REST endpoints
│   │   ├── services/             Prediction, screening, autocomplete
│   │   ├── schemas/              Pydantic request/response models
│   │   ├── Models/               Training scripts, checkpoints, figures
│   │   └── data/                 Processed dataset, history, cached embeddings
│   ├── requirements.txt
│   └── run.py
└── Frontend/               Gatsby/React UI
    ├── src/
    ├── package.json
    └── gatsby-config.js
```

## Requirements

- Python 3.10+
- Node.js 18+ and npm
- ~4 GB free disk for cached ProtBERT embeddings and model checkpoints
- macOS / Linux (tested on macOS 14, Darwin 25.3)

## Running the Backend

```bash
cd Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

The backend loads the trained checkpoint from `Backend/app/Models/Experiments/grey_area/data/original/original_best.pt` at startup (see `app/services/predict_service.py`). This file is included in the submission.

## Running the Frontend

```bash
cd Frontend
npm install
gatsby develop -p 8001
```

Open `http://localhost:8001/`. The frontend expects the backend to be running on port 8000; CORS is preconfigured for local development.

## Reproducing the Final Model

The final model reported in Chapter 5 (AUC-ROC 0.970, F1 0.931) was trained on the full BindingDB dataset via the grey-area experiment pipeline, using the architecture defined in `Backend/app/Models/Torch/dnn.py`. Hyperparameters match Table 3.1 of the report (Adam, lr 1e-3, weight decay 1e-4, batch 512, 100 epochs, CosineAnnealingLR, early stopping patience 15, dropout 0.4).

The trained checkpoint is included at:

```
Backend/app/Models/Experiments/grey_area/data/original/original_best.pt
```

This is the checkpoint loaded by the FastAPI backend at startup (`app/services/predict_service.py`) and used to produce all results in the dissertation.

To retrain from scratch:

```bash
cd Backend/app/Models/Experiments/grey_area
python prepare_data.py          # processes raw BindingDB into train/val/test splits
python train_and_compare.py original
```

`Backend/app/Models/Torch/run_dnn_adam.py` is the earlier standalone training script used during the iterative experiments in Chapter 5; it trains the same architecture but on the 2% subset used for the iteration comparisons, not the full dataset.

Training assumes the processed dataset and cached ProtBERT embeddings are present under `Backend/app/data/` and `Backend/app/Models/Experiments/grey_area/data/original/`. The raw BindingDB TSV is not shipped with this submission — see the Dataset section below.

## Dataset

The model was trained on BindingDB (November 2025 release). Due to Learn's 800 MB upload limit, the raw BindingDB archive is **not** included in this submission. The original source is:

> BindingDB: https://www.bindingdb.org/ — `BindingDB_All_202511.tsv`

Processed NumPy arrays required to run the backend and reproduce inference are included. To regenerate the dataset from scratch, download the raw TSV into `Backend/app/data/raw/` and run:

```bash
cd Backend/app/Models/Experiments/grey_area
python prepare_data.py
```

## Key Results

The final model achieves **AUC-ROC 0.970** and **F1 0.931** on the BindingDB test split (see Chapter 5 of the dissertation and `Backend/app/Models/Experiments/grey_area/figures/original_test_metrics.json`). The optimal decision threshold is τ = 0.3.

## What Is Not Included

To stay within the 800 MB submission cap, the following have been removed:

- `.git/`, `node_modules/`, Python `__pycache__/`, virtual environments
- Raw BindingDB archive and FASTA files (download from source if needed)
- Intermediate dataset variants (`grey_area/data/{wide,medium,narrow,no_grey}/`) used only for the ablation study in Chapter 5; their summary JSONs and figures are retained
- Redundant checkpoints; the two used by the API and reported in the dissertation are retained (`original_best.pt`, `dnn_adam_best.pt`)

## Notes

- Autocomplete suggestions are limited to proteins with pre-computed ProtBERT embeddings. Novel targets will not appear in suggestions and cannot be predicted at inference time because embedding on demand is disabled for latency reasons (see Chapter 6, Limitations).
- Batch virtual screening is capped at 100 compounds per request.
- Prediction history persists to `Backend/app/data/history.json` and is limited to the latest 100 entries.

## Academic Integrity

This codebase is submitted in fulfilment of 25COC251 and should not be reused, redistributed, or submitted by any other student.
