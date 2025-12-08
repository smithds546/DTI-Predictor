from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict, history, health, autocomplete

app = FastAPI(
    title="DTI Predictor API",
    version="0.1.0",
)

# CORS for Gatsby frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(autocomplete.router)