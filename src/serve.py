"""
FastAPI serving layer for the fine-tuned sentiment classifier.

Loads the Production model (LoRA adapter) from the MLflow registry on startup.
Exposes a /predict endpoint for single or batch inference. Includes basic
health checks and a /reload endpoint so a new Production version can be
picked up without restarting the server - useful for canary promotion.

Run:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d "{\"texts\": [\"this movie was amazing\", \"absolutely terrible film\"]}"
"""
import os
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


# ----- Config loading -----
ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

BASE_MODEL = CONFIG["model"]["base_model"]
NUM_LABELS = CONFIG["model"]["num_labels"]
MAX_LENGTH = CONFIG["data"]["max_length"]
MODEL_NAME = CONFIG["mlflow"]["registered_model_name"]
TRACKING_URI = CONFIG["mlflow"]["tracking_uri"]

LABEL_MAP = {0: "negative", 1: "positive"}

# Global model state - populated on startup
STATE = {"model": None, "tokenizer": None, "version": None, "stage": None}


def _load_production_model():
    """Fetch the current Production model from MLflow registry and load it."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Find the Production version; fall back to latest if none promoted yet
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    stage = "Production"
    if not versions:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        stage = "Staging"
    if not versions:
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not all_versions:
            raise RuntimeError(f"No versions of {MODEL_NAME} found in registry")
        versions = [sorted(all_versions, key=lambda v: int(v.version))[-1]]
        stage = versions[0].current_stage or "None"

    version = versions[0].version
    run_id = versions[0].run_id

    # Download LoRA adapter artifacts for this run
    local_dir = client.download_artifacts(run_id, "lora_adapter")

    # Rebuild: base model + LoRA adapter on top
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS
    )
    model = PeftModel.from_pretrained(base, local_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    STATE["version"] = version
    STATE["stage"] = stage
    print(f"[serve] loaded {MODEL_NAME} v{version} ({stage})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_production_model()
    yield
    # nothing to clean up


app = FastAPI(
    title="Sentiment Classifier API",
    description="Fine-tuned DistilBERT with LoRA, served from MLflow registry",
    version="1.0.0",
    lifespan=lifespan,
)


# ----- Schemas -----
class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64,
                              description="List of texts to classify")


class Prediction(BaseModel):
    text: str
    label: str
    confidence: float
    probs: dict


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_version: str
    model_stage: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: Optional[str]
    model_stage: Optional[str]


# ----- Endpoints -----
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if STATE["model"] is not None else "no_model",
        model_loaded=STATE["model"] is not None,
        model_name=MODEL_NAME,
        model_version=STATE["version"],
        model_stage=STATE["stage"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tokenizer = STATE["tokenizer"]
    model = STATE["model"]

    enc = tokenizer(
        req.texts,
        truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).numpy()

    preds = probs.argmax(axis=-1)
    results = []
    for text, pred_idx, prob_row in zip(req.texts, preds, probs):
        results.append(Prediction(
            text=text,
            label=LABEL_MAP[int(pred_idx)],
            confidence=float(prob_row[pred_idx]),
            probs={LABEL_MAP[i]: float(p) for i, p in enumerate(prob_row)},
        ))

    return PredictResponse(
        predictions=results,
        model_version=STATE["version"],
        model_stage=STATE["stage"],
    )


@app.post("/reload")
def reload_model():
    """Hot-reload the model from the registry. Call after promoting a new version."""
    try:
        old_version = STATE["version"]
        _load_production_model()
        return {
            "status": "reloaded",
            "old_version": old_version,
            "new_version": STATE["version"],
            "stage": STATE["stage"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "service": "sentiment-classifier",
        "endpoints": ["/health", "/predict", "/reload", "/docs"],
    }