"""Prediction service.   Run:  uvicorn app.main:app --port 8000"""
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.cinema.cleaning import FEATURES, clean_language

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
_model = None          # loaded once, on first use (tests can put a tiny model here)


# JSON logging
class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record):
        entry = {"time": self.formatTime(record), "level": record.levelname,
                 "message": record.getMessage()}
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry)


logger = logging.getLogger("cinema_api")
if not logger.handlers:                          # avoid duplicate lines on reload
    handler = logging.StreamHandler(sys.stdout)  # screen only, never a file
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# request schema
class PredictRequest(BaseModel):
    ticket_price_rs: float = Field(..., ge=50)             # rule 2: below 50 = typo
    days_since_release: float
    star_rating: float
    show_hour: float
    screen_seats: float
    distance_to_metro_km: float = Field(..., le=8.5)       # rule 1: above 8.5 impossible
    language: str
    location: str

    @field_validator("language")
    @classmethod
    def tidy_language(cls, v):
        return clean_language(v)

    @field_validator("location")
    @classmethod
    def tidy_location(cls, v):
        return v.strip()


class PredictResponse(BaseModel):
    prediction: float


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="model file not found")
        _model = joblib.load(MODEL_PATH)
    return _model


app = FastAPI(title="Cinema Show Occupancy API (MP152)")


@app.get("/health")
def health():
    return {"status": "ok", "model_file_present": MODEL_PATH.exists() or _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    model = get_model()
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    value = float(model.predict(row)[0])
    value = min(max(value, 0.0), 100.0)                    # occupancy is a percent
    logger.info("prediction", extra={"extra_fields": {
        "location": req.location, "language": req.language,
        "prediction": round(value, 2),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2)}})
    return PredictResponse(prediction=round(value, 4))