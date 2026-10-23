"""FastAPI service that returns churn risk scores for customers.

The score ranks customers by churn risk but is not a calibrated probability (see README,
"Limitations"). Instead of a fixed 0.5 cut-off, `high_risk` flags customers whose score would put
them in the riskiest 20% of the training customers, which matches how a retention campaign
picks its target group.

Every scored customer is appended to a JSON lines file (PREDICTION_LOG, default
logs/predictions.jsonl; set it to an empty string to disable), so that `python -m src.monitor
--prediction-log` can measure drift on the traffic the service actually receives.

Run locally:  uvicorn src.api:app --reload   ->  http://127.0.0.1:8000/docs
The model is loaded from MODEL_PATH (default: model/, created by src.export_model).
"""
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.features import FEATURES

HIGH_RISK_SHARE = 0.2  # flag the riskiest 20%
FALLBACK_THRESHOLD = 0.5  # only used if the model has no reference data
state: dict = {}
logger = logging.getLogger("churn_api")
_log_lock = threading.Lock()  # requests are served from a thread pool


class Customer(BaseModel):
    """RFM features of one customer, as of the scoring date."""

    recency_days: float = Field(ge=0, description="Days since the last purchase")
    frequency: float = Field(ge=1, description="Number of distinct orders")
    monetary: float = Field(description="Total spend")
    avg_order_value: float = Field(description="Spend per order")
    n_products: float = Field(ge=0, description="Number of distinct products bought")
    tenure_days: float = Field(ge=0, description="Days since the first purchase")
    return_rate: float = Field(ge=0, le=1, description="Share of order lines that were returned")
    orders_last_90d: float = Field(ge=0, description="Orders in the last 90 days")
    country: str = Field(description="Most frequent country of the customer")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recency_days": 45,
                "frequency": 6,
                "monetary": 1250.5,
                "avg_order_value": 208.4,
                "n_products": 38,
                "tenure_days": 300,
                "return_rate": 0.02,
                "orders_last_90d": 1,
                "country": "United Kingdom",
            }
        }
    )


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_score: float = Field(description="Churn risk score in [0, 1]; higher means more likely to churn")
    high_risk: bool = Field(description="Score is in the top 20% of the training customers")


class PredictResponse(BaseModel):
    model_version: str | None
    predictions: list[Prediction]


def load_model(path: Path) -> None:
    state["model"] = mlflow.sklearn.load_model(str(path))
    meta_path = path / "serving_meta.json"
    state["version"] = str(json.loads(meta_path.read_text())["version"]) if meta_path.exists() else None
    ref_path = path / "reference.parquet"
    if ref_path.exists():
        ref_scores = state["model"].predict_proba(pd.read_parquet(ref_path)[FEATURES])[:, 1]
        state["threshold"] = float(np.quantile(ref_scores, 1 - HIGH_RISK_SHARE))
    else:
        state["threshold"] = FALLBACK_THRESHOLD


def log_predictions(df: pd.DataFrame, scores, high_risk) -> None:
    """Append one JSON line per scored customer. Logging must never break a prediction."""
    path = state.get("log_path")
    if not path:
        return
    request_id = uuid.uuid4().hex
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    records = df.assign(churn_score=scores, high_risk=high_risk).to_dict("records")
    lines = "".join(
        json.dumps({"timestamp": timestamp, "request_id": request_id, "model_version": state["version"], **r}) + "\n"
        for r in records
    )
    try:
        with _log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(lines)
    except OSError:
        logger.exception("Could not write prediction log to %s", path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(Path(os.getenv("MODEL_PATH", "model")))
    log_path = os.getenv("PREDICTION_LOG", "logs/predictions.jsonl")
    state["log_path"] = Path(log_path) if log_path else None
    yield
    state.clear()


app = FastAPI(title="Online Retail Churn API", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "model_version": state.get("version"),
        "high_risk_threshold": round(state["threshold"], 4) if "threshold" in state else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    df = pd.DataFrame([c.model_dump() for c in request.customers])[FEATURES]
    scores = state["model"].predict_proba(df)[:, 1].round(4)
    high_risk = scores >= state["threshold"]
    log_predictions(df, scores, high_risk)
    return PredictResponse(
        model_version=state["version"],
        predictions=[Prediction(churn_score=float(s), high_risk=bool(h)) for s, h in zip(scores, high_risk)],
    )
