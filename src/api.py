"""FastAPI service that returns churn risk scores for customers.

The score ranks customers by churn risk but is not a calibrated probability (see README,
"Limitations"). Instead of a fixed 0.5 cut-off, `high_risk` flags customers whose score would put
them in the riskiest 20% of the training customers, which matches how a retention campaign
picks its target group.

Run locally:  uvicorn src.api:app --reload   ->  http://127.0.0.1:8000/docs
The model is loaded from MODEL_PATH (default: model/, created by src.export_model).
"""
import json
import os
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(Path(os.getenv("MODEL_PATH", "model")))
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
    proba = state["model"].predict_proba(df)[:, 1]
    return PredictResponse(
        model_version=state["version"],
        predictions=[
            Prediction(churn_score=round(float(p), 4), high_risk=bool(p >= state["threshold"])) for p in proba
        ],
    )
