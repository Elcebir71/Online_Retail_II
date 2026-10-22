"""FastAPI service that returns churn probabilities for customers.

Run locally:  uvicorn src.api:app --reload   ->  http://127.0.0.1:8000/docs
The model is loaded from MODEL_PATH (default: model/, created by src.export_model).
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.features import FEATURES

THRESHOLD = 0.5
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
    churn_probability: float
    churn: bool


class PredictResponse(BaseModel):
    model_version: str | None
    predictions: list[Prediction]


def load_model(path: Path) -> None:
    state["model"] = mlflow.sklearn.load_model(str(path))
    meta_path = path / "serving_meta.json"
    state["version"] = str(json.loads(meta_path.read_text())["version"]) if meta_path.exists() else None


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
    return {"status": "ok", "model_loaded": "model" in state, "model_version": state.get("version")}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    df = pd.DataFrame([c.model_dump() for c in request.customers])[FEATURES]
    proba = state["model"].predict_proba(df)[:, 1]
    return PredictResponse(
        model_version=state["version"],
        predictions=[Prediction(churn_probability=round(float(p), 4), churn=bool(p >= THRESHOLD)) for p in proba],
    )
