import json

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import Customer, app
from src.features import FEATURES, TARGET
from src.train import TRUSTED_TYPES, make_pipeline

EXAMPLE = Customer.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Train a tiny model on random data so the API can be tested without the real dataset."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "recency_days": rng.uniform(0, 300, n),
            "frequency": rng.integers(1, 20, n).astype(float),
            "monetary": rng.uniform(10, 5000, n),
            "avg_order_value": rng.uniform(10, 500, n),
            "n_products": rng.integers(1, 100, n).astype(float),
            "tenure_days": rng.uniform(0, 700, n),
            "return_rate": rng.uniform(0, 0.2, n),
            "orders_last_90d": rng.integers(0, 5, n).astype(float),
            "country": rng.choice(["United Kingdom", "France"], n),
        }
    )
    df[TARGET] = (df["recency_days"] > 150).astype(int)
    pipe = make_pipeline("logreg").fit(df[FEATURES], df[TARGET])

    model_dir = tmp_path_factory.mktemp("serving") / "model"
    mlflow.sklearn.save_model(pipe, str(model_dir), skops_trusted_types=TRUSTED_TYPES)
    (model_dir / "serving_meta.json").write_text(json.dumps({"version": 7}))

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_PATH", str(model_dir))
        with TestClient(app) as c:
            yield c


def test_root_redirects_to_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/docs"


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "7"


def test_predict(client):
    recent = {**EXAMPLE, "recency_days": 5}
    inactive = {**EXAMPLE, "recency_days": 290}
    resp = client.post("/predict", json={"customers": [recent, inactive]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert resp.json()["model_version"] == "7"
    assert len(preds) == 2
    assert all(0 <= p["churn_probability"] <= 1 for p in preds)
    assert preds[1]["churn_probability"] > preds[0]["churn_probability"]


def test_unknown_country_is_accepted(client):
    resp = client.post("/predict", json={"customers": [{**EXAMPLE, "country": "Narnia"}]})
    assert resp.status_code == 200


def test_invalid_input_is_rejected(client):
    assert client.post("/predict", json={"customers": [{**EXAMPLE, "return_rate": 3}]}).status_code == 422
    assert client.post("/predict", json={"customers": []}).status_code == 422
