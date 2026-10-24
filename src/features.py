"""Build customer-level RFM features and a churn label from invoice lines.

Idea: pick a cutoff date. Features are computed only from transactions BEFORE the
cutoff; the label checks whether the customer made any purchase in the
`horizon_days` days after the cutoff (no purchase -> churn = 1).

Lookback window: with the full history, cumulative features (tenure, frequency, spend) keep
growing as the dataset gets older, so they drift even when customer behaviour does not. With
`lookback_days`, only the last N days before the cutoff are used, so every snapshot sees the
same amount of history. Customers without a purchase in that window are left out.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "n_products",
    "tenure_days",
    "return_rate",
    "orders_last_90d",
]
CATEGORICAL_FEATURES = ["country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"


def _days(n: int) -> np.timedelta64:
    # Explicit unit avoids NumPy's deprecation warning for unit-less timedeltas
    return np.timedelta64(int(n), "D")


def build_features(tx: pd.DataFrame, cutoff: pd.Timestamp, lookback_days: int | None = None) -> pd.DataFrame:
    """Return one row per customer with at least one purchase in the history window."""
    hist = tx[tx["invoice_date"] < cutoff]
    if lookback_days:
        start = cutoff - _days(lookback_days)
        if tx["invoice_date"].min() > start:
            raise ValueError(f"Data starts after {start:%Y-%m-%d}; a {lookback_days}-day lookback is not available.")
        hist = hist[hist["invoice_date"] >= start]
    purchases = hist[~hist["is_return"]]
    returns = hist[hist["is_return"]]

    g = purchases.groupby("customer_id")
    feats = pd.DataFrame(
        {
            "last_purchase": g["invoice_date"].max(),
            "first_purchase": g["invoice_date"].min(),
            "frequency": g["invoice"].nunique(),
            "monetary": g["revenue"].sum(),
            "n_products": g["stock_code"].nunique(),
            "n_lines": g.size(),
            "country": g["country"].agg(lambda s: s.mode().iat[0]),
        }
    )
    recent = purchases[purchases["invoice_date"] >= cutoff - _days(90)]
    feats["orders_last_90d"] = recent.groupby("customer_id")["invoice"].nunique()
    feats["n_return_lines"] = returns.groupby("customer_id").size()
    feats = feats.fillna({"orders_last_90d": 0, "n_return_lines": 0})

    feats["recency_days"] = (cutoff - feats["last_purchase"]).dt.days
    feats["tenure_days"] = (cutoff - feats["first_purchase"]).dt.days
    feats["avg_order_value"] = feats["monetary"] / feats["frequency"]
    feats["return_rate"] = feats["n_return_lines"] / (feats["n_lines"] + feats["n_return_lines"])
    # Use float64 everywhere so the MLflow schema accepts missing values at inference time
    feats[NUMERIC_FEATURES] = feats[NUMERIC_FEATURES].astype("float64")
    return feats[FEATURES]


def build_labels(tx: pd.DataFrame, customers: pd.Index, cutoff: pd.Timestamp, horizon_days: int) -> pd.Series:
    """1 (churn) if there is no purchase in the window, otherwise 0."""
    end = cutoff + _days(horizon_days)
    window = tx[(tx["invoice_date"] >= cutoff) & (tx["invoice_date"] < end) & (~tx["is_return"])]
    active = set(window["customer_id"])
    return pd.Series([0 if c in active else 1 for c in customers], index=customers, name=TARGET)


def build_dataset(
    tx: pd.DataFrame, cutoff: str | pd.Timestamp, horizon_days: int = 90, lookback_days: int | None = None
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + _days(horizon_days)
    if tx["invoice_date"].max() < end - _days(1):
        raise ValueError(f"Data does not extend to {end:%Y-%m-%d}; the label window would be incomplete.")
    feats = build_features(tx, cutoff, lookback_days)
    feats[TARGET] = build_labels(tx, feats.index, cutoff, horizon_days)
    return feats


def read_lookback_days(model_path: Path) -> int | None:
    """Lookback window the exported model was trained with (None = full history)."""
    meta_path = Path(model_path) / "serving_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text()).get("lookback_days")


def signed_log1p(x):
    """Log transform for skewed (and possibly negative) values."""
    return np.sign(x) * np.log1p(np.abs(x))
