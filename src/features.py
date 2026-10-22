"""Build customer-level RFM features and a churn label from invoice lines.

Idea: pick a cutoff date. Features are computed only from transactions BEFORE the
cutoff; the label checks whether the customer made any purchase in the
`horizon_days` days after the cutoff (no purchase -> churn = 1).
"""
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


def build_features(tx: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return one row per customer with at least one purchase before the cutoff."""
    hist = tx[tx["invoice_date"] < cutoff]
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


def build_dataset(tx: pd.DataFrame, cutoff: str | pd.Timestamp, horizon_days: int = 90) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + _days(horizon_days)
    if tx["invoice_date"].max() < end - _days(1):
        raise ValueError(f"Data does not extend to {end:%Y-%m-%d}; the label window would be incomplete.")
    feats = build_features(tx, cutoff)
    feats[TARGET] = build_labels(tx, feats.index, cutoff, horizon_days)
    return feats


def signed_log1p(x):
    """Log transform for skewed (and possibly negative) values."""
    return np.sign(x) * np.log1p(np.abs(x))
