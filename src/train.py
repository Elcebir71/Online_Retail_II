"""Train and evaluate the churn model and log it to MLflow.

Time-based split: the model is trained on customers as of autumn 2010 and tested on
customers one year later (autumn 2011). This avoids the leakage from the future that
a random split would cause, and keeps seasonality the same on both sides.

Usage:    python -m src.train            (first: python -m src.data)
Results:  mlflow ui   ->  http://127.0.0.1:5000
"""
import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import PROCESSED_PATH
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_dataset,
    signed_log1p,
)

TRAIN_CUTOFF = "2010-09-09"
TEST_CUTOFF = "2011-09-09"
HORIZON_DAYS = 90
EXPERIMENT = "online-retail-churn"
MODEL_NAME = "churn-model"
TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "src.features.signed_log1p",
]


def make_pipeline(model_type: str) -> Pipeline:
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    if model_type == "logreg":
        # Log transform + scaling for skewed distributions such as spend
        numeric = Pipeline([("log", FunctionTransformer(signed_log1p)), ("scale", StandardScaler())])
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_type == "hgb":
        numeric = "passthrough"
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    pre = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        sparse_threshold=0,
    )
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(pipe: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    proba = pipe.predict_proba(df[FEATURES])[:, 1]
    return {
        "roc_auc": roc_auc_score(df[TARGET], proba),
        "pr_auc": average_precision_score(df[TARGET], proba),
        "f1": f1_score(df[TARGET], proba >= 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "hgb"], default="logreg")
    parser.add_argument("--register", action="store_true", help="Register the model in the MLflow Model Registry")
    args = parser.parse_args()

    tx = pd.read_parquet(PROCESSED_PATH)
    train = build_dataset(tx, TRAIN_CUTOFF, HORIZON_DAYS)
    test = build_dataset(tx, TEST_CUTOFF, HORIZON_DAYS)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.model):
        mlflow.log_params(
            {
                "model_type": args.model,
                "train_cutoff": TRAIN_CUTOFF,
                "test_cutoff": TEST_CUTOFF,
                "horizon_days": HORIZON_DAYS,
                "n_train": len(train),
                "n_test": len(test),
                "train_churn_rate": round(train[TARGET].mean(), 4),
                "test_churn_rate": round(test[TARGET].mean(), 4),
            }
        )
        pipe = make_pipeline(args.model).fit(train[FEATURES], train[TARGET])
        metrics = evaluate(pipe, test)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # Store the reference data (training set) with the model for drift monitoring
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "reference.parquet"
            train.to_parquet(ref_path)
            mlflow.log_artifact(str(ref_path))

        mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(train[FEATURES], pipe.predict_proba(train[FEATURES])[:, 1]),
            input_example=train[FEATURES].head(3),
            registered_model_name=MODEL_NAME if args.register else None,
            # MLflow 3 saves models in the safe skops format; explicitly trust our own types
            skops_trusted_types=TRUSTED_TYPES,
            pyfunc_predict_fn="predict_proba",
        )
        print(f"Train: {len(train):,} customers (churn {train[TARGET].mean():.1%})")
        print(f"Test:  {len(test):,} customers (churn {test[TARGET].mean():.1%})")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
