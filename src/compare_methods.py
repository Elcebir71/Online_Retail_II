"""Compare our results with published churn studies, and test their methods in our setup.

Published churn models often report 91-99% accuracy, while ours reaches ROC-AUC 0.745. This script
shows where the gap comes from, in three experiments on the same data:

1. Label leakage (as in a 2026 Scientific Reports study on Online Retail): churn is defined as
   "recency > 90 days" at the end of the data and Recency is also a model input, with a random
   80/20 split. The model only has to re-learn its own label rule, so it scores ~0.99 without
   predicting anything about the future.
2. SMOTE before the split (as in several bank churn studies): oversampling the whole dataset first
   puts synthetic copies of test customers into the training data and into the test set. Compared
   with SMOTE applied to the training data only, on the same random split.
3. Their methods in our setup: ANOVA feature selection, PCA, SMOTE (training data only), random
   forest, gradient boosting, AdaBoost, KNN and SVM, all trained on 2010 customers and tested on
   2011 customers (label = no purchase in the next 90 days), with the recency rule as baseline and
   95% bootstrap intervals of the ROC-AUC gain.

Notes: gradient boosting is scikit-learn's HistGradientBoostingClassifier (an XGBoost-style
model, no extra dependency). Genetic-algorithm feature selection is not reproduced; ANOVA
selection stands in for feature selection.

Usage:
    python -m src.compare_methods [--n-boot 1000]
Writes reports/comparison.md.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.data import PROCESSED_PATH
from src.evaluate import METRIC_LABELS, bootstrap, compute_metrics
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_dataset,
    build_features,
    signed_log1p,
)
from src.train import DEFAULT_LOOKBACK_DAYS, HORIZON_DAYS, TEST_CUTOFF, TRAIN_CUTOFF

REPORT_DIR = Path("reports")
SEED = 42
# The Online Retail (I) dataset used by the label-leakage study covers 2010-12-01 to 2011-12-09
ONLINE_RETAIL_I_START = "2010-12-01"


def smote(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Minimal SMOTE: add synthetic minority rows until both classes have the same size.

    Each synthetic row lies on the line between a minority row and one of its k nearest
    minority neighbours (Chawla et al., 2002).
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[np.argmin(counts)]
    n_new = counts.max() - counts.min()
    X_min = X[y == minority]
    if n_new == 0 or len(X_min) < 2:
        return X, y
    neighbours = NearestNeighbors(n_neighbors=min(k, len(X_min) - 1) + 1).fit(X_min)
    _, idx = neighbours.kneighbors(X_min)
    base = rng.integers(0, len(X_min), n_new)
    partner = idx[base, rng.integers(1, idx.shape[1], n_new)]  # column 0 is the row itself
    gap = rng.random((n_new, 1))
    X_new = X_min[base] + gap * (X_min[partner] - X_min[base])
    return np.vstack([X, X_new]), np.r_[y, np.full(n_new, minority)]


class SMOTEClassifier(ClassifierMixin, BaseEstimator):
    """Oversample with SMOTE inside fit(), so only training data is ever resampled."""

    def __init__(self, estimator, k: int = 5):
        self.estimator = estimator
        self.k = k

    def fit(self, X, y):
        X_res, y_res = smote(np.asarray(X, dtype=float), np.asarray(y), self.k)
        self.estimator_ = clone(self.estimator).fit(X_res, y_res)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def preprocessor() -> ColumnTransformer:
    """Same preprocessing as the served logistic regression: log + scale, one-hot country."""
    numeric = Pipeline([("log", FunctionTransformer(signed_log1p)), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)], sparse_threshold=0
    )


def logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def candidate_models() -> dict[str, list]:
    """Model steps that follow the shared preprocessing. The first entry is our served setup."""
    return {
        "Logistic regression (ours)": [("model", logreg())],
        "+ ANOVA top-5 features": [("select", SelectKBest(f_classif, k=5)), ("model", logreg())],
        "+ PCA (95% variance)": [("pca", PCA(n_components=0.95, random_state=SEED)), ("model", logreg())],
        "+ SMOTE (train only)": [("model", SMOTEClassifier(LogisticRegression(max_iter=1000)))],
        "Random forest": [
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED))
        ],
        "Gradient boosting": [("model", HistGradientBoostingClassifier(learning_rate=0.05, random_state=SEED))],
        "AdaBoost": [("model", AdaBoostClassifier(n_estimators=200, random_state=SEED))],
        "KNN (k=15)": [("model", KNeighborsClassifier(n_neighbors=15))],
        "KNN (k=15) + SMOTE (train only)": [("model", SMOTEClassifier(KNeighborsClassifier(n_neighbors=15)))],
        # Probabilities via Platt scaling on cross-validated predictions (SVC(probability=True) is deprecated)
        "SVM (RBF)": [("model", CalibratedClassifierCV(SVC(), cv=5, ensemble=False))],
    }


def make(steps: list) -> Pipeline:
    return Pipeline([("pre", preprocessor()), *[(name, clone(step)) for name, step in steps]])


def leaky_label_experiment(tx: pd.DataFrame) -> dict[str, float]:
    """Experiment 1: label = recency > 90 days, computed from the same features the model sees."""
    tx = tx[tx["invoice_date"] >= ONLINE_RETAIL_I_START]
    end = tx["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    df = build_features(tx, end)
    y = (df["recency_days"] > 90).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)
    model = make([("model", logreg())]).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "customers": len(df),
        "churn_rate": float(y.mean()),
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, proba >= 0.5),
    }


def smote_before_split_experiment(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Experiment 2: SMOTE on all data before a random split vs. on the training part only."""
    pre = preprocessor().fit(train[FEATURES])  # unsupervised scaling only; kept identical for both variants
    X, y = pre.transform(train[FEATURES]), train[TARGET].to_numpy()
    knn = KNeighborsClassifier(n_neighbors=5)
    results = {}

    X_all, y_all = smote(X, y)
    Xa, Xb, ya, yb = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
    proba = clone(knn).fit(Xa, ya).predict_proba(Xb)[:, 1]
    results["SMOTE before split (leaky)"] = {
        "roc_auc": roc_auc_score(yb, proba),
        "accuracy": accuracy_score(yb, proba >= 0.5),
        "test_churn_rate": float(yb.mean()),
    }

    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    proba = SMOTEClassifier(knn).fit(Xa, ya).predict_proba(Xb)[:, 1]
    results["SMOTE on training part only"] = {
        "roc_auc": roc_auc_score(yb, proba),
        "accuracy": accuracy_score(yb, proba >= 0.5),
        "test_churn_rate": float(yb.mean()),
    }
    return results


def method_comparison(train: pd.DataFrame, test: pd.DataFrame, n_boot: int) -> tuple[dict, dict, dict]:
    """Experiment 3: every method on our temporal split; the recency rule is the reference."""
    y = test[TARGET].to_numpy()
    scores = {"Recency rule": test["recency_days"].to_numpy(dtype=float)}
    accuracy = {"Recency rule": np.nan}
    for name, steps in candidate_models().items():
        print(f"  fitting {name} ...")
        model = make(steps).fit(train[FEATURES], train[TARGET])
        proba = model.predict_proba(test[FEATURES])[:, 1]
        scores[name] = proba
        accuracy[name] = accuracy_score(y, proba >= 0.5)
    point = {name: compute_metrics(y, s) for name, s in scores.items()}
    return point, bootstrap(y, scores, n_boot=n_boot), accuracy


def render_markdown(leaky: dict, smote_results: dict, point: dict, boot: dict, accuracy: dict, n_test: int) -> str:
    lines = [
        "## 1. Label leakage: churn = recency > 90 days, recency is also an input",
        "",
        (
            f"Online Retail I period ({ONLINE_RETAIL_I_START} onwards), {leaky['customers']:,} customers, "
            f"churn rate {leaky['churn_rate']:.1%}, random stratified 80/20 split, our logistic regression."
        ),
        "",
        "| Setup | ROC-AUC | Accuracy |",
        "|---|---|---|",
        f"| Leaky label (published setup) | {leaky['roc_auc']:.3f} | {leaky['accuracy']:.3f} |",
        (
            f"| Our setup (label from the next 90 days) | {point['Logistic regression (ours)']['roc_auc']:.3f} "
            f"| {accuracy['Logistic regression (ours)']:.3f} |"
        ),
        "",
        "## 2. SMOTE before vs. after the split (KNN, k=5, random 80/20 split of 2010 customers)",
        "",
        "| Setup | ROC-AUC | Accuracy | Churn rate in test set |",
        "|---|---|---|---|",
    ]
    for name, r in smote_results.items():
        lines.append(f"| {name} | {r['roc_auc']:.3f} | {r['accuracy']:.3f} | {r['test_churn_rate']:.1%} |")

    header = "| Model | " + " | ".join(METRIC_LABELS.values()) + " | Accuracy | ROC-AUC gain vs. recency rule |"
    lines += [
        "",
        "## 3. Published methods in our setup",
        "",
        (
            f"Trained on customers as of {TRAIN_CUTOFF}, tested on {n_test:,} customers as of {TEST_CUTOFF}; "
            f"{DEFAULT_LOOKBACK_DAYS}-day feature window, label = no purchase in the next {HORIZON_DAYS} days. "
            "Brackets: 95% paired bootstrap interval."
        ),
        "",
        header,
        "|---" * (len(METRIC_LABELS) + 3) + "|",
    ]
    for name, metrics in point.items():
        cells = []
        for key, value in metrics.items():
            low, high = boot["intervals"][name][key]
            cells.append(f"{value:.3f} [{low:.3f}–{high:.3f}]" if key == "roc_auc" else f"{value:.3f}")
        acc = "–" if np.isnan(accuracy[name]) else f"{accuracy[name]:.3f}"
        if name == boot["reference"]:
            gain = "reference"
        else:
            low, high = boot["auc_gain_vs_reference"][name]
            gain = f"{metrics['roc_auc'] - point[boot['reference']]['roc_auc']:+.3f} [{low:+.3f} to {high:+.3f}]"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {acc} | {gain} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap resamples")
    args = parser.parse_args()

    tx = pd.read_parquet(PROCESSED_PATH)
    train = build_dataset(tx, TRAIN_CUTOFF, HORIZON_DAYS, DEFAULT_LOOKBACK_DAYS)
    test = build_dataset(tx, TEST_CUTOFF, HORIZON_DAYS, DEFAULT_LOOKBACK_DAYS)

    print("1/3 label leakage")
    leaky = leaky_label_experiment(tx)
    print("2/3 SMOTE before split")
    smote_results = smote_before_split_experiment(train)
    print("3/3 published methods in our setup")
    point, boot, accuracy = method_comparison(train, test, args.n_boot)

    REPORT_DIR.mkdir(exist_ok=True)
    md_path = REPORT_DIR / "comparison.md"
    md_path.write_text(render_markdown(leaky, smote_results, point, boot, accuracy, len(test)), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"Written to {md_path}")


if __name__ == "__main__":
    main()
