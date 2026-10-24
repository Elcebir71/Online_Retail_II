import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.compare_methods import SMOTEClassifier, leaky_label_experiment, smote

rng = np.random.default_rng(0)


def test_smote_balances_classes_with_points_between_minority_rows():
    X = np.r_[rng.normal(0, 1, (90, 2)), rng.normal(10, 1, (10, 2))]
    y = np.r_[np.zeros(90), np.ones(10)]
    X_res, y_res = smote(X, y)
    assert (y_res == 1).sum() == (y_res == 0).sum() == 90
    synthetic = X_res[len(X) :]
    minority = X[y == 1]
    assert (synthetic >= minority.min(axis=0)).all()
    assert (synthetic <= minority.max(axis=0)).all()


def test_smote_classifier_resamples_only_in_fit():
    X = rng.normal(0, 1, (100, 3))
    y = (X[:, 0] > 0.8).astype(int)
    model = SMOTEClassifier(LogisticRegression()).fit(X, y)
    assert model.predict_proba(X).shape == (100, 2)
    assert list(model.classes_) == [0, 1]


@pytest.fixture
def tx():
    # 300 customers with random order dates over 2010-12 .. 2011-12
    days = pd.date_range("2010-12-01", "2011-12-09")
    rows = []
    for customer in range(300):
        n_orders = int(rng.integers(1, 6))
        for i, day in enumerate(rng.choice(days, n_orders)):
            rows.append(
                {
                    "invoice": f"{customer}-{i}",
                    "stock_code": "A",
                    "quantity": 1,
                    "invoice_date": pd.Timestamp(day),
                    "price": 5.0,
                    "customer_id": customer,
                    "country": "United Kingdom",
                    "is_return": False,
                    "revenue": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_leaky_label_is_almost_perfectly_predictable(tx):
    # The label is a rule on an input feature, so even random behaviour is "predicted"
    result = leaky_label_experiment(tx)
    assert result["roc_auc"] > 0.98
