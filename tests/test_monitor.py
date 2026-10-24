import json

import numpy as np
import pandas as pd
import pytest

from src.monitor import (
    PSI_DRIFT,
    PSI_WARNING,
    baseline_performance,
    psi_categorical,
    psi_numeric,
    psi_status,
)

rng = np.random.default_rng(42)


def test_psi_is_near_zero_for_same_distribution():
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(ref, cur) < PSI_WARNING


def test_psi_detects_shift():
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(1.5, 1, 5000))
    assert psi_numeric(ref, cur) >= PSI_DRIFT


def test_psi_handles_discrete_values():
    # Many ties produce duplicate quantile edges; must not crash
    ref = pd.Series(rng.integers(0, 3, 1000).astype(float))
    cur = pd.Series(rng.integers(0, 3, 1000).astype(float))
    assert psi_numeric(ref, cur) == pytest.approx(0, abs=0.05)


def test_psi_categorical():
    ref = pd.Series(["UK"] * 90 + ["France"] * 10)
    assert psi_categorical(ref, ref.copy()) == pytest.approx(0)
    shifted = pd.Series(["UK"] * 40 + ["France"] * 30 + ["Spain"] * 30)
    assert psi_categorical(ref, shifted) >= PSI_DRIFT


@pytest.mark.parametrize("value,expected", [(0.05, "ok"), (0.15, "warning"), (0.4, "drift")])
def test_psi_status(value, expected):
    assert psi_status(value) == expected


def test_baseline_performance_reads_test_metrics(tmp_path):
    (tmp_path / "serving_meta.json").write_text(
        json.dumps({"version": "3", "test_metrics": {"roc_auc": 0.7712, "pr_auc": 0.79, "f1": 0.69}})
    )
    assert baseline_performance(tmp_path) == {"roc_auc": 0.7712, "pr_auc": 0.79}


def test_baseline_performance_missing(tmp_path):
    assert baseline_performance(tmp_path) is None
    (tmp_path / "serving_meta.json").write_text(json.dumps({"version": "2"}))
    assert baseline_performance(tmp_path) is None


def test_evaluate_live_reads_api_prediction_log(tmp_path):
    from sklearn.dummy import DummyClassifier

    from src.features import FEATURES
    from src.monitor import evaluate_live, read_prediction_log

    n = 500
    reference = pd.DataFrame({f: rng.normal(0, 1, n) for f in FEATURES})
    reference["country"] = "United Kingdom"
    log = reference.assign(
        recency_days=reference["recency_days"] + 3,  # shifted feature
        churn_score=rng.random(n),
        high_risk=False,
        model_version="6",
        request_id="abc",
        timestamp="2026-10-23T06:00:00+00:00",
    )
    log_path = tmp_path / "predictions.jsonl"
    log.to_json(log_path, orient="records", lines=True)

    model = DummyClassifier(strategy="uniform", random_state=0).fit(reference[FEATURES], rng.integers(0, 2, n))
    result, _ = evaluate_live(model, reference, read_prediction_log(log_path), {"roc_auc": 0.77, "pr_auc": 0.79})
    assert result.n_customers == n
    assert result.performance is None
    assert result.drifted_features == ["recency_days"]
    assert read_prediction_log(log_path, since="2026-10-24").empty
