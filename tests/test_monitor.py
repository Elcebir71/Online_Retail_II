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
