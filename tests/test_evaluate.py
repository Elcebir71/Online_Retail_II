import numpy as np
import pytest

from src.evaluate import bootstrap, compute_metrics, lift_table, rank_order, top_k_metrics

rng = np.random.default_rng(0)


def test_perfect_score_reaches_all_churners_first():
    y = np.array([0] * 80 + [1] * 20)
    m = top_k_metrics(y, y.astype(float), k=0.2)
    assert m["precision"] == 1
    assert m["recall"] == 1
    assert m["lift"] == pytest.approx(5)


def test_ties_are_not_ranked_by_input_order():
    # A constant score must behave like random targeting, not pick the first rows
    y = np.array([1] * 50 + [0] * 950)
    m = top_k_metrics(y, np.zeros(len(y)), k=0.05)
    assert m["precision"] < 0.5


def test_rank_order_puts_highest_score_first():
    assert rank_order(np.array([0.1, 0.9, 0.5]))[0] == 1


def test_lift_table_covers_all_customers():
    y = rng.integers(0, 2, 1000)
    table = lift_table(y, rng.random(1000))
    assert table["customers"].sum() == 1000
    assert table["churners"].sum() == y.sum()
    assert table["cumulative_recall"].iloc[-1] == pytest.approx(1)


def test_bootstrap_interval_contains_point_estimate():
    y = rng.integers(0, 2, 500)
    scores = {"baseline": rng.random(500), "model": y + rng.normal(0, 0.8, 500)}
    boot = bootstrap(y, scores, n_boot=200)
    low, high = boot["intervals"]["model"]["roc_auc"]
    assert low <= compute_metrics(y, scores["model"])["roc_auc"] <= high
    gain_low, _ = boot["auc_gain_vs_reference"]["model"]
    assert gain_low > 0  # the informative model clearly beats the random baseline
