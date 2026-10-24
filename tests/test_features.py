import json

import pandas as pd
import pytest

from src.data import clean
from src.features import FEATURES, TARGET, build_dataset, read_lookback_days


def _raw(rows):
    return pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
    )


@pytest.fixture
def tx():
    return clean(
        _raw(
            [
                # Customer 1: 2 orders + 1 return before cutoff, buys again after it -> churn 0
                ["1001", "A", "x", 2, "2010-01-10", 5.0, 1.0, "United Kingdom"],
                ["1002", "B", "y", 1, "2010-02-10", 10.0, 1.0, "United Kingdom"],
                ["C1003", "B", "y", -1, "2010-02-15", 10.0, 1.0, "United Kingdom"],
                ["1004", "A", "x", 1, "2010-03-15", 5.0, 1.0, "United Kingdom"],
                # Customer 2: only buys before cutoff -> churn 1
                ["2001", "A", "x", 3, "2010-01-20", 5.0, 2.0, "France"],
                # Rows without a customer or with zero price must be dropped
                ["3001", "A", "x", 1, "2010-01-05", 5.0, None, "France"],
                ["3002", "A", "x", 1, "2010-01-05", 0.0, 3.0, "France"],
                # Customer 4: first purchase after cutoff -> must not be in the dataset
                ["4001", "A", "x", 1, "2010-03-20", 5.0, 4.0, "Germany"],
                # Makes the data cover the full label window
                ["9999", "A", "x", 1, "2010-06-01", 5.0, 9.0, "Germany"],
            ]
        )
    )


def test_clean(tx):
    assert set(tx["customer_id"]) == {1, 2, 4, 9}
    assert tx["is_return"].sum() == 1


def test_build_dataset(tx):
    ds = build_dataset(tx, "2010-03-01", horizon_days=90)
    assert list(ds.columns) == FEATURES + [TARGET]
    assert set(ds.index) == {1, 2}
    assert ds.loc[1, TARGET] == 0
    assert ds.loc[2, TARGET] == 1
    assert ds.loc[1, "frequency"] == 2
    assert ds.loc[1, "monetary"] == pytest.approx(20.0)
    assert ds.loc[1, "recency_days"] == 19
    assert ds.loc[1, "return_rate"] == pytest.approx(1 / 3)
    assert ds.loc[2, "country"] == "France"


def test_label_window_must_fit(tx):
    with pytest.raises(ValueError):
        build_dataset(tx, "2010-05-15", horizon_days=90)


def test_lookback_window_limits_history(tx):
    full = build_dataset(tx, "2010-03-01", horizon_days=90)
    windowed = build_dataset(tx, "2010-03-01", horizon_days=90, lookback_days=35)
    # The window starts on 2010-01-25; customer 2 only bought before that
    assert set(windowed.index) == {1}
    # Customer 1: the 2010-01-10 order falls outside the window, the 2010-02-10 order inside
    assert windowed.loc[1, "frequency"] == 1
    assert windowed.loc[1, "tenure_days"] == 19
    assert full.loc[1, "tenure_days"] == 50


def test_lookback_longer_than_data_is_rejected(tx):
    with pytest.raises(ValueError):
        build_dataset(tx, "2010-03-01", horizon_days=90, lookback_days=365)


def test_read_lookback_days(tmp_path):
    assert read_lookback_days(tmp_path) is None
    (tmp_path / "serving_meta.json").write_text(json.dumps({"version": "8", "lookback_days": 270}))
    assert read_lookback_days(tmp_path) == 270
