import pandas as pd
import pytest

from src.data import clean
from src.features import FEATURES, TARGET, build_dataset


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
