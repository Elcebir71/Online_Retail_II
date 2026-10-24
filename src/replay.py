"""Send real customers from the dataset to a running API, to simulate production traffic.

The API logs every scored customer, so after a replay the drift monitor can be run on the
prediction log, exactly as it would be on real traffic:

    docker run --rm -p 8000:8000 -v ${PWD}/logs:/app/logs churn-api      # or: uvicorn src.api:app
    python -m src.replay --cutoff 2011-09-09
    python -m src.monitor --prediction-log logs/predictions.jsonl

Usage:
    python -m src.replay --cutoff 2011-09-09 [--url http://127.0.0.1:8000] [--sample 2000]
"""
import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

from src.data import PROCESSED_PATH
from src.features import FEATURES, build_features, read_lookback_days

BATCH_SIZE = 500  # the API accepts at most 1000 customers per request


def post_batch(url: str, customers: list[dict]) -> dict:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/predict",
        data=json.dumps({"customers": customers}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", required=True, help="Score customers as of this date (YYYY-MM-DD)")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--sample", type=int, help="Random sample of customers instead of all of them")
    parser.add_argument("--model-path", type=Path, default=Path("model"), help="Served model, for its feature window")
    args = parser.parse_args()

    tx = pd.read_parquet(PROCESSED_PATH)
    lookback = read_lookback_days(args.model_path)
    customers = build_features(tx, pd.Timestamp(args.cutoff), lookback)[FEATURES]
    if args.sample:
        customers = customers.sample(min(args.sample, len(customers)), random_state=42)
    records = customers.to_dict("records")

    n_high_risk = 0
    for start in range(0, len(records), BATCH_SIZE):
        preds = post_batch(args.url, records[start : start + BATCH_SIZE])["predictions"]
        n_high_risk += sum(p["high_risk"] for p in preds)
    print(f"Scored {len(records):,} customers as of {args.cutoff}: {n_high_risk:,} flagged high risk")


if __name__ == "__main__":
    main()
