"""Monitor data drift and model performance against the training reference.

Two sources for the "current" data:

- Historical simulation (--cutoff): rebuild the customer features from the dataset as of past
  dates. Churn labels are known, so performance can be tracked as well.
- Live traffic (--prediction-log): the customers the API actually scored, read from its
  prediction log. Labels are not known yet, so only drift is measured.

Both are compared with the training snapshot stored next to the served model (reference.parquet):

- Data drift: Population Stability Index (PSI) per feature and for the predicted probability.
- Performance: ROC-AUC / PR-AUC once the 90-day churn labels for the period are known.

If drift or the performance drop crosses a threshold, the report recommends retraining.

Usage:
    python -m src.monitor --cutoff 2010-12-01 2011-03-01 2011-06-01 2011-09-09
    python -m src.monitor --cutoff 2011-09-09 --fail-on-alert   # exit code 1 on alert (for CI/cron)
    python -m src.monitor --prediction-log logs/predictions.jsonl
Reports are written to reports/ and logged to MLflow.
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import PROCESSED_PATH
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_dataset,
    build_features,
    read_lookback_days,
)

MODEL_PATH = Path("model")
REPORT_DIR = Path("reports")
EXPERIMENT = "online-retail-churn-monitoring"
HORIZON_DAYS = 90

# Common PSI rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift
PSI_WARNING = 0.1
PSI_DRIFT = 0.25
# Alert when ROC-AUC drops more than this below the reference (training period) score
MAX_AUC_DROP = 0.05
EPS = 1e-4
# Below this many scored customers, PSI on ten bins is too noisy to act on
MIN_LIVE_ROWS = 200


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI with bins taken from the reference quantiles, so each bin holds ~10% of the reference."""
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = pd.cut(reference, edges).value_counts(normalize=True, sort=False).to_numpy()
    cur_pct = pd.cut(current, edges).value_counts(normalize=True, sort=False).to_numpy()
    return _psi(ref_pct, cur_pct)


def psi_categorical(reference: pd.Series, current: pd.Series, top_n: int = 10) -> float:
    """PSI over the most frequent reference categories; everything else goes to "other"."""
    top = reference.value_counts().index[:top_n]

    def shares(s: pd.Series) -> pd.Series:
        return s.where(s.isin(top), "other").value_counts(normalize=True)

    ref, cur = shares(reference).align(shares(current), fill_value=0)
    return _psi(ref.to_numpy(), cur.to_numpy())


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_status(value: float) -> str:
    if value >= PSI_DRIFT:
        return "drift"
    if value >= PSI_WARNING:
        return "warning"
    return "ok"


def drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["churn_probability"]:
        psi_fn = psi_categorical if col in CATEGORICAL_FEATURES else psi_numeric
        value = psi_fn(reference[col], current[col])
        rows.append({"feature": col, "psi": round(value, 4), "status": psi_status(value)})
    return pd.DataFrame(rows)


def scores(df: pd.DataFrame) -> dict[str, float]:
    return {
        "roc_auc": round(roc_auc_score(df[TARGET], df["churn_probability"]), 4),
        "pr_auc": round(average_precision_score(df[TARGET], df["churn_probability"]), 4),
    }


@dataclass
class MonitoringResult:
    cutoff: str
    n_customers: int
    churn_rate: float | None
    performance: dict[str, float] | None
    reference_performance: dict[str, float]  # baseline the current scores are compared with
    drifted_features: list[str]
    alerts: list[str] = field(default_factory=list)

    @property
    def retrain(self) -> bool:
        return bool(self.alerts)


def baseline_performance(model_path: Path) -> dict[str, float] | None:
    """Hold-out test scores recorded at training time (written by src.export_model)."""
    meta_path = model_path / "serving_meta.json"
    if not meta_path.exists():
        return None
    metrics = json.loads(meta_path.read_text()).get("test_metrics") or {}
    if {"roc_auc", "pr_auc"} <= metrics.keys():
        return {"roc_auc": round(metrics["roc_auc"], 4), "pr_auc": round(metrics["pr_auc"], 4)}
    return None


def evaluate_period(
    model,
    reference: pd.DataFrame,
    tx: pd.DataFrame,
    cutoff: str,
    baseline: dict[str, float] | None = None,
    lookback_days: int | None = None,
) -> tuple[MonitoringResult, pd.DataFrame]:
    reference = reference.assign(churn_probability=model.predict_proba(reference[FEATURES])[:, 1])
    # Scores on the training data itself are optimistic, so prefer the hold-out test baseline
    ref_perf = baseline or scores(reference)

    labels_known = tx["invoice_date"].max() >= pd.Timestamp(cutoff) + np.timedelta64(HORIZON_DAYS - 1, "D")
    if labels_known:
        current = build_dataset(tx, cutoff, HORIZON_DAYS, lookback_days)
    else:
        # Labels for this period are not observable yet: monitor drift only
        current = build_features(tx, pd.Timestamp(cutoff), lookback_days)
    current = current.assign(churn_probability=model.predict_proba(current[FEATURES])[:, 1])

    perf = scores(current) if labels_known else None
    churn_rate = round(float(current[TARGET].mean()), 4) if labels_known else None
    return _compare(reference, current, cutoff, perf, ref_perf, churn_rate)


def _compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    period: str,
    perf: dict[str, float] | None,
    ref_perf: dict[str, float],
    churn_rate: float | None,
) -> tuple[MonitoringResult, pd.DataFrame]:
    table = drift_table(reference, current)
    drifted = table.loc[(table["status"] == "drift") & (table["feature"] != "churn_probability"), "feature"].tolist()

    alerts = []
    if drifted:
        alerts.append(f"Significant drift (PSI >= {PSI_DRIFT}) in: {', '.join(drifted)}")
    pred_psi = table.set_index("feature").loc["churn_probability", "psi"]
    if pred_psi >= PSI_DRIFT:
        alerts.append(f"Prediction distribution shifted (PSI = {pred_psi:.3f})")
    if perf and ref_perf["roc_auc"] - perf["roc_auc"] > MAX_AUC_DROP:
        alerts.append(f"ROC-AUC dropped from {ref_perf['roc_auc']:.3f} to {perf['roc_auc']:.3f}")

    result = MonitoringResult(
        cutoff=period,
        n_customers=len(current),
        churn_rate=churn_rate,
        performance=perf,
        reference_performance=ref_perf,
        drifted_features=drifted,
        alerts=alerts,
    )
    return result, table


def read_prediction_log(path: Path, since: str | None = None) -> pd.DataFrame:
    """Customers scored by the API (one JSON line each), optionally only those after `since`."""
    log = pd.read_json(path, lines=True, dtype=False)
    log["timestamp"] = pd.to_datetime(log["timestamp"], utc=True)
    if since:
        log = log[log["timestamp"] >= pd.Timestamp(since, tz="UTC")]
    return log


def evaluate_live(
    model, reference: pd.DataFrame, log: pd.DataFrame, baseline: dict[str, float] | None = None
) -> tuple[MonitoringResult, pd.DataFrame]:
    """Drift between the training snapshot and the traffic the API actually served."""
    reference = reference.assign(churn_probability=model.predict_proba(reference[FEATURES])[:, 1])
    ref_perf = baseline or scores(reference)
    # Use the scores the API returned, i.e. exactly what consumers of the service saw
    current = log[FEATURES].assign(churn_probability=log["churn_score"])
    start, end = log["timestamp"].min(), log["timestamp"].max()
    period = f"live {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC"
    return _compare(reference, current, period, None, ref_perf, None)


def render_html(result: MonitoringResult, table: pd.DataFrame) -> str:
    colors = {"ok": "#1a7f37", "warning": "#9a6700", "drift": "#cf222e"}
    rows = "".join(
        f"<tr><td>{r.feature}</td><td>{r.psi:.4f}</td>"
        f"<td style='color:{colors[r.status]};font-weight:600'>{r.status}</td></tr>"
        for r in table.itertuples()
    )
    perf = result.performance or {}
    ref = result.reference_performance
    verdict = (
        "<p style='color:#cf222e;font-weight:600'>Retraining recommended</p><ul>"
        + "".join(f"<li>{a}</li>" for a in result.alerts)
        + "</ul>"
        if result.retrain
        else "<p style='color:#1a7f37;font-weight:600'>No action needed</p>"
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Churn monitoring {result.cutoff}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}</style>
</head><body>
<h1>Churn model monitoring</h1>
<p>Period <b>{result.cutoff}</b> &middot; {result.n_customers:,} customers
&middot; churn rate {result.churn_rate if result.churn_rate is not None else "not known yet"}</p>
{verdict}
<h2>Performance</h2>
<table><tr><th>Metric</th><th>Baseline</th><th>Current</th></tr>
<tr><td>ROC-AUC</td><td>{ref["roc_auc"]:.3f}</td><td>{perf.get("roc_auc", "n/a")}</td></tr>
<tr><td>PR-AUC</td><td>{ref["pr_auc"]:.3f}</td><td>{perf.get("pr_auc", "n/a")}</td></tr></table>
<h2>Data drift (PSI)</h2>
<p>&lt; {PSI_WARNING} ok &middot; {PSI_WARNING}-{PSI_DRIFT} warning &middot; &ge; {PSI_DRIFT} drift</p>
<table><tr><th>Feature</th><th>PSI</th><th>Status</th></tr>{rows}</table>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cutoff", nargs="+", help="Historical simulation: one or more dates (YYYY-MM-DD)")
    source.add_argument("--prediction-log", type=Path, help="Live traffic: the API's JSON lines prediction log")
    parser.add_argument("--since", help="With --prediction-log: only predictions from this date (YYYY-MM-DD)")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--fail-on-alert", action="store_true", help="Exit with code 1 if any period raises an alert")
    args = parser.parse_args()

    model = mlflow.sklearn.load_model(str(args.model_path))
    reference = pd.read_parquet(args.model_path / "reference.parquet")
    baseline = baseline_performance(args.model_path)
    lookback = read_lookback_days(args.model_path)  # same feature window as in training
    if baseline is None:
        print("No hold-out baseline in serving_meta.json; comparing against in-sample training scores.")
    REPORT_DIR.mkdir(exist_ok=True)
    mlflow.set_experiment(EXPERIMENT)

    if args.prediction_log:
        log = read_prediction_log(args.prediction_log, args.since)
        if log.empty:
            sys.exit(f"No predictions in {args.prediction_log}" + (f" since {args.since}" if args.since else ""))
        if len(log) < MIN_LIVE_ROWS:
            print(f"Only {len(log)} scored customers; PSI is unreliable below {MIN_LIVE_ROWS}.")
        periods = [("live", lambda: evaluate_live(model, reference, log, baseline))]
    else:
        tx = pd.read_parquet(PROCESSED_PATH)
        periods = [(c, lambda c=c: evaluate_period(model, reference, tx, c, baseline, lookback)) for c in args.cutoff]

    any_alert = False
    for name, evaluate in periods:
        result, table = evaluate()
        cutoff = result.cutoff
        html_path = REPORT_DIR / f"monitoring_{name}.html"
        json_path = REPORT_DIR / f"monitoring_{name}.json"
        html_path.write_text(render_html(result, table), encoding="utf-8")
        json_path.write_text(
            json.dumps({**asdict(result), "retrain": result.retrain, "psi": table.to_dict("records")}, indent=2),
            encoding="utf-8",
        )

        with mlflow.start_run(run_name=f"monitor-{name}"):
            mlflow.log_param("cutoff", cutoff)
            mlflow.log_metric("n_customers", result.n_customers)
            mlflow.log_metrics({f"psi_{r.feature}": r.psi for r in table.itertuples()})
            mlflow.log_metric("n_drifted_features", len(result.drifted_features))
            mlflow.log_metric("retrain", int(result.retrain))
            if result.performance:
                mlflow.log_metrics({f"current_{k}": v for k, v in result.performance.items()})
            mlflow.log_artifact(str(html_path))

        status = "RETRAIN" if result.retrain else "OK"
        auc = result.performance["roc_auc"] if result.performance else "n/a"
        print(f"{cutoff}: {status:7} customers={result.n_customers:,} roc_auc={auc} drifted={result.drifted_features}")
        for alert in result.alerts:
            print(f"    - {alert}")
        any_alert |= result.retrain

    print(f"Reports written to {REPORT_DIR}/")
    if args.fail_on_alert and any_alert:
        sys.exit(1)


if __name__ == "__main__":
    main()
