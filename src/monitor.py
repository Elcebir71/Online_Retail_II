"""Monitor data drift and model performance against the training reference.

For each monitoring cutoff, the script rebuilds the customer features as of that date and
compares them with the training snapshot stored next to the served model (reference.parquet):

- Data drift: Population Stability Index (PSI) per feature and for the predicted probability.
- Performance: ROC-AUC / PR-AUC once the 90-day churn labels for the period are known.

If drift or the performance drop crosses a threshold, the report recommends retraining.

Usage:
    python -m src.monitor --cutoff 2010-12-01 2011-03-01 2011-06-01 2011-09-09
    python -m src.monitor --cutoff 2011-09-09 --fail-on-alert   # exit code 1 on alert (for CI/cron)
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
    model, reference: pd.DataFrame, tx: pd.DataFrame, cutoff: str, baseline: dict[str, float] | None = None
) -> tuple[MonitoringResult, pd.DataFrame]:
    reference = reference.assign(churn_probability=model.predict_proba(reference[FEATURES])[:, 1])
    # Scores on the training data itself are optimistic, so prefer the hold-out test baseline
    ref_perf = baseline or scores(reference)

    labels_known = tx["invoice_date"].max() >= pd.Timestamp(cutoff) + np.timedelta64(HORIZON_DAYS - 1, "D")
    if labels_known:
        current = build_dataset(tx, cutoff, HORIZON_DAYS)
    else:
        # Labels for this period are not observable yet: monitor drift only
        current = build_features(tx, pd.Timestamp(cutoff))
    current = current.assign(churn_probability=model.predict_proba(current[FEATURES])[:, 1])

    table = drift_table(reference, current)
    drifted = table.loc[(table["status"] == "drift") & (table["feature"] != "churn_probability"), "feature"].tolist()
    perf = scores(current) if labels_known else None

    alerts = []
    if drifted:
        alerts.append(f"Significant drift (PSI >= {PSI_DRIFT}) in: {', '.join(drifted)}")
    pred_psi = table.set_index("feature").loc["churn_probability", "psi"]
    if pred_psi >= PSI_DRIFT:
        alerts.append(f"Prediction distribution shifted (PSI = {pred_psi:.3f})")
    if perf and ref_perf["roc_auc"] - perf["roc_auc"] > MAX_AUC_DROP:
        alerts.append(f"ROC-AUC dropped from {ref_perf['roc_auc']:.3f} to {perf['roc_auc']:.3f}")

    result = MonitoringResult(
        cutoff=cutoff,
        n_customers=len(current),
        churn_rate=round(float(current[TARGET].mean()), 4) if labels_known else None,
        performance=perf,
        reference_performance=ref_perf,
        drifted_features=drifted,
        alerts=alerts,
    )
    return result, table


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
<p>Cutoff <b>{result.cutoff}</b> &middot; {result.n_customers:,} customers
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
    parser.add_argument("--cutoff", nargs="+", required=True, help="One or more monitoring dates (YYYY-MM-DD)")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--fail-on-alert", action="store_true", help="Exit with code 1 if any period raises an alert")
    args = parser.parse_args()

    model = mlflow.sklearn.load_model(str(args.model_path))
    reference = pd.read_parquet(args.model_path / "reference.parquet")
    baseline = baseline_performance(args.model_path)
    if baseline is None:
        print("No hold-out baseline in serving_meta.json; comparing against in-sample training scores.")
    tx = pd.read_parquet(PROCESSED_PATH)
    REPORT_DIR.mkdir(exist_ok=True)
    mlflow.set_experiment(EXPERIMENT)

    any_alert = False
    for cutoff in args.cutoff:
        result, table = evaluate_period(model, reference, tx, cutoff, baseline)
        html_path = REPORT_DIR / f"monitoring_{cutoff}.html"
        json_path = REPORT_DIR / f"monitoring_{cutoff}.json"
        html_path.write_text(render_html(result, table), encoding="utf-8")
        json_path.write_text(
            json.dumps({**asdict(result), "retrain": result.retrain, "psi": table.to_dict("records")}, indent=2),
            encoding="utf-8",
        )

        with mlflow.start_run(run_name=f"monitor-{cutoff}"):
            mlflow.log_param("cutoff", cutoff)
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
