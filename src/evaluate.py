"""Compare the served model with simple rule-based baselines on the hold-out test set.

ROC-AUC alone does not say whether a model is worth deploying. This script answers two questions:

1. How much better is the model than simple rules a marketing team could apply without ML?
   - Recency rule: rank customers by days since their last purchase.
   - "No order in the last 90 days": a yes/no rule.
2. What does it mean for a campaign? If marketing targets the riskiest 10% / 20% of customers,
   how many of them churn (precision@k), what share of all churners is reached (recall@k), and
   how much better is that than targeting at random (lift@k)?

Uncertainty: 95% confidence intervals from a paired bootstrap over test customers, including the
ROC-AUC gain of the model over the recency rule.

Calibration: does a predicted churn probability of 70% mean that ~70% of those customers churn?
Reported as the Brier score, the mean predicted vs. actual churn rate, and a reliability chart.

Usage:
    python -m src.evaluate
Writes reports/evaluation.md (tables for the README), reports/gains_chart.png and
reports/calibration_chart.png.
"""
import argparse
import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data import PROCESSED_PATH
from src.features import FEATURES, TARGET, build_dataset
from src.train import HORIZON_DAYS, TEST_CUTOFF

MODEL_PATH = Path("model")
REPORT_DIR = Path("reports")
TOP_K = (0.1, 0.2)
METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "precision_at_10": "Precision@10%",
    "precision_at_20": "Precision@20%",
    "recall_at_20": "Recall@20%",
    "lift_at_20": "Lift@20%",
}


def rank_order(score: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices from highest to lowest score; ties are broken at random.

    Without random tie-breaking, a yes/no rule would be ranked by the input order of the data.
    """
    tiebreak = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tiebreak, -score))


def top_k_metrics(y: np.ndarray, score: np.ndarray, k: float, seed: int = 0) -> dict[str, float]:
    n_top = max(1, round(k * len(y)))
    top = y[rank_order(score, seed)[:n_top]]
    precision = top.mean()
    return {
        "precision": precision,
        "recall": top.sum() / max(y.sum(), 1),
        "lift": precision / y.mean(),
    }


def compute_metrics(y: np.ndarray, score: np.ndarray, seed: int = 0) -> dict[str, float]:
    top10 = top_k_metrics(y, score, TOP_K[0], seed)
    top20 = top_k_metrics(y, score, TOP_K[1], seed)
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "precision_at_10": top10["precision"],
        "precision_at_20": top20["precision"],
        "recall_at_20": top20["recall"],
        "lift_at_20": top20["lift"],
    }


def bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap: every model is scored on the same resampled customers.

    Returns {model: {metric: (low, high)}} plus the 95% interval of the ROC-AUC gain of each
    model over the first entry in `scores` (the reference baseline).
    """
    rng = np.random.default_rng(seed)
    reference = next(iter(scores))
    samples = {name: {m: [] for m in METRIC_LABELS} for name in scores}
    gains = {name: [] for name in scores if name != reference}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        auc = {}
        for name, score in scores.items():
            m = compute_metrics(yb, score[idx])
            for key, value in m.items():
                samples[name][key].append(value)
            auc[name] = m["roc_auc"]
        for name, values in gains.items():
            values.append(auc[name] - auc[reference])

    def ci(values: list[float]) -> tuple[float, float]:
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    return {
        "intervals": {name: {m: ci(v) for m, v in metrics.items()} for name, metrics in samples.items()},
        "auc_gain_vs_reference": {name: ci(v) for name, v in gains.items()},
        "reference": reference,
    }


def lift_table(y: np.ndarray, score: np.ndarray, n_bins: int = 10, seed: int = 0) -> pd.DataFrame:
    """Customers split into equal-size groups by risk (decile 1 = riskiest 10%)."""
    order = rank_order(score, seed)
    groups = np.array_split(y[order], n_bins)
    rows, caught = [], 0
    for i, g in enumerate(groups, start=1):
        caught += g.sum()
        rows.append(
            {
                "decile": i,
                "customers": len(g),
                "churners": int(g.sum()),
                "churn_rate": g.mean(),
                "lift": g.mean() / y.mean(),
                "cumulative_recall": caught / y.sum(),
            }
        )
    return pd.DataFrame(rows)


def gains_curve(y: np.ndarray, score: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Share of customers targeted (x) vs. share of all churners reached (y)."""
    caught = np.cumsum(y[rank_order(score, seed)])
    x = np.arange(1, len(y) + 1) / len(y)
    return np.r_[0, x], np.r_[0, caught / y.sum()]


def plot_gains(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, score in scores.items():
        ax.plot(*gains_curve(y, score), label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random targeting")
    for k in TOP_K:
        ax.axvline(k, color="grey", lw=0.5, ls=":")
    ax.set(
        xlabel="Share of customers targeted (riskiest first)",
        ylabel="Share of all churners reached",
        title=f"Cumulative gains on the hold-out set ({TEST_CUTOFF})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibration_summary(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    brier = brier_score_loss(y, proba)
    # Reference: always predicting the observed churn rate (knows the base rate, but cannot rank)
    brier_base_rate = y.mean() * (1 - y.mean())
    return {
        "brier": brier,
        "brier_skill": 1 - brier / brier_base_rate,
        "mean_predicted": proba.mean(),
        "actual_rate": y.mean(),
    }


def calibration_table(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Customers split into equal-size groups by predicted probability (low to high)."""
    order = np.argsort(proba, kind="stable")
    rows = [
        {"customers": len(idx), "mean_predicted": proba[idx].mean(), "actual_rate": y[idx].mean()}
        for idx in np.array_split(order, n_bins)
    ]
    return pd.DataFrame(rows)


def plot_calibration(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(table["mean_predicted"], table["actual_rate"], "o-", label="Served model")
    ax.set(
        xlabel="Mean predicted churn probability (per decile)",
        ylabel="Actual churn rate",
        title=f"Calibration on the hold-out set ({TEST_CUTOFF})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_markdown(
    point: dict[str, dict[str, float]],
    boot: dict,
    lifts: pd.DataFrame,
    model_name: str,
    n: int,
    churn_rate: float,
    calibration: dict[str, float],
    calibration_bins: pd.DataFrame,
) -> str:
    header = "| Model | " + " | ".join(METRIC_LABELS.values()) + " |"
    lines = [
        (
            f"Hold-out test set: {n:,} customers as of {TEST_CUTOFF}, churn rate {churn_rate:.1%} "
            "(= precision of random targeting). Brackets: 95% bootstrap confidence interval."
        ),
        "",
        header,
        "|---" * (len(METRIC_LABELS) + 1) + "|",
    ]
    for name, metrics in point.items():
        cells = []
        for key, value in metrics.items():
            low, high = boot["intervals"][name][key]
            cells.append(f"{value:.3f} [{low:.3f}–{high:.3f}]" if key == "roc_auc" else f"{value:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", f"ROC-AUC gain over the {boot['reference'].lower()}:", ""]
    for name, (low, high) in boot["auc_gain_vs_reference"].items():
        gain = point[name]["roc_auc"] - point[boot["reference"]]["roc_auc"]
        lines.append(f"- {name}: {gain:+.3f} [{low:+.3f} to {high:+.3f}]")

    lines += [
        "",
        f"Lift table, {model_name.lower()}. Decile 1 = riskiest 10% of customers.",
        "",
        "| Decile | Customers | Churners | Churn rate | Lift | Cumulative recall |",
        "|---|---|---|---|---|---|",
    ]
    for r in lifts.itertuples():
        lines.append(
            f"| {r.decile} | {r.customers} | {r.churners} | {r.churn_rate:.1%} | {r.lift:.2f} | {r.cumulative_recall:.1%} |"
        )
    c = calibration
    lines += [
        "",
        f"Calibration, {model_name.lower()}:",
        "",
        (
            f"- Brier score: {c['brier']:.3f} (always predicting the churn rate: "
            f"{c['actual_rate'] * (1 - c['actual_rate']):.3f}, skill score {c['brier_skill']:.2f})"
        ),
        f"- Mean predicted churn probability: {c['mean_predicted']:.1%}, actual churn rate: {c['actual_rate']:.1%}",
        "",
        "| Predicted (decile mean) | " + " | ".join(f"{v:.0%}" for v in calibration_bins["mean_predicted"]) + " |",
        "|---" * (len(calibration_bins) + 1) + "|",
        "| Actual churn rate | " + " | ".join(f"{v:.0%}" for v in calibration_bins["actual_rate"]) + " |",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap resamples")
    args = parser.parse_args()

    tx = pd.read_parquet(PROCESSED_PATH)
    test = build_dataset(tx, TEST_CUTOFF, HORIZON_DAYS)
    y = test[TARGET].to_numpy()

    meta_path = args.model_path / "serving_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model_name = f"Served model ({meta.get('model_type', 'model')} v{meta.get('version', '?')})"
    model = mlflow.sklearn.load_model(str(args.model_path))

    # The first entry is the reference baseline for the ROC-AUC gain
    scores = {
        "Recency rule": test["recency_days"].to_numpy(),
        "No order in last 90 days": (test["orders_last_90d"] == 0).to_numpy(dtype=float),
        model_name: model.predict_proba(test[FEATURES])[:, 1],
    }
    point = {name: compute_metrics(y, s) for name, s in scores.items()}
    boot = bootstrap(y, scores, n_boot=args.n_boot)
    lifts = lift_table(y, scores[model_name])
    calibration = calibration_summary(y, scores[model_name])
    calibration_bins = calibration_table(y, scores[model_name])

    REPORT_DIR.mkdir(exist_ok=True)
    md_path = REPORT_DIR / "evaluation.md"
    md_path.write_text(
        render_markdown(point, boot, lifts, model_name, len(y), y.mean(), calibration, calibration_bins),
        encoding="utf-8",
    )
    plot_gains(y, scores, REPORT_DIR / "gains_chart.png")
    plot_calibration(calibration_bins, REPORT_DIR / "calibration_chart.png")

    print(md_path.read_text(encoding="utf-8"))
    print(f"Written to {md_path}, {REPORT_DIR / 'gains_chart.png'} and {REPORT_DIR / 'calibration_chart.png'}")


if __name__ == "__main__":
    main()
