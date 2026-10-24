"""Promote a registered model version to "champion" and export it for serving.

The exported folder (model/) is what the API and the Docker image load, so the
serving side does not need access to the MLflow tracking database.

Usage:
    python -m src.export_model --version 2   # set version 2 as champion, then export
    python -m src.export_model               # export the current champion
    python -m src.export_model --list        # show registered versions and their model type
"""
import argparse
import json
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.train import MODEL_NAME

ALIAS = "champion"
EXPORT_DIR = Path("model")


def list_versions(client: MlflowClient) -> None:
    versions = sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    except mlflow.exceptions.MlflowException:
        champion_version = None
    for v in versions:
        run = client.get_run(v.run_id)
        auc = run.data.metrics.get("test_roc_auc")
        champion = " (champion)" if str(v.version) == str(champion_version) else ""
        auc_text = f"{auc:.3f}" if auc is not None else "n/a"
        lookback = run.data.params.get("lookback_days", "full")
        print(
            f"v{v.version}  {run.data.params.get('model_type', '?'):7} lookback={lookback:5} "
            f"test_roc_auc={auc_text}{champion}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, help="Registered model version to promote to champion")
    parser.add_argument("--list", action="store_true", help="List registered versions and exit")
    args = parser.parse_args()

    client = MlflowClient()
    if args.list:
        list_versions(client)
        return
    if args.version is not None:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, str(args.version))
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}@{ALIAS}", dst_path=str(EXPORT_DIR))

    # Training data snapshot, used later as the reference for drift monitoring
    mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path="reference.parquet", dst_path=str(EXPORT_DIR))

    run = client.get_run(version.run_id)
    lookback = run.data.params.get("lookback_days", "full")
    meta = {
        "name": MODEL_NAME,
        "version": str(version.version),
        "run_id": version.run_id,
        "model_type": run.data.params.get("model_type"),
        # Feature window the model expects; the API, monitor and evaluation must use the same one
        "lookback_days": None if lookback == "full" else int(lookback),
        # Hold-out test scores from training, used as the baseline for performance monitoring
        "test_metrics": {k.removeprefix("test_"): v for k, v in run.data.metrics.items() if k.startswith("test_")},
    }
    (EXPORT_DIR / "serving_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"Exported {MODEL_NAME} v{version.version} ({meta['model_type']}, run {version.run_id}) to {EXPORT_DIR}/"
    )


if __name__ == "__main__":
    main()
