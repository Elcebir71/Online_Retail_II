# Online Retail II – Customer Churn MLOps Pipeline

[![CI/CD](https://github.com/Elcebir71/Online_Retail_II/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Elcebir71/Online_Retail_II/actions/workflows/ci-cd.yml)

End-to-end MLOps project that predicts which customers of an online retailer will churn,
so marketing can target them with win-back campaigns.

**Stack:** scikit-learn · MLflow (tracking + model registry) · FastAPI · Docker · GitHub Actions

## Problem

The [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset
contains ~1M invoice lines from a UK online retailer (Dec 2009 – Dec 2011). It has no churn
column, so the label is derived:

- Pick a **cutoff date**. Features are computed only from transactions *before* the cutoff.
- A customer is **churned (1)** if they make no purchase in the **90 days after** the cutoff.

Features (RFM and behaviour): recency, frequency, monetary value, average order value,
number of distinct products, tenure, return rate, orders in the last 90 days, country.

## Results

Time-based split: train on customers as of 2010-09-09, test on customers as of 2011-09-09
(same season one year later, no leakage from the future).

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **Logistic regression** (served) | **0.77** | **0.79** | **0.69** |
| HistGradientBoosting | 0.74 | 0.76 | 0.67 |

The simpler model wins: RFM features are few and close to monotonic, and its coefficients are
easy to explain to a marketing team.

## Project structure

```
src/
  data.py          # download + clean raw data -> data/processed/transactions.parquet
  features.py      # RFM features and churn label for a given cutoff
  train.py         # train, evaluate, log to MLflow, optionally register
  export_model.py  # promote a version to "champion" and export it to model/
  api.py           # FastAPI service (/health, /predict)
model/             # exported champion model served by the API and the Docker image
tests/             # unit tests for features and the API
Dockerfile
.github/workflows/ci-cd.yml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.data                                # download and clean the data
python -m src.train --model logreg --register     # train and register in MLflow
python -m src.export_model --version <N>          # promote version N to champion and export
mlflow ui                                         # compare runs at http://127.0.0.1:5000
```

### Serve the model

```bash
uvicorn src.api:app --reload      # http://127.0.0.1:8000/docs
```

or with Docker:

```bash
docker build -t churn-api .
docker run -it --rm -p 8000:8000 churn-api
```

or pull the image published by CI:

```bash
docker run -it --rm -p 8000:8000 ghcr.io/elcebir71/online-retail-churn:latest
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customers": [{"recency_days": 45, "frequency": 6, "monetary": 1250.5, "avg_order_value": 208.4,
       "n_products": 38, "tenure_days": 300, "return_rate": 0.02, "orders_last_90d": 1,
       "country": "United Kingdom"}]}'
```

```json
{"model_version": "2", "predictions": [{"churn_probability": 0.1373, "churn": false}]}
```

## CI/CD

On every push and pull request, GitHub Actions:

1. Lints the code with **ruff** and runs the **pytest** suite.
2. Builds the Docker image and **smoke tests** it (`/health` and `/predict`).
3. On `main` only: publishes the image to **GitHub Container Registry**
   (`latest` and commit-SHA tags).

## Roadmap

- [x] Data pipeline, churn label and model tracking with MLflow
- [x] FastAPI service and Docker image
- [x] CI/CD with GitHub Actions
- [ ] Data drift and performance monitoring
