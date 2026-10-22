# Keep the Python version the same as the one used for training
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/model

WORKDIR /app

# Install the exact library versions MLflow recorded when the model was trained.
# The full mlflow package (UI, server) is not needed to load a model, so use mlflow-skinny,
# and drop test-only packages.
COPY model/requirements.txt model/requirements.txt
COPY requirements-serve.txt .
RUN sed -e 's/^mlflow==/mlflow-skinny==/' -e '/^pytest/d' model/requirements.txt > /tmp/model-requirements.txt \
    && pip install -r /tmp/model-requirements.txt -r requirements-serve.txt

COPY src/ src/
COPY model/ model/

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
