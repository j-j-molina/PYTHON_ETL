# Dockerfile — cdp_credito-scoring
# Proyecto: cdp_credito  |  API: v1
# Generado por model_deploy.py — CDP Entregable 3
FROM python:3.11-slim

LABEL project="cdp_credito"
LABEL description="Scoring de mora — cdp_credito-scoring"
LABEL maintainer="equipo-datos@empresa.com"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ENV=production \
    PORT=5000

WORKDIR /app

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY mlops_pipeline/src/ft_engineering.py  ./src/
COPY mlops_pipeline/src/model_deploy.py    ./src/
COPY mlops_pipeline/src/config.json        ./src/

COPY mlops_pipeline/artifacts/best_model.joblib       ./mlops_pipeline/artifacts/best_model.joblib
COPY mlops_pipeline/artifacts/best_model_meta.json  ./mlops_pipeline/artifacts/best_model_meta.json
COPY mlops_pipeline/artifacts/pipeline_ml.pkl  ./mlops_pipeline/artifacts/pipeline_ml.pkl
COPY mlops_pipeline/artifacts/pipeline_base.pkl ./mlops_pipeline/artifacts/pipeline_base.pkl

RUN mkdir -p /app/mlops_pipeline/artifacts /app/mlops_pipeline/reports

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["python", "src/model_deploy.py", "--serve", "--host", "0.0.0.0", "--port", "5000"]
