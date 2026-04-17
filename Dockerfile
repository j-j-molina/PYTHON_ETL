# Dockerfile — cdp_credito-scoring
# Proyecto: cdp_credito  |  API: v1
# Build desde mlops_pipeline/: docker build -t cdp_credito-scoring .
FROM python:3.11-slim

LABEL project="cdp_credito"
LABEL description="Scoring del evento — cdp_credito-scoring"
LABEL maintainer="equipo-datos@empresa.com"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ENV=production \
    PORT=5000

WORKDIR /app

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt && \
    mkdir -p mlops_pipeline/src artifacts/scoring_mora reports/scoring_mora

COPY src/ft_engineering.py  mlops_pipeline/src/
COPY src/model_deploy.py    mlops_pipeline/src/
COPY src/config.json        mlops_pipeline/src/

COPY artifacts/scoring_mora/best_model.joblib       ./artifacts/scoring_mora/best_model.joblib
COPY artifacts/scoring_mora/best_model_meta.json  ./artifacts/scoring_mora/best_model_meta.json
COPY artifacts/scoring_mora/pipeline_ml.pkl  ./artifacts/scoring_mora/pipeline_ml.pkl
COPY artifacts/scoring_mora/pipeline_base.pkl ./artifacts/scoring_mora/pipeline_base.pkl

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["python", "mlops_pipeline/src/model_deploy.py", "--serve", "--host", "0.0.0.0", "--port", "5000", "--use-case", "scoring_mora"]
