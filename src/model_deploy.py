"""
model_deploy.py
===============
Toda la configuración de despliegue (rutas, puerto, imagen Docker, umbrales
de riesgo, límite de batch) se lee desde config.json.

Secciones relevantes de config.json:
  paths.model_file          → modelo serializado
  paths.model_meta_file     → metadatos del modelo
  paths.pipeline_ml_file    → ColumnTransformer serializado
  paths.logs_file           → log de predicciones para monitoring
  paths.deploy_summary_file → resumen para DevOps
  target.event_col          → columna target
  feature_engineering.leakage_cols → columnas a excluir del input
  split.date_col            → columna de fecha
  deploy.host / .port       → API Flask
  deploy.docker_image       → nombre de la imagen Docker
  deploy.docker_port        → puerto del contenedor
  deploy.api_version        → versión de la API
  deploy.risk_thresholds    → fracciones del threshold para Bajo/Medio/Alto
  deploy.batch_max_records  → límite de registros por batch

Uso:
  python model_deploy.py                   # genera Dockerfile + artefactos
  python model_deploy.py --serve           # levanta API Flask
  python model_deploy.py --batch data.csv  # predicción batch inmediata
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Carga del modelo y metadatos
# ──────────────────────────────────────────────────────────

def load_deployed_model(config_path: str | Path | None = None):
    """Carga modelo, metadatos y config desde cfg.paths."""
    import sys
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from ft_engineering import load_config

    cfg, repo_root = load_config(config_path)
    model_path = repo_root / cfg["paths"]["model_file"]
    meta_path  = repo_root / cfg["paths"]["model_meta_file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}. "
            "Ejecuta model_training.py primero."
        )

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    logger.info("Modelo cargado: %s  (threshold=%.4f)",
                meta["model_name"], meta["threshold"])
    return model, meta, cfg, repo_root


# ──────────────────────────────────────────────────────────
#  Clasificación de riesgo — configurable desde config.json
# ──────────────────────────────────────────────────────────

def _make_risk_classifier(threshold: float, risk_cfg: dict):
    """
    Retorna una función de clasificación de riesgo basada en
    cfg.deploy.risk_thresholds (fracciones del threshold calibrado).

      score < threshold * bajo   → "Bajo"
      score < threshold * medio  → "Medio"
      score >= threshold         → "Alto"
    """
    bajo_limit  = threshold * risk_cfg.get("bajo",  0.5)
    medio_limit = threshold * risk_cfg.get("medio", 1.0)

    def classify(score: float) -> str:
        if score < bajo_limit:
            return "Bajo"
        elif score < medio_limit:
            return "Medio"
        else:
            return "Alto"

    return classify


# ──────────────────────────────────────────────────────────
#  Pipeline de predicción batch
# ──────────────────────────────────────────────────────────

def predict_batch(
    input_data:      pd.DataFrame | str | Path,
    model=None,
    meta:            dict | None = None,
    cfg:             dict | None = None,
    repo_root:       Path | None = None,
    log_predictions: bool = True,
) -> pd.DataFrame:
    """
    Genera predicciones de mora para un batch de nuevos créditos.

    Lee toda la configuración desde cfg (config.json):
      - Umbral de decisión: meta["threshold"]
      - Columnas a excluir: cfg.feature_engineering.leakage_cols
      - Columna de fecha: cfg.split.date_col
      - Pipeline serializado: cfg.paths.pipeline_ml_file
      - Clasificación de riesgo: cfg.deploy.risk_thresholds
      - Límite de batch: cfg.deploy.batch_max_records

    Returns:
        DataFrame con: prediction_id, score_mora, pred_mora, riesgo,
                       fecha_prediccion. Si el input tenía fecha_prestamo,
                       se incluye como primera columna.
    """
    import sys
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from ft_engineering import apply_pipeline_steps

    if model is None or meta is None:
        model, meta, cfg, repo_root = load_deployed_model()

    # ── Leer parámetros desde config ──────────────────────
    threshold    = meta["threshold"]
    deploy_cfg   = cfg.get("deploy", {})
    max_records  = deploy_cfg.get("batch_max_records", 10_000)
    risk_cfg     = deploy_cfg.get("risk_thresholds", {"bajo": 0.5, "medio": 1.0})
    date_col     = cfg["split"]["date_col"]
    event_col    = cfg["target"]["event_col"]
    leakage_cols = cfg["feature_engineering"]["leakage_cols"]

    classify_risk = _make_risk_classifier(threshold, risk_cfg)

    # ── Cargar datos ──────────────────────────────────────
    if isinstance(input_data, (str, Path)):
        logger.info("Cargando CSV: %s", input_data)
        df_raw = pd.read_csv(input_data)
    else:
        df_raw = input_data.copy()

    n_records = len(df_raw)
    logger.info("Batch recibido: %d registros", n_records)

    if n_records > max_records:
        raise ValueError(
            f"El batch tiene {n_records} registros, "
            f"máximo permitido: {max_records} (cfg.deploy.batch_max_records)."
        )

    # ── Parsear fecha (cfg.split.date_col) ───────────────
    if date_col in df_raw.columns:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col],
                                          dayfirst=False, errors="coerce")
    fechas = df_raw[date_col].copy() if date_col in df_raw.columns else None

    # ── Eliminar columnas de target (cfg.feature_engineering.leakage_cols) ─
    cols_a_eliminar = [c for c in leakage_cols + [event_col] if c in df_raw.columns]
    if cols_a_eliminar:
        logger.warning("Columnas de target eliminadas del input: %s", cols_a_eliminar)
        df_raw = df_raw.drop(columns=cols_a_eliminar)

    # ── Cargar pipeline_base serializado (cfg.paths.pipeline_base_file) ────
    # Usamos el pipeline ajustado SOLO sobre train (guardado por model_training.py).
    # Esto garantiza que Winsorizar aplique los caps de train y que
    # ImputacionSegmentada use las medianas de train — sin data leakage.
    pipeline_base_path = repo_root / cfg["paths"]["pipeline_base_file"]
    if not pipeline_base_path.exists():
        raise FileNotFoundError(
            f"pipeline_base no encontrado: {pipeline_base_path}. "
            "Ejecuta model_training.py primero."
        )
    pipeline_base  = joblib.load(pipeline_base_path)
    logger.info("pipeline_base cargado: %s", pipeline_base_path)
    df_transformed = apply_pipeline_steps(pipeline_base, df_raw)

    # Limpiar residuos (mora, fecha) si quedaron tras el pipeline
    drop_residual = [c for c in [event_col, date_col] if c in df_transformed.columns]
    if drop_residual:
        df_transformed = df_transformed.drop(columns=drop_residual)

    # ── Cargar pipeline_ml serializado (cfg.paths.pipeline_ml_file) ──
    pipeline_ml_path = repo_root / cfg["paths"]["pipeline_ml_file"]
    if not pipeline_ml_path.exists():
        raise FileNotFoundError(
            f"pipeline_ml no encontrado: {pipeline_ml_path}. "
            "Ejecuta model_training.py primero."
        )

    pipeline_ml = joblib.load(pipeline_ml_path)
    logger.info("pipeline_ml cargado: %s", pipeline_ml_path)
    X = pipeline_ml.transform(df_transformed)
    feature_names = (
        list(pipeline_ml.get_feature_names_out())
        if hasattr(pipeline_ml, "get_feature_names_out")
        else [f"f_{i}" for i in range(X.shape[1])]
    )
    X_df = pd.DataFrame(X, columns=feature_names)

    # ── Predicción ────────────────────────────────────────
    t0     = time.time()
    scores = model.predict_proba(X_df)[:, 1]
    preds  = (scores >= threshold).astype(int)
    elapsed = time.time() - t0
    logger.info(
        "Predicciones: %d records en %.3fs | mora=%d (%.1f%%)",
        n_records, elapsed, preds.sum(), preds.mean() * 100,
    )

    # ── Construir resultado ───────────────────────────────
    result = pd.DataFrame({
        "prediction_id":    [str(uuid.uuid4()) for _ in range(n_records)],
        "score_mora":       np.round(scores, 6),
        "pred_mora":        preds,
        "riesgo":           [classify_risk(s) for s in scores],
        "fecha_prediccion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    if fechas is not None:
        result.insert(0, date_col, fechas.values)

    # ── Log (cfg.paths.logs_file) ─────────────────────────
    if log_predictions and repo_root is not None:
        _log_predictions(result, X_df, cfg, repo_root)

    return result


def _log_predictions(
    result_df: pd.DataFrame,
    X_df:      pd.DataFrame,
    cfg:       dict,
    repo_root: Path,
) -> None:
    """Persiste predicciones en cfg.paths.logs_file para model_monitoring.py."""
    logs_path = repo_root / cfg["paths"]["logs_file"]
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    log_df = pd.concat(
        [result_df.reset_index(drop=True), X_df.reset_index(drop=True)],
        axis=1,
    )
    if logs_path.exists():
        log_df.to_csv(logs_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(logs_path, index=False)

    logger.info("Log actualizado: %s  (+%d rows)", logs_path, len(log_df))


# ──────────────────────────────────────────────────────────
#  API Flask — host/puerto/versión desde cfg.deploy
# ──────────────────────────────────────────────────────────

def create_app(model=None, meta=None, cfg=None, repo_root=None):
    """
    Crea la aplicación Flask.
    Parámetros de red desde cfg.deploy (host, port, api_version).
    Límites de batch desde cfg.deploy.batch_max_records.
    Métricas de drift desde cfg.monitoring.psi_threshold.

    Endpoints:
      GET  /health   → estado del servicio
      GET  /metrics  → métricas del modelo + drift
      POST /predict  → predicción batch (JSON o CSV)
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise ImportError("Flask no está instalado. pip install flask")

    if model is None:
        model, meta, cfg, repo_root = load_deployed_model()

    deploy_cfg  = cfg.get("deploy", {})
    api_version = deploy_cfg.get("api_version", "v1")
    start_time  = datetime.now()

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status":      "ok",
            "project":     cfg["project_code"],
            "model":       meta["model_name"],
            "threshold":   meta["threshold"],
            "api_version": api_version,
            "uptime_s":    (datetime.now() - start_time).total_seconds(),
        })

    @app.route("/metrics", methods=["GET"])
    def metrics():
        reports_dir = repo_root / cfg["paths"]["reports_dir"]
        eval_json   = reports_dir / "evaluation_report.json"
        drift_json  = reports_dir / "monitoring_report.json"

        response = {
            "model_name": meta["model_name"],
            "threshold":  meta["threshold"],
            "gap_auc":    meta.get("gap_auc"),
        }
        if eval_json.exists():
            with open(eval_json) as f:
                ed = json.load(f)
            response["test_metrics"] = ed.get("test_metrics", {})
            response["cv_metrics"] = {
                k: v.get("mean") for k, v in ed.get("cv_metrics", {}).items()
            }
        if drift_json.exists():
            with open(drift_json) as f:
                dd = json.load(f)
            response["drift"] = {
                "psi_score":          dd.get("psi_score"),
                "n_features_drift":   dd.get("n_features_drift"),
                "n_features_monitor": dd.get("n_features_monitor"),
                # Umbral desde config — no hardcodeado
                "psi_threshold":      cfg["monitoring"]["psi_threshold"],
            }
        return jsonify(response)

    @app.route("/predict", methods=["POST"])
    def predict():
        """Predicción batch. Acepta application/json o text/csv."""
        try:
            ct = request.content_type or ""
            if "application/json" in ct:
                data = request.get_json(force=True)
                df_input = pd.DataFrame(
                    data["records"] if isinstance(data, dict) and "records" in data
                    else data
                )
            elif "text/csv" in ct:
                from io import StringIO
                df_input = pd.read_csv(StringIO(request.data.decode("utf-8")))
            else:
                return jsonify({"error": "Content-Type no soportado"}), 415

            if len(df_input) == 0:
                return jsonify({"error": "Payload vacío"}), 400

            result = predict_batch(
                df_input, model=model, meta=meta, cfg=cfg, repo_root=repo_root,
            )
            return jsonify({
                "n_records":   len(result),
                "n_mora_pred": int(result["pred_mora"].sum()),
                "predictions": result.to_dict(orient="records"),
            })

        except ValueError as e:
            return jsonify({"error": str(e)}), 422
        except Exception as e:
            logger.exception("Error en /predict")
            return jsonify({"error": str(e)}), 500

    return app


# ──────────────────────────────────────────────────────────
#  Generación de artefactos Docker — todo desde config.json
# ──────────────────────────────────────────────────────────

def generate_dockerfile(repo_root: Path, cfg: dict) -> None:
    """Dockerfile usando cfg.deploy.docker_image, .docker_port y cfg.paths."""
    d           = cfg.get("deploy", {})
    image       = d.get("docker_image", cfg["project_code"] + "-scoring")
    docker_port = d.get("docker_port", 5000)

    content = f"""# Dockerfile — {image}
# Proyecto: {cfg["project_code"]}  |  API: {d.get("api_version", "v1")}
# Generado por model_deploy.py — CDP Entregable 3
FROM python:3.11-slim

LABEL project="{cfg["project_code"]}"
LABEL description="Scoring de mora — {image}"
LABEL maintainer="equipo-datos@empresa.com"

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    MODEL_ENV=production \\
    PORT={docker_port}

WORKDIR /app

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY mlops_pipeline/src/ft_engineering.py  ./src/
COPY mlops_pipeline/src/model_deploy.py    ./src/
COPY mlops_pipeline/src/config.json        ./src/

COPY {cfg["paths"]["model_file"]}       ./{cfg["paths"]["model_file"]}
COPY {cfg["paths"]["model_meta_file"]}  ./{cfg["paths"]["model_meta_file"]}
COPY {cfg["paths"]["pipeline_ml_file"]}  ./{cfg["paths"]["pipeline_ml_file"]}
COPY {cfg["paths"]["pipeline_base_file"]} ./{cfg["paths"]["pipeline_base_file"]}

RUN mkdir -p /app/{cfg["paths"]["artifacts_dir"]} /app/{cfg["paths"]["reports_dir"]}

EXPOSE {docker_port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{docker_port}/health')"

CMD ["python", "src/model_deploy.py", "--serve", "--host", "0.0.0.0", "--port", "{docker_port}"]
"""
    # El Dockerfile va en la raíz del git repo (repo_root/mlops_pipeline/),
    # NO en la carpeta padre del curso (repo_root/).
    git_root = repo_root / "mlops_pipeline"
    git_root.mkdir(parents=True, exist_ok=True)
    path = git_root / "Dockerfile"
    with open(path, "w") as f:
        f.write(content)
    logger.info("Dockerfile generado: %s  (image=%s, port=%d)", path, image, docker_port)


def generate_dockerignore(repo_root: Path, cfg: dict) -> None:
    """.dockerignore — excluye paths de cfg que no deben ir en la imagen."""
    content = f"""# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.env

# Git
.git/
.gitignore

# Notebooks
*.ipynb
.ipynb_checkpoints/

# Datos crudos (cfg.paths.base_data_csv / train_reference_file)
{cfg["paths"]["base_data_csv"]}
{cfg["paths"]["train_reference_file"]}

# Reports (se generan en ejecución)
{cfg["paths"]["reports_dir"]}/

# CI/CD
.sonarcloud.properties
sonar-project.properties
Jenkinsfile
"""
    git_root = repo_root / "mlops_pipeline"
    with open(git_root / ".dockerignore", "w") as f:
        f.write(content)
    logger.info(".dockerignore generado: %s", git_root / ".dockerignore")



def generate_requirements_deploy(repo_root: Path) -> None:
    """requirements_deploy.txt mínimo para la imagen Docker de producción.
    Distinto del requirements.txt del repo (que incluye jupyter, matplotlib, etc.)."""
    content = """# requirements_deploy.txt — Solo producción (generado por model_deploy.py)
# Para desarrollo usa requirements.txt del repo.
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
flask>=3.0.0
python-dateutil>=2.8.0
"""
    git_root = repo_root / "mlops_pipeline"
    with open(git_root / "requirements_deploy.txt", "w") as f:
        f.write(content)
    logger.info("requirements_deploy.txt generado: %s", git_root / "requirements_deploy.txt")


def generate_deploy_summary(repo_root: Path, meta: dict, cfg: dict) -> None:
    """
    Genera cfg.paths.deploy_summary_file con todos los parámetros de
    despliegue consolidados desde meta y config.json.
    """
    d = cfg.get("deploy", {})
    summary = {
        "deploy_timestamp":   datetime.now().isoformat(),
        "project_code":       cfg["project_code"],
        "model_name":         meta["model_name"],
        "threshold":          meta["threshold"],
        "threshold_strategy": meta.get("threshold_strategy"),
        "train_size":         meta.get("train_size"),
        "test_size":          meta.get("test_size"),
        "train_mora_rate":    meta.get("train_mora_rate"),
        "test_mora_rate":     meta.get("test_mora_rate"),
        "feature_count":      meta.get("feature_count"),
        "train_cutoff":       cfg["split"]["train_cutoff"],
        "cv_recall_mean":     (meta.get("cv_metrics", {})
                               .get("recall", {}).get("mean")),
        "cv_roc_auc_mean":    (meta.get("cv_metrics", {})
                               .get("roc_auc", {}).get("mean")),
        "test_roc_auc":       meta.get("test_metrics", {}).get("roc_auc"),
        "gap_auc":            meta.get("gap_auc"),
        # Parámetros de despliegue — todos desde cfg.deploy
        "api_endpoints":      ["GET /health", "GET /metrics", "POST /predict"],
        "docker_image":       d.get("docker_image"),
        "docker_port":        d.get("docker_port"),
        "api_version":        d.get("api_version"),
        "batch_max_records":  d.get("batch_max_records"),
        "risk_thresholds":    d.get("risk_thresholds"),
        # Rutas clave desde cfg.paths
        "paths": {
            "model":          cfg["paths"]["model_file"],
            "pipeline_ml":    cfg["paths"]["pipeline_ml_file"],
            "logs":           cfg["paths"]["logs_file"],
            "deploy_summary": cfg["paths"]["deploy_summary_file"],
        },
    }

    summary_path = repo_root / cfg["paths"]["deploy_summary_file"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("deploy_summary guardado: %s", summary_path)


# ──────────────────────────────────────────────────────────
#  Ejecución principal
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="model_deploy — artefactos Docker y API Flask"
    )
    parser.add_argument("--serve",  action="store_true",
                        help="Levantar la API Flask (usa cfg.deploy.host/port)")
    parser.add_argument("--host",   default=None,
                        help="Override host (default: cfg.deploy.host)")
    parser.add_argument("--port",   type=int, default=None,
                        help="Override puerto (default: cfg.deploy.port)")
    parser.add_argument("--batch",  type=str, default=None,
                        help="CSV para predicción batch inmediata")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    model, meta, cfg, repo_root = load_deployed_model()
    d    = cfg.get("deploy", {})
    host = args.host or d.get("host", "127.0.0.1")
    port = args.port or d.get("port", 5000)

    # ── Batch directo ─────────────────────────────────────
    if args.batch:
        result = predict_batch(
            args.batch, model=model, meta=meta, cfg=cfg, repo_root=repo_root,
        )
        out = Path(args.batch).stem + "_predictions.csv"
        result.to_csv(out, index=False)
        logger.info("Predicciones: %s", out)
        logger.info("\n%s", result.head(10).to_string(index=False))

    # ── Artefactos Docker ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("ARTEFACTOS DE DESPLIEGUE")
    logger.info("  proyecto : %s", cfg["project_code"])
    logger.info("  imagen   : %s", d.get("docker_image"))
    logger.info("  puerto   : %d", port)
    logger.info("=" * 60)

    generate_dockerfile(repo_root, cfg)
    generate_dockerignore(repo_root, cfg)
    generate_requirements_deploy(repo_root)
    generate_deploy_summary(repo_root, meta, cfg)

    logger.info("docker build -t %s:latest .", d.get("docker_image"))
    logger.info("docker run -p %d:%d %s:latest",
                d.get("docker_port"), d.get("docker_port"), d.get("docker_image"))

    # ── Flask ─────────────────────────────────────────────
    if args.serve:
        logger.info("Flask en %s:%d", host, port)
        app = create_app(model=model, meta=meta, cfg=cfg, repo_root=repo_root)
        app.run(host=host, port=port, debug=False)
    else:
        # Demo
        csv_path = repo_root / cfg["paths"]["base_data_csv"]
        if not csv_path.exists():
            csv_path = Path(__file__).resolve().parent / cfg["paths"]["base_data_csv"]
        if csv_path.exists():
            raw_df = pd.read_csv(csv_path)
            result = predict_batch(
                raw_df.tail(50).copy(),
                model=model, meta=meta, cfg=cfg, repo_root=repo_root,
            )
            logger.info("Demo:\n%s",
                result[["score_mora","pred_mora","riesgo","fecha_prediccion"]]
                .head(10).to_string(index=False))
            logger.info("Mora predicha: %d/%d (%.1f%%)",
                result["pred_mora"].sum(), len(result),
                result["pred_mora"].mean() * 100)