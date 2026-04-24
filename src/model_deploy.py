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
import re
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

def resolve_runtime_paths(cfg_global: dict, cfg: dict) -> dict:
    top_paths = cfg_global.get("paths", {})
    paths = dict(cfg.get("paths", {}))

    artifacts_dir = Path(paths["artifacts_dir"])
    reports_dir   = Path(paths["reports_dir"])

    derived = {
        "model_file":           artifacts_dir / Path(top_paths.get("model_file", "best_model.joblib")).name,
        "model_meta_file":      artifacts_dir / Path(top_paths.get("model_meta_file", "best_model_meta.json")).name,
        "train_reference_file": artifacts_dir / Path(top_paths.get("train_reference_file", "train_reference.csv")).name,
        "logs_file":            artifacts_dir / Path(top_paths.get("logs_file", "prediction_logs.csv")).name,
        "pipeline_ml_file":     artifacts_dir / Path(top_paths.get("pipeline_ml_file", "pipeline_ml.pkl")).name,
        "pipeline_base_file":   artifacts_dir / Path(top_paths.get("pipeline_base_file", "pipeline_base.pkl")).name,
        "deploy_summary_file":  artifacts_dir / Path(top_paths.get("deploy_summary_file", "deploy_summary.json")).name,
        "metrics_file":         reports_dir / Path(top_paths.get("metrics_file", "metrics_latest.json")).name,
        "drift_report_file":    reports_dir / Path(top_paths.get("drift_report_file", "drift_report.csv")).name,
    }
    for key, derived_path in derived.items():
        if paths.get(key) == top_paths.get(key) or key not in paths:
            paths[key] = str(derived_path)
    return paths


def get_event_metadata(cfg: dict) -> dict:
    event_col = cfg["target"]["event_col"]
    event_title = event_col.replace("_", " ").title()
    return {
        "event_col": event_col,
        "event_title": event_title,
        "score_col": f"score_{event_col}",
        "pred_col": f"pred_{event_col}",
        "actual_col": f"{event_col}_real",
    }


# ──────────────────────────────────────────────────────────
#  Carga del modelo y metadatos

def load_deployed_model(
    config_path: str | Path | None = None,
    use_case: str = "scoring_mora",
):
    """Carga modelo, metadatos y config resuelto para un use_case."""
    import sys
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from ft_engineering import load_config, resolve_cfg

    cfg_global, repo_root = load_config(config_path)
    cfg = resolve_cfg(cfg_global, use_case)
    cfg["paths"] = resolve_runtime_paths(cfg_global, cfg)

    model_path = Path(cfg["paths"]["model_file"])
    meta_path  = Path(cfg["paths"]["model_meta_file"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}. "
            "Ejecuta model_training.py primero."
        )

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    safe_name = re.sub(r'[\x00-\x1f\x7f]', '_', str(meta["model_name"]))
    logger.info("Modelo cargado: %s  (threshold=%.4f)", safe_name, meta["threshold"])
    return model, meta, cfg, repo_root

# ──────────────────────────────────────────────────────────
#  Clasificación de riesgo — configurable desde config.json

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

def _load_batch_input(input_data: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load raw input as DataFrame from a CSV path or an existing DataFrame."""
    if isinstance(input_data, (str, Path)):
        logger.info("Cargando CSV: %s", input_data)
        return pd.read_csv(input_data)
    return input_data.copy()


def _parse_predict_request(request) -> "tuple[pd.DataFrame | None, tuple | None]":
    """
    Parse a Flask request body as DataFrame.
    Returns (df, None) on success or (None, error_response) on failure.
    """
    from flask import jsonify
    ct = request.content_type or ""
    if "application/json" in ct:
        data = request.get_json(force=True)
        df = pd.DataFrame(
            data["records"] if isinstance(data, dict) and "records" in data else data
        )
        return df, None
    if "text/csv" in ct:
        from io import StringIO
        return pd.read_csv(StringIO(request.data.decode("utf-8"))), None
    return None, (jsonify({"error": "Content-Type no soportado"}), 415)


def _handle_predict(request, model, meta: dict, cfg: dict, repo_root: Path):
    """Handle POST /predict: parse input, run batch, return JSON response."""
    from flask import jsonify
    try:
        df_input, err = _parse_predict_request(request)
        if err is not None:
            return err
        if len(df_input) == 0:
            return jsonify({"error": "Payload vacío"}), 400
        result   = predict_batch(df_input, model=model, meta=meta, cfg=cfg, repo_root=repo_root)
        pred_col = get_event_metadata(cfg)["pred_col"]
        return jsonify({
            "n_records":    len(result),
            "n_event_pred": int(result[pred_col].sum()),
            "predictions":  result.to_dict(orient="records"),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Error en /predict")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────
#  Pipeline de predicción batch

def predict_batch(
    input_data:      pd.DataFrame | str | Path,
    model=None,
    meta:            dict | None = None,
    cfg:             dict | None = None,
    repo_root:       Path | None = None,
    log_predictions: bool = True,
) -> pd.DataFrame:
    """
    Genera predicciones del evento objetivo para un batch de registros nuevos.

    Lee toda la configuración desde cfg (config.json):
      - Umbral de decisión: meta["threshold"]
      - Columnas a excluir: cfg.feature_engineering.leakage_cols
      - Columna de fecha: cfg.split.date_col
      - Pipeline serializado: cfg.paths.pipeline_ml_file
      - Clasificación de riesgo: cfg.deploy.risk_thresholds
      - Límite de batch: cfg.deploy.batch_max_records

    Returns:
        DataFrame con: prediction_id, score_event, pred_event, risk_level,
                       fecha_prediccion. Si el input tenía fecha_prestamo,
                       se incluye como primera columna.
    """
    import sys
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from ft_engineering import apply_pipeline_steps, build_pipeline_stateless

    if model is None or meta is None:
        model, meta, cfg, repo_root = load_deployed_model()

    # ── Leer parámetros desde config ──────────────────────
    threshold    = meta["threshold"]
    deploy_cfg   = cfg.get("deploy", {})
    max_records  = deploy_cfg.get("batch_max_records", 10_000)
    risk_cfg     = deploy_cfg.get("risk_thresholds", {"bajo": 0.5, "medio": 1.0})
    date_col     = cfg["split"]["date_col"]
    event_meta   = get_event_metadata(cfg)
    event_col    = event_meta["event_col"]
    score_col    = event_meta["score_col"]
    pred_col     = event_meta["pred_col"]
    leakage_cols = cfg["feature_engineering"]["leakage_cols"]

    classify_risk = _make_risk_classifier(threshold, risk_cfg)

    # ── Cargar datos ──────────────────────────────────────
    df_raw = _load_batch_input(input_data)

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

    # ── Pasos SIN estado: aplicar freshly desde config (no requieren pkl) ──
    # CrearFeaturesDerivadas + LimpiarTendenciaIngresos no aprenden parámetros
    # de los datos, por lo que no necesitan ser cargados desde disco.
    # IMPORTANTE: df_raw debe contener fecha_prestamo para que
    # CrearFeaturesDerivadas pueda crear anio_prestamo, mes_prestamo, etc.
    pipeline_stateless = build_pipeline_stateless(cfg)
    df_pre = pipeline_stateless.fit_transform(df_raw)   # fit() es no-op aquí

    # ── Pasos CON estado: cargar pipeline_base ajustado sobre train ──────────
    # Contiene ImputacionSegmentada (medianas por tipo_laboral) y Winsorizar
    # (caps al p99), ambos ajustados SOLO sobre el train set en model_training.py.
    # Cargar el pkl garantiza que se apliquen exactamente los mismos parámetros
    # de entrenamiento — sin recalcular nada sobre datos nuevos.
    pipeline_base_path = repo_root / cfg["paths"]["pipeline_base_file"]
    if not pipeline_base_path.exists():
        raise FileNotFoundError(
            f"pipeline_base no encontrado: {pipeline_base_path}. "
            "Ejecuta model_training.py primero."
        )
    pipeline_base  = joblib.load(pipeline_base_path)
    logger.info("pipeline_base cargado: %s", pipeline_base_path)
    df_transformed = apply_pipeline_steps(pipeline_base, df_pre)

    # Limpiar residuos (target, fecha) si quedaron tras el pipeline
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
    x_df = pd.DataFrame(X, columns=feature_names)

    # ── Predicción ────────────────────────────────────────
    t0     = time.time()
    scores = model.predict_proba(x_df)[:, 1]
    preds  = (scores >= threshold).astype(int)
    elapsed = time.time() - t0
    logger.info(
        "Predicciones: %d records en %.3fs | positivos=%d (%.1f%%)",
        n_records, elapsed, preds.sum(), preds.mean() * 100,
    )

    # ── Construir resultado ───────────────────────────────
    result = pd.DataFrame({
        "prediction_id":    [str(uuid.uuid4()) for _ in range(n_records)],
        score_col:          np.round(scores, 6),
        pred_col:           preds,
        "risk_level":       [classify_risk(s) for s in scores],
        "fecha_prediccion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    if fechas is not None:
        result.insert(0, date_col, fechas.values)

    # ── Log (cfg.paths.logs_file) ─────────────────────────
    if log_predictions and repo_root is not None:
        _log_predictions(result, x_df, cfg, repo_root)

    return result

def _log_predictions(
    result_df: pd.DataFrame,
    x_df:      pd.DataFrame,
    cfg:       dict,
    repo_root: Path,
) -> None:
    """Persiste predicciones en cfg.paths.logs_file para model_monitoring.py."""
    logs_path = repo_root / cfg["paths"]["logs_file"]
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    log_df = pd.concat(
        [result_df.reset_index(drop=True), x_df.reset_index(drop=True)],
        axis=1,
    )
    if logs_path.exists():
        log_df.to_csv(logs_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(logs_path, index=False)

    logger.info("Log actualizado: %s  (+%d rows)", logs_path, len(log_df))

# ──────────────────────────────────────────────────────────
#  API Flask — host/puerto/versión desde cfg.deploy

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
        return _handle_predict(request, model, meta, cfg, repo_root)

    return app

# ──────────────────────────────────────────────────────────
#  Generación de artefactos Docker — todo desde config.json

def generate_dockerfile(repo_root: Path, cfg: dict) -> None:
    """Dockerfile usando cfg.deploy.docker_image, .docker_port y cfg.paths."""
    d           = cfg.get("deploy", {})
    image       = d.get("docker_image", cfg["project_code"] + "-scoring")
    docker_port = d.get("docker_port", 5000)
    use_case    = cfg["use_case"]

    content = f"""# Dockerfile — {image}
# Proyecto: {cfg["project_code"]}  |  API: {d.get("api_version", "v1")}
# Build desde mlops_pipeline/: docker build -t {image} .
FROM python:3.11-slim

LABEL project="{cfg["project_code"]}"
LABEL description="Scoring del evento — {image}"
LABEL maintainer="{d.get("maintainer_email", cfg["project_code"] + "@empresa.com")}"

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    MODEL_ENV=production \\
    PORT={docker_port}

WORKDIR /app

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt && \\
    mkdir -p mlops_pipeline/src {cfg["paths"]["artifacts_dir"]} {cfg["paths"]["reports_dir"]}

COPY src/ft_engineering.py  mlops_pipeline/src/
COPY src/model_deploy.py    mlops_pipeline/src/
COPY src/config.json        mlops_pipeline/src/

COPY {cfg["paths"]["model_file"]}       ./{cfg["paths"]["model_file"]}
COPY {cfg["paths"]["model_meta_file"]}  ./{cfg["paths"]["model_meta_file"]}
COPY {cfg["paths"]["pipeline_ml_file"]}  ./{cfg["paths"]["pipeline_ml_file"]}
COPY {cfg["paths"]["pipeline_base_file"]} ./{cfg["paths"]["pipeline_base_file"]}

EXPOSE {docker_port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \\
  CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{{os.environ.get(\"PORT\", {docker_port})}}/health')"

CMD ["sh", "-c", "python mlops_pipeline/src/model_deploy.py --serve --host 0.0.0.0 --port ${{PORT}} --use-case {use_case}"]
"""
    path = repo_root / "Dockerfile"
    with open(path, "w") as f:
        f.write(content)
    logger.info("Dockerfile generado: %s  (image=%s, port=%d)", path, image, docker_port)

def generate_dockerignore(repo_root: Path, cfg: dict) -> None:
    """.dockerignore — excluye paths de cfg que no deben ir en la imagen."""
    content = f"""# Python
__pycache__/
*.py[cod]
*.egg-info/
*-venv/
.venv/
venv/
.env
.coverage

# Git
.git/
.gitignore

# Notebooks y análisis
*.ipynb
.ipynb_checkpoints/

# Datos crudos y de estado
{cfg["paths"].get("base_data_csv", "")}
data/
{cfg["paths"]["train_reference_file"]}

# Reports (se generan en ejecución)
{cfg["paths"]["reports_dir"]}/

# Tests y cobertura
tests/
coverage.xml
.pytest_cache/

# Scripts de entorno y ejecución local
set_up.sh
set_up.bat
run_pipeline.sh
run_pipeline.bat

# CI/CD y docs
.github/
sonar-project.properties
README.md
"""
    with open(repo_root / ".dockerignore", "w") as f:
        f.write(content)
    logger.info(".dockerignore generado: %s", repo_root / ".dockerignore")

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
    with open(repo_root / "requirements_deploy.txt", "w") as f:
        f.write(content)
    logger.info("requirements_deploy.txt generado: %s", repo_root / "requirements_deploy.txt")

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
        "train_event_rate":   meta.get("train_event_rate"),
        "test_event_rate":    meta.get("test_event_rate"),
        "feature_count":      meta.get("feature_count"),
        "train_cutoff":       cfg["split"].get("train_cutoff"),
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
    parser.add_argument(
        "--use-case", type=str, dest="use_case", required=True,
        help="Caso de uso definido en config.json > use_cases.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Ruta opcional a config.json.",
    )
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    model, meta, cfg, repo_root = load_deployed_model(
        config_path=Path(args.config) if args.config else None,
        use_case=args.use_case,
    )
    d    = cfg.get("deploy", {})
    host = args.host or d.get("host", "127.0.0.1")
    port = args.port or d.get("port", 5000)

    if args.batch:
        result = predict_batch(
            args.batch, model=model, meta=meta, cfg=cfg, repo_root=repo_root,
        )
        out = Path(args.batch).stem + "_predictions.csv"
        result.to_csv(out, index=False)
        logger.info("Predicciones: %s", out)
        logger.info("\n%s", result.head(10).to_string(index=False))

    logger.info("=" * 60)
    logger.info("ARTEFACTOS DE DESPLIEGUE")
    logger.info("  use_case : %s", args.use_case)
    logger.info("  proyecto : %s", cfg["project_code"])
    logger.info("  imagen   : %s", d.get("docker_image"))
    logger.info("  puerto   : %d", port)
    logger.info("=" * 60)

    if not args.serve:
        cfg["use_case"] = args.use_case
        generate_dockerfile(repo_root, cfg)
        generate_dockerignore(repo_root, cfg)
        generate_requirements_deploy(repo_root)
        generate_deploy_summary(repo_root, meta, cfg)
        logger.info("docker build -t %s:latest .", d.get("docker_image"))
        logger.info("docker run -p %d:%d %s:latest",
                    d.get("docker_port"), d.get("docker_port"), d.get("docker_image"))

    if args.serve:
        logger.info("Flask en %s:%d", host, port)
        app = create_app(model=model, meta=meta, cfg=cfg, repo_root=repo_root)
        app.run(host=host, port=port, debug=False)
    else:
        csv_path = Path(cfg["paths"]["base_data_csv"])
        if csv_path.exists():
            raw_df = pd.read_csv(csv_path)
            result = predict_batch(
                raw_df.tail(50).copy(),
                model=model, meta=meta, cfg=cfg, repo_root=repo_root,
            )
            event_meta = get_event_metadata(cfg)
            logger.info("Demo:\n%s",
                result[[event_meta["score_col"], event_meta["pred_col"], "risk_level", "fecha_prediccion"]]
                .head(10).to_string(index=False))
            logger.info("Evento predicho: %d/%d (%.1f%%)",
                result[event_meta["pred_col"]].sum(), len(result),
                result[event_meta["pred_col"]].mean() * 100)

# ──────────────────────────────────────────────────────────