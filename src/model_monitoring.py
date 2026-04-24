"""
model_monitoring.py
===================
Monitoreo del modelo en producción: DataDrift y degradación de performance.

¿Qué hace este script?
  1. Carga el dataset de referencia (train) y los datos de producción (test
     o prediction_logs).
  2. Genera predicciones del modelo desplegado sobre los datos de producción.
  3. Calcula el PSI (Population Stability Index) por feature para detectar
     DataDrift — cambios en la distribución de la población.
  4. Calcula métricas de performance sobre ventanas temporales para detectar
     degradación del modelo.
  5. Genera alertas cuando PSI > umbral (config.monitoring.psi_threshold).
  6. Exporta un reporte estructurado (CSV + HTML) para la pestaña de monitoreo.

Métricas de DataDrift:
  PSI < 0.10  → Estable     (no acción requerida)
  PSI 0.10–0.20 → Monitorear  (investigar causa)
  PSI > 0.20  → Drift severo (considerar reentrenamiento)

Fórmula PSI:
  PSI = Σ (p_prod - p_ref) × ln(p_prod / p_ref)
  donde p_ref y p_prod son las frecuencias relativas por bin del feature.

Periodicidad:
  Configurable en config.monitoring. Por defecto el script procesa todos
  los datos de producción disponibles en prediction_logs.csv. En producción
  real se ejecutaría como un cron job (diario/semanal).

Outputs:
  reports/drift_report.csv          → PSI por feature (para dashboards externos)
  reports/monitoring_report.html    → Dashboard completo de monitoreo
  reports/monitoring_report.json    → Métricas estructuradas
  reports/monitor_psi_heatmap.png   → Heatmap de PSI por feature
  reports/monitor_score_drift.png   → Deriva del score del evento en el tiempo
  reports/monitor_performance.png   → Métricas de performance por ventana

Uso:
  python model_monitoring.py
  python model_monitoring.py --logs path/to/prediction_logs.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


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
#  PSI — Population Stability Index

def compute_psi_feature(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Calcula el PSI para un único feature numérico.

    Usa los bin-edges del dataset de referencia para garantizar que ambas
    distribuciones se comparen sobre el mismo espacio.

    Args:
        reference:  valores del feature en el conjunto de referencia (train).
        production: valores del feature en producción.
        n_bins:     número de bins para discretizar.
        eps:        suavizado para evitar log(0).

    Returns:
        float: PSI score.
    """
    # Definir bins sobre la referencia
    _, bin_edges = np.histogram(reference, bins=n_bins)
    bin_edges[0]  -= 1e-9   # incluir el mínimo exacto
    bin_edges[-1] += 1e-9   # incluir el máximo exacto

    ref_counts,  _ = np.histogram(reference,  bins=bin_edges)
    prod_counts, _ = np.histogram(production, bins=bin_edges)

    ref_pct  = (ref_counts  + eps) / (len(reference)  + n_bins * eps)
    prod_pct = (prod_counts + eps) / (len(production) + n_bins * eps)

    psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
    return round(psi, 6)


def compute_psi_all_features(
    reference_df:  pd.DataFrame,
    production_df: pd.DataFrame,
    numeric_features: list[str],
    n_bins: int = 10,
    psi_thresholds: dict = None,
) -> pd.DataFrame:
    """
    Calcula el PSI para todos los features numéricos.

    Args:
        reference_df:     DataFrame de referencia (train).
        production_df:    DataFrame de producción.
        numeric_features: lista de columnas a evaluar.
        n_bins:           número de bins.
        psi_thresholds:   dict con claves 'stable' y 'monitor' (valores float).
                          Si es None usa los valores de config.monitoring.psi_thresholds
                          o los defaults 0.10 / 0.20.

    Returns:
        DataFrame con columnas: feature, psi, status.
    """
    thresholds    = psi_thresholds or {}
    thr_stable    = float(thresholds.get("stable",  0.10))
    thr_monitor   = float(thresholds.get("monitor", 0.20))

    results = []
    for feat in numeric_features:
        if feat not in reference_df.columns or feat not in production_df.columns:
            continue
        ref_vals  = reference_df[feat].dropna().values
        prod_vals = production_df[feat].dropna().values
        if len(ref_vals) < 2 or len(prod_vals) < 2:
            continue
        psi = compute_psi_feature(ref_vals, prod_vals, n_bins=n_bins)
        results.append({"feature": feat, "psi": psi})

    df = pd.DataFrame(results).sort_values("psi", ascending=False)

    def _status(psi):
        if psi < thr_stable:
            return "Estable"
        if psi < thr_monitor:
            return "Monitorear"
        return "Drift"

    df["status"] = df["psi"].apply(_status)
    return df.reset_index(drop=True)

def compute_psi_score(
    reference_scores: np.ndarray,
    production_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """PSI sobre el score del evento (output del modelo)."""
    return compute_psi_feature(reference_scores, production_scores, n_bins=n_bins)

# ──────────────────────────────────────────────────────────
#  Métricas de performance por ventana temporal

def performance_by_window(
    logs_df: pd.DataFrame,
    threshold: float,
    score_col: str,
    actual_col: str,
    window_col: str = "window",
) -> pd.DataFrame:
    """
    Calcula métricas de performance por ventana (mes, semana, etc.).

    Requiere que logs_df tenga columnas:
      - score_mora:  probabilidad predicha por el modelo
      - mora_real:   etiqueta real (si está disponible; 0/1 o NaN)
      - window:      identificador de la ventana temporal

    Si mora_real no está disponible, retorna solo métricas de distribución
    (media del score, % predicho como mora). Es la situación real en
    producción donde el label tarda en materializarse.

    Returns:
        DataFrame con métricas por ventana.
    """
    rows = []
    for window, grp in logs_df.groupby(window_col, sort=True):
        row = {
            "window":        window,
            "n":             len(grp),
            "score_mean":    round(grp[score_col].mean(), 4),
            "score_std":     round(grp[score_col].std(),  4),
            "pct_pred_event": round((grp[score_col] >= threshold).mean(), 4),
        }
        # Si hay etiquetas reales disponibles
        if actual_col in grp.columns and grp[actual_col].notna().sum() > 0:
            labeled = grp.dropna(subset=[actual_col])
            y_true  = labeled[actual_col].astype(int).values
            y_pred  = (labeled[score_col] >= threshold).astype(int).values
            y_proba = labeled[score_col].values

            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())

            row.update({
                "event_rate_real": round(y_true.mean(), 4),
                "recall_event":    round(tp / max(tp + fn, 1), 4),
                "precision_event": round(tp / max(tp + fp, 1), 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "n_labeled": len(labeled),
            })
            try:
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y_true)) > 1:
                    row["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
            except Exception:
                pass
        rows.append(row)

    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────
#  Gráficos de monitoreo

def plot_psi_heatmap(
    psi_df: pd.DataFrame,
    save_path: Path,
    top_n: int = 25,
    psi_thresholds: dict = None,
) -> None:
    """
    Heatmap de PSI por feature. Colores: verde=estable, naranja=monitorear,
    rojo=drift. Muestra los top_n features con mayor PSI.
    """
    thresholds  = psi_thresholds or {}
    thr_stable  = float(thresholds.get("stable",  0.10))
    thr_monitor = float(thresholds.get("monitor", 0.20))

    df = psi_df.head(top_n).copy()

    color_map = {"Estable": "#1D9E75", "Monitorear": "#FFC107", "Drift": "#E24B4A"}
    colors = [color_map[s] for s in df["status"]]

    _, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    bars = ax.barh(df["feature"][::-1], df["psi"][::-1],
                   color=colors[::-1], alpha=0.85, edgecolor="white")

    ax.axvline(thr_stable,  color="#FFC107", linestyle="--", linewidth=1.2,
               label=f"Umbral monitoreo ({thr_stable})")
    ax.axvline(thr_monitor, color="#E24B4A", linestyle="--", linewidth=1.2,
               label=f"Umbral drift ({thr_monitor})")

    for bar, val in zip(bars, df["psi"][::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_xlabel("PSI (Population Stability Index)")
    ax.set_title("DataDrift — PSI por Feature\n"
                 "(Verde=Estable | Naranja=Monitorear | Rojo=Drift)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("PSI heatmap guardado: %s", save_path)

def plot_score_drift(
    ref_scores: np.ndarray,
    prod_scores: np.ndarray,
    model_name: str,
    psi_score: float,
    save_path: Path,
    event_title: str = "Evento",
) -> None:
    """
    Distribución del score del evento: referencia vs producción.
    Detecta visualmente el DataDrift en el output del modelo.
    """
    _, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Histograma comparativo
    ax = axes[0]
    ax.hist(ref_scores,  bins=30, alpha=0.6, color="#378ADD",
            label=f"Referencia (train, n={len(ref_scores):,})", density=True)
    ax.hist(prod_scores, bins=30, alpha=0.6, color="#E24B4A",
            label=f"Produccion (test, n={len(prod_scores):,})", density=True)
    ax.set_xlabel("Score del evento")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución del Score — PSI={psi_score:.4f}",
                 fontweight="bold", fontsize=11)
    ax.legend()

    # Estadísticas descriptivas
    ax = axes[1]
    stats = {
        "media":   [ref_scores.mean(),  prod_scores.mean()],
        "mediana": [np.median(ref_scores), np.median(prod_scores)],
        "p75":     [np.percentile(ref_scores, 75), np.percentile(prod_scores, 75)],
        "p90":     [np.percentile(ref_scores, 90), np.percentile(prod_scores, 90)],
        "p95":     [np.percentile(ref_scores, 95), np.percentile(prod_scores, 95)],
    }
    stats_df = pd.DataFrame(stats, index=["Referencia", "Produccion"]).T

    x = np.arange(len(stats_df))
    width = 0.35
    ax.bar(x - width / 2, stats_df["Referencia"], width, color="#378ADD",
           alpha=0.8, label="Referencia", edgecolor="white")
    ax.bar(x + width / 2, stats_df["Produccion"], width, color="#E24B4A",
           alpha=0.8, label="Produccion", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df.index)
    ax.set_title("Estadísticas del Score", fontweight="bold", fontsize=11)
    ax.legend()

    plt.suptitle(f"DataDrift del Score de {event_title} — {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Score drift guardado: %s", save_path)

def plot_performance_over_time(
    perf_df: pd.DataFrame,
    save_path: Path,
    event_title: str = "Evento",
) -> None:
    """
    Evolución de métricas de performance por ventana temporal.
    Permite detectar degradación gradual del modelo en producción.
    """
    has_labels = "recall_event" in perf_df.columns

    _, axes = plt.subplots(1, 2 if has_labels else 1,
                           figsize=(13 if has_labels else 8, 4))
    if not has_labels:
        axes = [axes]

    # Score medio por ventana (siempre disponible)
    ax = axes[0]
    ax.plot(perf_df["window"].astype(str), perf_df["score_mean"],
            "o-", color="#1D9E75", linewidth=2, label="Score medio")
    ax.fill_between(
        perf_df["window"].astype(str),
        perf_df["score_mean"] - perf_df["score_std"],
        perf_df["score_mean"] + perf_df["score_std"],
        alpha=0.2, color="#1D9E75",
    )
    ax.bar(perf_df["window"].astype(str), perf_df["pct_pred_event"],
           alpha=0.3, color="#E24B4A", label="% pred evento")
    ax.set_xlabel("Ventana")
    ax.set_ylabel("Score / % predicho evento")
    ax.set_title(f"Evolución del Score de {event_title}", fontweight="bold", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    # Métricas de performance (solo si hay etiquetas)
    if has_labels and len(axes) > 1:
        ax = axes[1]
        if "roc_auc" in perf_df.columns:
            ax.plot(perf_df["window"].astype(str), perf_df["roc_auc"],
                    "o-", color="#1D9E75", linewidth=2, label="ROC-AUC")
        ax.plot(perf_df["window"].astype(str), perf_df["recall_event"],
                "s-", color="#E24B4A", linewidth=2, label="Recall evento")
        ax.plot(perf_df["window"].astype(str), perf_df["precision_event"],
                "^-", color="#378ADD", linewidth=2, label="Precision evento")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Ventana")
        ax.set_ylabel("Score")
        ax.set_title("Métricas de Performance por Ventana",
                     fontweight="bold", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.legend()

    plt.suptitle("Monitoreo del Modelo — Evolución Temporal",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Performance monitoring guardado: %s", save_path)

def plot_feature_drift_detail(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    top_drift_features: list[str],
    save_path: Path,
) -> None:
    """
    Histogramas comparativos para los features con mayor PSI.
    Permite entender qué tipo de drift está ocurriendo.
    """
    n = min(len(top_drift_features), 6)
    if n == 0:
        return

    cols = 3
    rows = (n + cols - 1) // cols
    _, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, feat in enumerate(top_drift_features[:n]):
        ax = axes[i]
        ref_vals  = reference_df[feat].dropna()
        prod_vals = production_df[feat].dropna()
        ax.hist(ref_vals,  bins=25, alpha=0.6, color="#378ADD",
                label="Referencia", density=True)
        ax.hist(prod_vals, bins=25, alpha=0.6, color="#E24B4A",
                label="Produccion", density=True)
        ax.set_title(feat.replace("numeric__", "").replace("categoric__", ""),
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Ocultar ejes sobrantes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Detalle de Drift — Top Features por PSI",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Feature drift detail guardado: %s", save_path)

# ──────────────────────────────────────────────────────────
#  Reporte HTML de monitoreo

def build_monitoring_html(
    model_name:    str,
    psi_df:        pd.DataFrame,
    psi_score:     float,
    perf_df:       pd.DataFrame,
    n_ref:         int,
    n_prod:        int,
    psi_threshold: float,
    event_title:   str = "Evento",
) -> str:

    def status_badge(status):
        colors = {"Estable": "#1D9E75", "Monitorear": "#FFC107",
                  "Drift": "#E24B4A", "OK": "#1D9E75", "ALERTA": "#E24B4A"}
        c = colors.get(status, "#999")
        return (f'<span style="background:{c};color:white;padding:3px 10px;'
                f'border-radius:12px;font-size:12px;font-weight:600;">{status}</span>')

    # Tabla PSI
    psi_rows = ""
    for _, row in psi_df.iterrows():
        psi_rows += f"""
        <tr>
          <td>{row['feature']}</td>
          <td style="text-align:right">{row['psi']:.4f}</td>
          <td style="text-align:center">{status_badge(row['status'])}</td>
        </tr>"""

    # Tabla de performance
    perf_cols = ["window", "n", "score_mean", "pct_pred_event"]
    if "recall_event" in perf_df.columns:
        perf_cols += ["event_rate_real", "recall_event", "precision_event"]
    perf_header = "".join(f"<th>{c}</th>" for c in perf_cols)
    perf_rows = ""
    for _, row in perf_df.iterrows():
        cells = ""
        for c in perf_cols:
            val = row.get(c, "–")
            cells += f"<td style='text-align:right'>{val}</td>"
        perf_rows += f"<tr>{cells}</tr>"

    n_drift   = int((psi_df["status"] == "Drift").sum())
    n_monitor = int((psi_df["status"] == "Monitorear").sum())
    if psi_score > psi_threshold:
        global_status = "ALERTA"
    elif psi_score > psi_threshold * 0.5:
        global_status = "Monitorear"
    else:
        global_status = "Estable"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Monitoreo — {model_name}</title>
<style>
  body  {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          margin:0;padding:24px 40px;background:#fff;color:#222;}}
  h1    {{color:#1a1a2e;font-size:26px;margin-bottom:4px;}}
  h2    {{border-bottom:2px solid #eee;padding-bottom:8px;color:#333;}}
  .sub  {{color:#666;font-size:13px;margin-bottom:28px;}}
  .cards{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px;}}
  .card {{background:#f9f9f9;border-left:4px solid #1D9E75;
          padding:10px 16px;border-radius:6px;min-width:130px;}}
  .card .label {{font-size:10px;color:#888;text-transform:uppercase;
                 letter-spacing:.5px;}}
  .card .value {{font-size:24px;font-weight:700;color:#222;}}
  table {{border-collapse:collapse;width:100%;font-size:13px;}}
  th    {{background:#1D9E75;color:white;padding:8px 12px;text-align:left;}}
  td    {{padding:7px 12px;border-bottom:1px solid #eee;}}
  .warn {{background:#fff8e1;border-left:4px solid #FFC107;
          padding:10px 16px;border-radius:4px;font-size:13px;margin:8px 0;}}
  .ok   {{background:#e8f5e9;border-left:4px solid #1D9E75;
          padding:10px 16px;border-radius:4px;font-size:13px;margin:8px 0;}}
  .err  {{background:#fdecea;border-left:4px solid #E24B4A;
          padding:10px 16px;border-radius:4px;font-size:13px;margin:8px 0;}}
</style>
</head>
<body>
<h1>🔍 Monitoreo del Modelo — {model_name}</h1>
<div class="sub">
  Referencia: train (n={n_ref:,}) &nbsp;|&nbsp;
  Produccion: test/logs (n={n_prod:,}) &nbsp;|&nbsp;
  PSI umbral: {psi_threshold} &nbsp;|&nbsp;
  Estado global: {status_badge(global_status)}
</div>

<h2>1. Resumen de DataDrift</h2>
<div class="cards">
  <div class="card"><div class="label">PSI Score (output)</div>
    <div class="value">{psi_score:.4f}</div></div>
  <div class="card" style="border-color:#E24B4A"><div class="label">Features en Drift</div>
    <div class="value">{n_drift}</div></div>
  <div class="card" style="border-color:#FFC107"><div class="label">Features Monitorear</div>
    <div class="value">{n_monitor}</div></div>
  <div class="card" style="border-color:#1D9E75"><div class="label">Features Estables</div>
    <div class="value">{int((psi_df["status"] == "Estable").sum())}</div></div>
</div>
{"<div class='err'>⚠️ ALERTA: PSI del score del evento supera el umbral de " + str(psi_threshold) + ". Se recomienda revisar si hay cambios en la población y considerar reentrenamiento.</div>"
 if psi_score > psi_threshold else
 "<div class='ok'>✅ El score del evento no presenta drift significativo respecto al periodo de entrenamiento.</div>"}

<h2>2. PSI por Feature (top 30)</h2>
<table>
  <tr><th>Feature</th><th>PSI</th><th>Estado</th></tr>
  {psi_rows}
</table>

<h2>3. Evolución Temporal — Score de {event_title}</h2>
<table>
  <tr>{perf_header}</tr>
  {perf_rows}
</table>

<h2>4. Gráficos de Monitoreo</h2>
<ul style="font-size:13px;color:#555;">
  <li>monitor_psi_heatmap.png — PSI por feature</li>
  <li>monitor_score_drift.png — Distribución del score: referencia vs producción</li>
  <li>monitor_performance.png — Evolución del score por ventana temporal</li>
  <li>monitor_feature_drift.png — Histogramas de features con mayor drift</li>
</ul>

<h2>5. Interpretación del PSI</h2>
<table>
  <tr><th>Rango PSI</th><th>Interpretación</th><th>Acción recomendada</th></tr>
  <tr><td>&lt; 0.10</td><td>Distribución estable</td><td>No acción requerida</td></tr>
  <tr><td>0.10 – 0.20</td><td>Cambio menor en la población</td>
      <td>Investigar causa, monitorear de cerca</td></tr>
  <tr><td>&gt; 0.20</td><td>Cambio significativo (drift)</td>
      <td>Analizar features afectados, evaluar reentrenamiento</td></tr>
</table>

<h2>6. Metadatos del Monitoreo</h2>
<table>
  <tr><th>Parámetro</th><th>Valor</th></tr>
  <tr><td>Modelo monitoreado</td><td>{model_name}</td></tr>
  <tr><td>Tamaño referencia (train)</td><td>{n_ref:,}</td></tr>
  <tr><td>Tamaño produccion</td><td>{n_prod:,}</td></tr>
  <tr><td>PSI score (output)</td><td>{psi_score:.4f}</td></tr>
  <tr><td>PSI threshold</td><td>{psi_threshold}</td></tr>
  <tr><td>Estado global</td><td>{global_status}</td></tr>
  <tr><td>Features con drift (&gt;0.20)</td><td>{n_drift}</td></tr>
  <tr><td>Features a monitorear (0.10–0.20)</td><td>{n_monitor}</td></tr>
</table>

<hr style="border:none;border-top:1px solid #eee;margin:32px 0 16px;">
<div style="font-size:11px;color:#aaa;">
  Generado por model_monitoring.py — CDP Entregable 3
</div>
</body>
</html>"""
    return html

# ──────────────────────────────────────────────────────────
#  Ejecución principal


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Model monitoring — DataDrift y performance")
    parser.add_argument("--logs", type=str, default=None,
                        help="Path a prediction_logs.csv. Si no existe, usa X_test como produccion.")
    parser.add_argument(
        "--use-case", type=str, dest="use_case", required=True,
        help="Caso de uso definido en config.json > use_cases.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Ruta opcional a config.json.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ft_engineering import load_features_from_cache, load_config, resolve_cfg

    cfg_global, _ = load_config(Path(args.config) if args.config else None)
    cfg = resolve_cfg(cfg_global, args.use_case)
    paths = resolve_runtime_paths(cfg_global, cfg)

    reports_dir = Path(paths["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    mon_cfg        = cfg.get("monitoring", {})
    psi_threshold  = mon_cfg.get("psi_threshold", 0.20)
    psi_thresholds = mon_cfg.get("psi_thresholds", {"stable": 0.10, "monitor": 0.20})
    psi_n_bins     = int(mon_cfg.get("psi_n_bins", 10))
    min_rows       = mon_cfg.get("min_rows", 200)

    model_path = Path(paths["model_file"])
    meta_path  = Path(paths["model_meta_file"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}. Ejecuta model_training.py primero."
        )
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    model_name = meta["model_name"]
    threshold  = meta["threshold"]
    event_meta = get_event_metadata(cfg)
    event_col  = meta.get("event_col", event_meta["event_col"])
    score_col  = event_meta["score_col"]
    pred_col   = event_meta["pred_col"]
    actual_col = event_meta["actual_col"]
    logger.info("Modelo: %s | Threshold: %.4f | use_case=%s",
                model_name, threshold, args.use_case)

    ref_path = Path(paths["train_reference_file"])
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Dataset de referencia no encontrado: {ref_path}. "
            "Ejecuta model_training.py primero."
        )
    reference_df = pd.read_csv(ref_path)
    logger.info("Referencia cargada: %s  shape=%s", ref_path, reference_df.shape)

    ref_feature_cols = [c for c in reference_df.columns if c != event_col]
    ref_scores = model.predict_proba(reference_df[ref_feature_cols])[:, 1]

    logs_path = Path(args.logs) if args.logs else Path(paths["logs_file"])

    if logs_path.exists():
        logger.info("Cargando logs de produccion: %s", logs_path)
        production_raw = pd.read_csv(logs_path)
    else:
        logger.warning(
            "prediction_logs.csv no encontrado. "
            "Usando X_test como proxy de produccion."
        )
        X_train, X_test, y_train, y_test, _, _, _, _ = load_features_from_cache(
            use_case=args.use_case,
            config_path=Path(args.config) if args.config else None,
        )
        production_raw = X_test.copy()
        production_raw[actual_col] = np.asarray(y_test)

    logger.info("Produccion: shape=%s", production_raw.shape)

    if len(production_raw) < min_rows:
        logger.warning(
            "Produccion tiene solo %d filas (min=%d). "
            "Los resultados pueden no ser representativos.",
            len(production_raw), min_rows,
        )

    pred_date_col     = cfg.get("deploy", {}).get("prediction_date_col", "fecha_prediccion")
    prod_feature_cols = [c for c in production_raw.columns
                         if c not in (event_col, actual_col, score_col,
                                      pred_col, pred_date_col, "window")]
    common_cols = [c for c in ref_feature_cols if c in prod_feature_cols]
    prod_scores = model.predict_proba(production_raw[common_cols])[:, 1]
    production_raw[score_col] = prod_scores

    logger.info("=" * 55)
    logger.info("CALCULO DE PSI — DataDrift")
    logger.info("=" * 55)

    numeric_features = [c for c in ref_feature_cols
                        if reference_df[c].dtype in [np.float64, np.float32,
                                                     np.int64, np.int32]]

    psi_df = compute_psi_all_features(
        reference_df[ref_feature_cols],
        production_raw[common_cols],
        numeric_features=numeric_features,
        n_bins=psi_n_bins,
        psi_thresholds=psi_thresholds,
    )

    psi_score_output = compute_psi_score(ref_scores, prod_scores)

    n_drift   = int((psi_df["status"] == "Drift").sum())
    n_monitor = int((psi_df["status"] == "Monitorear").sum())
    n_stable  = int((psi_df["status"] == "Estable").sum())

    logger.info("PSI score (output del modelo): %.4f  %s",
                psi_score_output,
                "⚠️ ALERTA" if psi_score_output > psi_threshold else "✅ OK")
    logger.info("Features Drift=%d | Monitorear=%d | Estables=%d",
                n_drift, n_monitor, n_stable)
    logger.info("\nTop 10 features por PSI:\n%s",
                psi_df.head(10).to_string(index=False))

    logger.info("=" * 55)
    logger.info("PERFORMANCE POR VENTANA TEMPORAL")
    logger.info("=" * 55)

    pred_date_col = cfg.get("deploy", {}).get("prediction_date_col", "fecha_prediccion")
    if pred_date_col in production_raw.columns:
        dates = pd.to_datetime(production_raw[pred_date_col], errors="coerce")
        production_raw["window"] = dates.dt.to_period("M").astype(str)
    else:
        production_raw["window"] = pd.qcut(
            np.arange(len(production_raw)), q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        ).astype(str)

    perf_df = performance_by_window(production_raw, threshold, score_col=score_col, actual_col=actual_col)
    logger.info("\n%s", perf_df.to_string(index=False))

    drift_path = Path(paths["drift_report_file"])
    drift_path.parent.mkdir(parents=True, exist_ok=True)
    psi_df.to_csv(drift_path, index=False)
    logger.info("drift_report.csv guardado: %s", drift_path)

    logger.info("Generando graficos de monitoreo...")

    plot_psi_heatmap(
        psi_df,
        save_path=reports_dir / "monitor_psi_heatmap.png",
        psi_thresholds=psi_thresholds,
    )
    plot_score_drift(
        ref_scores, prod_scores, model_name, psi_score_output,
        save_path=reports_dir / "monitor_score_drift.png",
        event_title=event_meta["event_title"],
    )
    plot_performance_over_time(
        perf_df,
        save_path=reports_dir / "monitor_performance.png",
        event_title=event_meta["event_title"],
    )

    top_drift = psi_df[psi_df["status"].isin(["Drift", "Monitorear"])]["feature"].tolist()
    if top_drift:
        plot_feature_drift_detail(
            reference_df[ref_feature_cols],
            production_raw[common_cols],
            top_drift_features=top_drift[:6],
            save_path=reports_dir / "monitor_feature_drift.png",
        )

    summary = {
        "use_case":       args.use_case,
        "event_col":      event_col,
        "model_name":     model_name,
        "psi_score":      psi_score_output,
        "psi_threshold":  psi_threshold,
        "n_features_drift":   n_drift,
        "n_features_monitor": n_monitor,
        "n_features_stable":  n_stable,
        "n_reference":    len(reference_df),
        "n_production":   len(production_raw),
    }

    html = build_monitoring_html(
        model_name=model_name,
        psi_df=psi_df.head(30),
        psi_score=psi_score_output,
        perf_df=perf_df,
        n_ref=len(reference_df),
        n_prod=len(production_raw),
        psi_threshold=psi_threshold,
        event_title=event_meta["event_title"],
    )
    html_path = reports_dir / "monitoring_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Reporte HTML guardado: %s", html_path)

    monitor_json = {
        **summary,
        "psi_by_feature":      psi_df.to_dict(orient="records"),
        "performance_windows": perf_df.to_dict(orient="records"),
    }
    json_path = reports_dir / "monitoring_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(monitor_json, f, indent=2)
    logger.info("Reporte JSON guardado: %s", json_path)

    logger.info("=" * 55)
    logger.info("MONITOREO COMPLETADO")
    logger.info("=" * 55)
    logger.info("use_case        : %s", args.use_case)
    logger.info("Modelo          : %s", model_name)
    logger.info("PSI output      : %.4f  (%s)",
                psi_score_output,
                "ALERTA" if psi_score_output > psi_threshold else "OK")
    logger.info("Features Drift  : %d / %d", n_drift, len(psi_df))
    logger.info("Reportes en     : %s", reports_dir)
    logger.info("  drift_report.csv")
    logger.info("  monitoring_report.html")
    logger.info("  monitoring_report.json")
    logger.info("  monitor_psi_heatmap.png")
    logger.info("  monitor_score_drift.png")
    logger.info("  monitor_performance.png")
    if top_drift:
        logger.info("  monitor_feature_drift.png")
    logger.info("=" * 55)

# ──────────────────────────────────────────────────────────