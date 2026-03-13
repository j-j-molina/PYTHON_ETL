"""
model_evaluation.py
===================
Genera el proceso de evaluación del modelo desplegado.

Crea una "pestaña de métricas" (reporte HTML interactivo + JSON)
que permite conocer el desempeño del modelo en producción.

Fuentes de datos:
  1. artifacts/best_model.joblib      → modelo desplegado
  2. artifacts/best_model_meta.json   → metadatos (threshold, nombre, etc.)
  3. reports/metrics_latest.json      → métricas de entrenamiento y CV
  4. ft_engineering.build_features()  → X_test / y_test actualizados

Outputs:
  reports/evaluation_report.html  → dashboard HTML con todas las métricas
  reports/evaluation_report.json  → métricas estructuradas (para monitoring)
  reports/eval_calibration.png    → curva de calibración (reliability diagram)
  reports/eval_score_distribution.png → distribución de scores por clase
  reports/eval_threshold_analysis.png → métricas vs umbral

Uso:
  python model_evaluation.py
"""

from __future__ import annotations

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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


# ──────────────────────────────────────────────────────────
#  Métricas por decil
# ──────────────────────────────────────────────────────────

def decile_analysis(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """
    Análisis por decil de score: tasa de mora real por decil de riesgo predicho.
    Permite validar que el modelo ordena correctamente el riesgo.
    Un buen modelo → tasa de mora monotonamente decreciente al bajar el score.
    """
    df = pd.DataFrame({"mora": y_true, "score": y_proba})
    df["decil"] = pd.qcut(df["score"], q=10, labels=False, duplicates="drop")
    df["decil"] = df["decil"] + 1  # 1 = menor score, 10 = mayor score

    summary = (
        df.groupby("decil")
        .agg(
            n=("mora", "count"),
            n_mora=("mora", "sum"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            score_mean=("score", "mean"),
        )
        .assign(tasa_mora=lambda x: x["n_mora"] / x["n"])
        .reset_index()
    )
    summary["tasa_mora_pct"] = (summary["tasa_mora"] * 100).round(2)
    return summary


# ──────────────────────────────────────────────────────────
#  Gráficos de evaluación
# ──────────────────────────────────────────────────────────

def plot_score_distribution(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str,
    save_path: Path,
) -> None:
    """
    Distribución de scores (probabilidades) separados por clase real.
    Permite visualizar qué tan bien el modelo separa mora vs al día.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    scores_ok   = y_proba[y_true == 0]
    scores_mora = y_proba[y_true == 1]

    ax.hist(scores_ok,   bins=30, alpha=0.6, color="#378ADD",
            label=f"Al día  (n={len(scores_ok)})",   density=True)
    ax.hist(scores_mora, bins=30, alpha=0.7, color="#E24B4A",
            label=f"Mora    (n={len(scores_mora)})", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel("Score de mora (probabilidad predicha)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de Scores por Clase — {model_name}",
                 fontweight="bold", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Score distribution guardada: %s", save_path)


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    """
    Reliability diagram (curva de calibración).
    Compara la probabilidad predicha vs la tasa de mora real observada.
    Un modelo bien calibrado → puntos cercanos a la diagonal.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba,
                                             n_bins=n_bins, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Reliability diagram
    ax = axes[0]
    ax.plot(prob_pred, prob_true, "o-", color="#1D9E75", linewidth=2,
            label=model_name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha (media del bin)")
    ax.set_ylabel("Fracción de positivos reales")
    ax.set_title("Curva de Calibración (Reliability Diagram)",
                 fontweight="bold", fontsize=11)
    ax.legend()
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    # Histograma de scores
    ax = axes[1]
    ax.hist(y_proba, bins=30, color="#378ADD", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Score de mora")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución Global de Scores", fontweight="bold", fontsize=11)

    plt.suptitle(f"Calibración del Modelo — {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Calibration guardada: %s", save_path)


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    current_threshold: float,
    model_name: str,
    save_path: Path,
) -> None:
    """
    Métricas clave (precision, recall, F1) en función del umbral.
    Permite al equipo de negocio entender el trade-off y ajustar
    el threshold según el apetito de riesgo de la empresa.
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_arr = np.where(
            (precision_arr + recall_arr) > 0,
            2 * precision_arr * recall_arr / (precision_arr + recall_arr),
            0.0,
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(thresholds, precision_arr[:-1], color="#1D9E75",
            linewidth=2, label="Precision mora")
    ax.plot(thresholds, recall_arr[:-1],    color="#E24B4A",
            linewidth=2, label="Recall mora")
    ax.plot(thresholds, f1_arr[:-1],        color="#378ADD",
            linewidth=2, label="F1 mora")
    ax.axvline(current_threshold, color="black", linestyle="--",
               linewidth=1.5, label=f"Threshold actual = {current_threshold:.3f}")

    # Anotar el punto actual
    idx = np.argmin(np.abs(thresholds - current_threshold))
    ax.annotate(
        f"P={precision_arr[idx]:.2f}\nR={recall_arr[idx]:.2f}\nF1={f1_arr[idx]:.2f}",
        xy=(current_threshold, f1_arr[idx]),
        xytext=(current_threshold + 0.05, f1_arr[idx] + 0.1),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Score")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_title(f"Precision / Recall / F1 vs Umbral — {model_name}",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Threshold analysis guardada: %s", save_path)


def plot_decile_chart(
    decile_df: pd.DataFrame,
    model_name: str,
    save_path: Path,
) -> None:
    """
    Tasa de mora real por decil de score.
    Es el gráfico que el negocio entiende mejor:
    'Los clientes del decil 10 tienen X% de mora real.'
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Tasa de mora por decil
    ax = axes[0]
    bars = ax.bar(decile_df["decil"], decile_df["tasa_mora_pct"],
                  color="#E24B4A", alpha=0.85, edgecolor="white")
    ax.axhline(decile_df["tasa_mora_pct"].mean(), color="black",
               linestyle="--", linewidth=1, label="Promedio global")
    for bar, val in zip(bars, decile_df["tasa_mora_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Decil de score (1=menor riesgo, 10=mayor riesgo)")
    ax.set_ylabel("Tasa de mora real (%)")
    ax.set_title("Tasa de Mora por Decil de Score",
                 fontweight="bold", fontsize=11)
    ax.legend(); ax.set_xticks(decile_df["decil"])

    # Volumen por decil
    ax = axes[1]
    ax.bar(decile_df["decil"], decile_df["n"],
           color="#378ADD", alpha=0.85, edgecolor="white")
    ax.bar(decile_df["decil"], decile_df["n_mora"],
           color="#E24B4A", alpha=0.85, edgecolor="white",
           label="Mora real")
    ax.set_xlabel("Decil de score")
    ax.set_ylabel("Número de clientes")
    ax.set_title("Volumen por Decil", fontweight="bold", fontsize=11)
    ax.legend(); ax.set_xticks(decile_df["decil"])

    plt.suptitle(f"Análisis por Decil — {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Decile chart guardada: %s", save_path)


def plot_roc_pr_eval(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str,
    baseline_roc: float,
    save_path: Path,
) -> None:
    """Curvas ROC y PR del modelo desplegado vs baseline heurístico."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    idx = np.argmin(np.abs(y_proba[np.argsort(y_proba)] - threshold))

    axes[0].plot(fpr, tpr, color="#1D9E75", linewidth=2,
                 label=f"{model_name}  (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1,
                 label=f"Random (AUC=0.500)")
    if baseline_roc > 0:
        axes[0].axhline(baseline_roc, color="#BA7517", linestyle=":",
                        linewidth=1.5, label=f"Baseline heurístico ({baseline_roc:.3f})")
    axes[0].set_xlabel("Tasa Falsos Positivos")
    axes[0].set_ylabel("Tasa Verdaderos Positivos")
    axes[0].set_title("Curva ROC", fontweight="bold", fontsize=12)
    axes[0].legend(fontsize=9)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[1].plot(rec, prec, color="#1D9E75", linewidth=2,
                 label=f"{model_name}  (AP={ap:.3f})")
    axes[1].axhline(y_true.mean(), color="k", linestyle="--", linewidth=1,
                    label=f"Random (AP={y_true.mean():.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Curva Precision-Recall", fontweight="bold", fontsize=12)
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.suptitle(f"Evaluación del Modelo Desplegado — {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("ROC/PR eval guardada: %s", save_path)


# ──────────────────────────────────────────────────────────
#  Reporte HTML
# ──────────────────────────────────────────────────────────

def build_html_report(
    model_name:    str,
    threshold:     float,
    test_metrics:  dict,
    cv_metrics:    dict,
    decile_df:     pd.DataFrame,
    train_metrics: dict,
    baseline_roc:  float,
    meta:          dict,
) -> str:
    """
    Genera el HTML del dashboard de evaluación.
    Replica el concepto de 'pestaña de métricas' del proyecto.
    """

    def metric_card(label, value, color="#1D9E75", fmt=".4f"):
        return f"""
        <div style="background:#f9f9f9;border-left:4px solid {color};
                    padding:12px 18px;border-radius:6px;min-width:140px;">
          <div style="font-size:11px;color:#666;text-transform:uppercase;
                      letter-spacing:.5px;">{label}</div>
          <div style="font-size:26px;font-weight:700;color:#222;">
            {format(value, fmt) if isinstance(value, float) else value}
          </div>
        </div>"""

    def section(title, content):
        return f"""
        <div style="margin-bottom:32px;">
          <h2 style="border-bottom:2px solid #eee;padding-bottom:8px;
                     color:#333;">{title}</h2>
          {content}
        </div>"""

    # Tabla de deciles
    decile_rows = ""
    for _, row in decile_df.iterrows():
        color = "#fde8e8" if row["tasa_mora_pct"] > decile_df["tasa_mora_pct"].mean() * 1.5 else "white"
        decile_rows += f"""
        <tr style="background:{color};">
          <td style="text-align:center">{int(row['decil'])}</td>
          <td style="text-align:right">{int(row['n'])}</td>
          <td style="text-align:right">{int(row['n_mora'])}</td>
          <td style="text-align:right">{row['tasa_mora_pct']:.2f}%</td>
          <td style="text-align:right">{row['score_min']:.3f}</td>
          <td style="text-align:right">{row['score_max']:.3f}</td>
          <td style="text-align:right">{row['score_mean']:.3f}</td>
        </tr>"""

    # Pre-compute cv values to avoid nested f-string dict-lookup issues
    _cv_roc_mean  = cv_metrics.get("roc_auc",   {}).get("mean", 0)
    _cv_rec_mean  = cv_metrics.get("recall",    {}).get("mean", 0)
    _cv_rec_std   = cv_metrics.get("recall",    {}).get("std",  0)
    _cv_f1_mean   = cv_metrics.get("f1",        {}).get("mean", 0)
    _cv_prec_mean = cv_metrics.get("precision", {}).get("std",  0)
    _cv_acc_mean  = cv_metrics.get("accuracy",  {}).get("mean", 0)
    _cv_rec_pct   = f"{_cv_rec_mean:.1%}"
    _cv_rec_std_s = f"{_cv_rec_std:.3f}"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Evaluación del Modelo — {model_name}</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:0;padding:24px 40px;background:#fff;color:#222;}}
  h1   {{color:#1a1a2e;font-size:28px;margin-bottom:4px;}}
  .sub {{color:#666;font-size:14px;margin-bottom:32px;}}
  .cards {{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px;}}
  table {{border-collapse:collapse;width:100%;font-size:13px;}}
  th    {{background:#1D9E75;color:white;padding:8px 12px;text-align:left;}}
  td    {{padding:7px 12px;border-bottom:1px solid #eee;}}
  img   {{max-width:100%;border-radius:8px;margin-top:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);}}
  .warn {{background:#fff8e1;border-left:4px solid #FFC107;
          padding:10px 16px;border-radius:4px;font-size:13px;margin-top:8px;}}
  .ok   {{background:#e8f5e9;border-left:4px solid #1D9E75;
          padding:10px 16px;border-radius:4px;font-size:13px;margin-top:8px;}}
</style>
</head>
<body>

<h1>📊 Evaluación del Modelo Desplegado</h1>
<div class="sub">
  Modelo: <strong>{model_name}</strong> &nbsp;|&nbsp;
  Threshold: <strong>{threshold:.4f}</strong> &nbsp;|&nbsp;
  Train cutoff: <strong>{meta.get('split_cutoff','2025-09')}</strong> &nbsp;|&nbsp;
  Test: oct-2025 en adelante &nbsp;|&nbsp;
  Features: <strong>{meta.get('feature_count', '-')}</strong>
</div>

{section("1. Métricas de Performance — Test Set", f'''
  <div class="cards">
    {metric_card("ROC-AUC", test_metrics["roc_auc"], "#1D9E75")}
    {metric_card("PR-AUC",  test_metrics["pr_auc"],  "#378ADD")}
    {metric_card("Recall mora",    test_metrics["recall_mora"],    "#E24B4A")}
    {metric_card("Precision mora", test_metrics["precision_mora"], "#BA7517")}
    {metric_card("F1 mora",        test_metrics["f1_mora"],        "#7F77DD")}
    {metric_card("Accuracy",       test_metrics["accuracy"],       "#555")}
  </div>
  <div class="cards">
    {metric_card("TP (moras detectadas)", test_metrics["tp"], "#E24B4A", "d")}
    {metric_card("FN (moras perdidas)",   test_metrics["fn"], "#E24B4A", "d")}
    {metric_card("FP (falsas alarmas)",   test_metrics["fp"], "#378ADD", "d")}
    {metric_card("TN (correctos al día)", test_metrics["tn"], "#1D9E75", "d")}
    {metric_card("Support mora (test)",   test_metrics["support_mora"], "#666", "d")}
  </div>
  <div class="warn">
    ⚠️ El test set contiene solo {test_metrics['support_mora']} moras (créditos oct-2025 en adelante,
    aún sin tiempo de madurar). El indicador de desempeño más confiable es el CV sobre train.
  </div>
''')}

{section("2. Métricas de Consistencia — Cross-Validation StratifiedKFold(5)", f'''
  <div class="cards">
    {metric_card("ROC-AUC CV", _cv_roc_mean, "#1D9E75")}
    {metric_card("Recall CV",  _cv_rec_mean, "#E24B4A")}
    {metric_card("F1 CV",      _cv_f1_mean, "#7F77DD")}
    {metric_card("Std recall CV", _cv_rec_std, "#999")}
  </div>
  <div class="ok">
    ✅ CV recall medio de {_cv_rec_mean:.1%} con
    std={_cv_rec_std:.3f} sobre train set.
    Este es el estimador más robusto del desempeño real del modelo.
  </div>
''')}

{section("3. Comparación vs Baseline Heurístico", f'''
  <div class="cards">
    {metric_card("Baseline ROC-AUC", baseline_roc, "#BA7517")}
    {metric_card("Modelo ROC-AUC",   test_metrics["roc_auc"], "#1D9E75")}
    {metric_card("Mejora absoluta",  test_metrics["roc_auc"] - baseline_roc, "#378ADD")}
  </div>
  <div class="{"ok" if test_metrics["roc_auc"] > baseline_roc else "warn"}">
    {"✅ El modelo ML supera el baseline heurístico en ROC-AUC." if test_metrics["roc_auc"] > baseline_roc else "⚠️ El modelo no supera el baseline en ROC-AUC."}
  </div>
''')}

{section("4. Análisis por Decil", f'''
  <p style="font-size:13px;color:#555;">
    Decil 10 = clientes con mayor score de mora predicho.
    Un buen modelo → tasa de mora real creciente con el decil.
  </p>
  <table>
    <tr>
      <th>Decil</th><th>N clientes</th><th>N mora</th>
      <th>Tasa mora real</th><th>Score mín</th>
      <th>Score máx</th><th>Score medio</th>
    </tr>
    {decile_rows}
  </table>
''')}

{section("5. Gráficos de Evaluación", '''
  <p style="font-size:13px;color:#555;">Ver archivos PNG en reports/</p>
  <ul style="font-size:13px;color:#555;">
    <li>eval_roc_pr.png — Curvas ROC y PR vs baseline</li>
    <li>eval_score_distribution.png — Distribución de scores por clase</li>
    <li>eval_calibration.png — Reliability diagram</li>
    <li>eval_threshold_analysis.png — Trade-off precision/recall por umbral</li>
    <li>eval_decile_chart.png — Tasa de mora real por decil</li>
  </ul>
''')}

{section("6. Metadatos del Modelo", f'''
  <table>
    <tr><th>Parámetro</th><th>Valor</th></tr>
    <tr><td>Modelo</td><td>{model_name}</td></tr>
    <tr><td>Threshold</td><td>{threshold:.4f}</td></tr>
    <tr><td>Threshold strategy</td><td>{meta.get("threshold_strategy","f1")}</td></tr>
    <tr><td>Score de selección</td><td>{meta.get("score","–")}</td></tr>
    <tr><td>Gap AUC (train–test)</td><td>{meta.get("gap_auc","–")}</td></tr>
    <tr><td>Train size</td><td>{meta.get("train_size","–"):,}</td></tr>
    <tr><td>Test size</td><td>{meta.get("test_size","–"):,}</td></tr>
    <tr><td>Tasa mora train</td><td>{meta.get("train_mora_rate",0):.2%}</td></tr>
    <tr><td>Tasa mora test</td><td>{meta.get("test_mora_rate",0):.2%}</td></tr>
    <tr><td>Features</td><td>{meta.get("feature_count","–")}</td></tr>
  </table>
''')}

<hr style="border:none;border-top:1px solid #eee;margin:32px 0 16px;">
<div style="font-size:11px;color:#aaa;">
  Generado por model_evaluation.py — CDP Entregable 3
</div>
</body>
</html>"""
    return html


# ──────────────────────────────────────────────────────────
#  Ejecución principal
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from ft_engineering import build_features, load_config

    # ── Config y directorios ──────────────────────────────
    cfg, repo_root = load_config()
    artifacts_dir  = repo_root / cfg["paths"]["artifacts_dir"]
    reports_dir    = repo_root / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar modelo desplegado ──────────────────────────
    model_path = repo_root / cfg["paths"]["model_file"]
    meta_path  = repo_root / cfg["paths"]["model_meta_file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. "
            "Ejecuta model_training.py primero."
        )

    model = joblib.load(model_path)
    logger.info("Modelo cargado: %s", model_path)

    with open(meta_path) as f:
        meta = json.load(f)

    model_name = meta["model_name"]
    threshold  = meta["threshold"]
    baseline_roc = meta.get("baseline_roc", 0.0)
    cv_metrics   = meta.get("cv_metrics", {})

    logger.info("Modelo: %s | Threshold: %.4f", model_name, threshold)

    # ── Features ─────────────────────────────────────────
    logger.info("Cargando features...")
    X_train, X_test, y_train, y_test, _, _ = build_features(return_dataframe=True)

    y_proba_test  = model.predict_proba(X_test)[:, 1]
    y_pred_test   = (y_proba_test >= threshold).astype(int)
    y_proba_train = model.predict_proba(X_train)[:, 1]

    # ── Métricas test ─────────────────────────────────────
    roc_auc = roc_auc_score(y_test, y_proba_test)
    pr_auc  = average_precision_score(y_test, y_proba_test)
    cm      = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    report  = classification_report(
        y_test, y_pred_test,
        target_names=["Al dia", "Mora"],
        output_dict=True, zero_division=0,
    )

    test_metrics = {
        "model":           model_name,
        "split":           "Test",
        "threshold":       round(threshold, 4),
        "roc_auc":         round(roc_auc, 4),
        "pr_auc":          round(pr_auc, 4),
        "accuracy":        round(report["accuracy"], 4),
        "precision_mora":  round(report["Mora"]["precision"], 4),
        "recall_mora":     round(report["Mora"]["recall"], 4),
        "f1_mora":         round(report["Mora"]["f1-score"], 4),
        "support_mora":    int(report["Mora"]["support"]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    # Métricas train (para comparación)
    metrics_path = repo_root / cfg["paths"]["metrics_file"]
    train_metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_m = json.load(f)
        train_metrics = next(
            (m for m in all_m if m.get("model") == model_name
             and m.get("split") == "Train"), {}
        )

    logger.info("=" * 55)
    logger.info("EVALUACION DEL MODELO DESPLEGADO — %s", model_name)
    logger.info("=" * 55)
    cv_roc_mean = cv_metrics.get("roc_auc", {}).get("mean", 0)
    cv_rec_mean = cv_metrics.get("recall",  {}).get("mean", 0)
    logger.info("ROC-AUC test : %.4f  (CV: %.4f)", roc_auc, cv_roc_mean)
    logger.info("PR-AUC  test : %.4f", pr_auc)
    logger.info("Recall  mora : %.4f  (CV: %.4f)", test_metrics["recall_mora"], cv_rec_mean)
    logger.info("CM  TN=%d  FP=%d  FN=%d  TP=%d", tn, fp, fn, tp)

    # ── Análisis por decil ────────────────────────────────
    decile_df = decile_analysis(y_test, y_proba_test)
    logger.info("\nAnalisis por decil (test):\n%s", decile_df.to_string(index=False))

    # ── Gráficos ──────────────────────────────────────────
    logger.info("Generando graficos de evaluacion...")

    plot_roc_pr_eval(
        y_test, y_proba_test, threshold, model_name, baseline_roc,
        save_path=reports_dir / "eval_roc_pr.png",
    )
    plot_score_distribution(
        y_test, y_proba_test, threshold, model_name,
        save_path=reports_dir / "eval_score_distribution.png",
    )
    plot_calibration(
        y_test, y_proba_test, model_name,
        save_path=reports_dir / "eval_calibration.png",
    )
    plot_threshold_analysis(
        y_test, y_proba_test, threshold, model_name,
        save_path=reports_dir / "eval_threshold_analysis.png",
    )
    plot_decile_chart(
        decile_df, model_name,
        save_path=reports_dir / "eval_decile_chart.png",
    )

    # ── Reporte HTML ──────────────────────────────────────
    meta["split_cutoff"] = cfg["split"]["train_cutoff"]
    html = build_html_report(
        model_name=model_name,
        threshold=threshold,
        test_metrics=test_metrics,
        cv_metrics=cv_metrics,
        decile_df=decile_df,
        train_metrics=train_metrics,
        baseline_roc=baseline_roc,
        meta=meta,
    )
    html_path = reports_dir / "evaluation_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Reporte HTML guardado: %s", html_path)

    # ── Reporte JSON ──────────────────────────────────────
    eval_report = {
        "model_name":   model_name,
        "threshold":    threshold,
        "test_metrics": test_metrics,
        "cv_metrics":   cv_metrics,
        "decile_analysis": decile_df.to_dict(orient="records"),
        "baseline_roc": baseline_roc,
        "gap_auc":      meta.get("gap_auc"),
    }
    json_path = reports_dir / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Reporte JSON guardado: %s", json_path)

    # ── Resumen final ─────────────────────────────────────
    logger.info("=" * 55)
    logger.info("EVALUACION COMPLETADA")
    logger.info("=" * 55)
    logger.info("Reportes en: %s", reports_dir)
    logger.info("  evaluation_report.html")
    logger.info("  evaluation_report.json")
    logger.info("  eval_roc_pr.png")
    logger.info("  eval_score_distribution.png")
    logger.info("  eval_calibration.png")
    logger.info("  eval_threshold_analysis.png")
    logger.info("  eval_decile_chart.png")
    logger.info("=" * 55)