"""
model_training.py
=================
Entrenamiento, evaluación y selección del mejor modelo predictivo del evento objetivo.

Funciones principales requeridas por el proyecto:
  - summarize_classification(): métricas completas de clasificación
  - build_model():              entrena un estimador y retorna el modelo ajustado

Modelos comparados:
  1. Logistic Regression    (baseline lineal, interpretable)
  2. Decision Tree          (baseline no lineal, visualizable)
  3. Random Forest          (ensemble bagging)
  4. Gradient Boosting      (ensemble boosting)

Correcciones respecto a la primera versión:
  ─────────────────────────────────────────────────────────────
  PROBLEMA 1 — Threshold fijo en 0.50
    Con mora=3.5% en test, predict() con threshold=0.50 siempre
    devuelve 0 para RF y GB. Solución: find_optimal_threshold()
    busca el umbral óptimo sobre el train set y lo aplica en test.

  PROBLEMA 2 — GradientBoosting sin manejo de desbalance
    GB no acepta class_weight. Solución: sample_weight calculado
    como la razón de clases (pos_weight ≈ 19x) se pasa a fit().

  PROBLEMA 3 — Evaluación sobre 29 moras en test
    Soporte estadístico insuficiente para métricas estables.
    Solución: StratifiedKFold(5) sobre train da métricas más
    robustas. El test sigue siendo el juez final, pero el CV
    es el criterio principal de selección.
  ─────────────────────────────────────────────────────────────

Criterio de selección (config.training.selection_weights):
  Performance  → ROC-AUC (40%) y PR-AUC (30%)  sobre CV-train
  Consistency  → gap roc_auc train-test (10%)
  Recall del evento → capturar positivos (20%)

Outputs:
  artifacts/<use_case>/best_model.joblib
  artifacts/<use_case>/best_model_meta.json
  reports/metrics_latest.json
  reports/curvas_roc_pr.png
  reports/comparacion_metricas.png
  reports/matrices_confusion.png
  reports/feature_importance.png
  reports/learning_curve_best.png

Uso:
  python model_training.py
  from model_training import build_model, summarize_classification
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


def resolve_runtime_paths(cfg_global: dict, cfg: dict) -> dict:
    """
    Resuelve rutas runtime con awareness por use_case.

    Si el use_case sobreescribe artifacts_dir / reports_dir pero NO redefine
    model_file, metrics_file, etc., se derivan automáticamente dentro de esos
    directorios para evitar colisiones entre casos de uso.
    """
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

# ──────────────────────────────────────────────────────────
#  Threshold calibration

def find_optimal_threshold(
    y_true:        np.ndarray,
    y_proba:       np.ndarray,
    strategy:      str   = "f1",
    min_precision: float = 0.05,
) -> float:
    """
    Encuentra el umbral de decisión óptimo sobre el conjunto de TRAIN.

    Por qué es necesario:
      Con una clase positiva minoritaria en train, el threshold default de 0.50 hace que
      ningún modelo prediga el evento, ya que la probabilidad estimada para
      la clase minoritaria rara vez supera ese umbral.

    Estrategias (configurables en config.training.threshold_strategy):
      'f1'     → maximiza F1 del evento (equilibrio precision/recall).
      'recall' → maximiza recall con precision >= min_precision.
      'prior'  → usa la tasa del evento en train como umbral directo.

    Args:
        y_true:        etiquetas reales del train.
        y_proba:       probabilidades de la clase positiva del train.
        strategy:      'f1' | 'recall' | 'prior'.
        min_precision: precisión mínima aceptable (solo para 'recall').

    Returns:
        float: umbral óptimo en [0, 1].
    """
    if strategy == "prior":
        threshold = float(y_true.mean())
        logger.info("Threshold (prior): %.4f", threshold)
        return threshold

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)

    if strategy == "f1":
        # F1 = 2*P*R / (P+R); ignorar divisiones por cero
        with np.errstate(invalid="ignore", divide="ignore"):
            f1_arr = np.where(
                (precision_arr + recall_arr) > 0,
                2 * precision_arr * recall_arr / (precision_arr + recall_arr),
                0.0,
            )
        best_idx  = int(np.argmax(f1_arr[:-1]))
        threshold = float(thresholds[best_idx])
        logger.info(
            "Threshold (f1): %.4f  → P=%.3f R=%.3f F1=%.3f",
            threshold,
            precision_arr[best_idx], recall_arr[best_idx], f1_arr[best_idx],
        )

    elif strategy == "recall":
        valid = precision_arr[:-1] >= min_precision
        if valid.any():
            recall_valid = np.where(valid, recall_arr[:-1], 0.0)
            best_idx     = int(np.argmax(recall_valid))
            threshold    = float(thresholds[best_idx])
        else:
            threshold = float(y_true.mean())
        logger.info(
            "Threshold (recall, min_p=%.2f): %.4f  → P=%.3f R=%.3f",
            min_precision, threshold,
            precision_arr[best_idx] if valid.any() else 0.0,
            recall_arr[best_idx] if valid.any() else 0.0,
        )
    else:
        raise ValueError(f"Estrategia desconocida: '{strategy}'. Usa 'f1', 'recall' o 'prior'.")

    # Clip de seguridad: evitar umbrales extremos
    threshold = float(np.clip(threshold, 0.01, 0.95))
    return threshold

# ──────────────────────────────────────────────────────────
#  summarize_classification


def summarize_classification(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_proba:    np.ndarray,
    model_name: str,
    split_name: str  = "Test",
    threshold:  float = 0.50,
    event_label: str = "event",
    neg_label:  str  = "Negativo",
    verbose:    bool  = True,
) -> dict:
    """
    Genera un resumen completo de métricas de clasificación.
    """
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc  = average_precision_score(y_true, y_proba)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y_true, y_pred,
        target_names=[neg_label, event_label],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model":            model_name,
        "split":            split_name,
        "threshold":        round(threshold, 4),
        "roc_auc":          round(roc_auc, 4),
        "pr_auc":           round(pr_auc, 4),
        "accuracy":         round(report["accuracy"], 4),
        "precision_event":  round(report[event_label]["precision"], 4),
        "recall_event":     round(report[event_label]["recall"], 4),
        "f1_event":         round(report[event_label]["f1-score"], 4),
        "support_event":    int(report[event_label]["support"]),
        "event_label":      event_label,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }

    if verbose:
        logger.info("-" * 60)
        logger.info("%-25s  [%s]  threshold=%.3f", model_name, split_name, threshold)
        logger.info("-" * 60)
        logger.info("ROC-AUC : %.4f  |  PR-AUC: %.4f", roc_auc, pr_auc)
        logger.info("Recall %s : %.4f  |  Precision %s: %.4f",
                    event_label,
                    metrics["recall_event"],
                    event_label,
                    metrics["precision_event"])
        logger.info("F1 %s : %.4f  |  Accuracy: %.4f",
                    event_label,
                    metrics["f1_event"], metrics["accuracy"])
        logger.info("CM  TN=%-5d FP=%-5d FN=%-5d TP=%-5d", tn, fp, fn, tp)

    return metrics

# ──────────────────────────────────────────────────────────
#  build_model

def build_model(
    name:              str,
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold_strategy:    str   = "f1",
    threshold_min_precision: float = 0.05,
    fit_params:        dict = None,
    event_label:       str  = "event",
    neg_label:         str  = "Negativo",
) -> tuple:
    """
    Entrena un estimador, calibra el threshold sobre train y evalúa en test.

    Args:
        name:                      nombre descriptivo del modelo.
        estimator:                 instancia sklearn sin ajustar.
        X_train, y_train:          datos de entrenamiento.
        X_test, y_test:            datos de evaluación.
        threshold_strategy:        estrategia de calibración ('f1'|'recall'|'prior').
        threshold_min_precision:   precisión mínima para estrategia 'recall'.
        fit_params:                kwargs adicionales para estimator.fit() (ej. sample_weight).

    Returns:
        (fitted_estimator, metrics_train, metrics_test, fit_time, threshold)
    """
    logger.info("Entrenando: %s ...", name)
    t0 = time.time()

    if fit_params:
        estimator.fit(X_train, y_train, **fit_params)
    else:
        estimator.fit(X_train, y_train)

    fit_time = round(time.time() - t0, 2)
    logger.info("  Tiempo de entrenamiento: %.2fs", fit_time)

    # Probabilidades
    y_proba_train = estimator.predict_proba(X_train)[:, 1]
    y_proba_test  = estimator.predict_proba(X_test)[:, 1]

    # Threshold óptimo calibrado sobre TRAIN (no se toca el test)
    threshold = find_optimal_threshold(
        y_train, y_proba_train,
        strategy=threshold_strategy,
        min_precision=threshold_min_precision,
    )

    # Predicciones binarias con threshold calibrado
    y_pred_train = (y_proba_train >= threshold).astype(int)
    y_pred_test  = (y_proba_test  >= threshold).astype(int)

    metrics_train = summarize_classification(
        y_train, y_pred_train, y_proba_train, name, "Train", threshold,
        event_label=event_label, neg_label=neg_label,
    )
    metrics_test = summarize_classification(
        y_test, y_pred_test, y_proba_test, name, "Test", threshold,
        event_label=event_label, neg_label=neg_label,
    )

    metrics_train["fit_time"] = fit_time
    metrics_test["fit_time"]  = fit_time

    return estimator, metrics_train, metrics_test, fit_time, threshold

# ──────────────────────────────────────────────────────────
#  Cross-Validation sobre train  (Consistency)

def cross_validate_model(
    name:                    str,
    estimator,
    X_train,
    y_train,
    n_splits:                int   = 5,
    scoring:                 str   = "recall",
    threshold_strategy:      str   = "f1",
    threshold_min_precision: float = 0.05,
    fit_params:              dict  = None,
) -> dict:
    """
    StratifiedKFold CV con threshold calibrado por fold.

    Corrección respecto a la versión anterior:
      La versión anterior usaba cross_val_score() que internamente llama
      predict() con threshold=0.5 fijo. Esto era inconsistente con el
      threshold calibrado (ej. 0.64) usado en la evaluación final, lo que
      hacía que el recall CV reportado no reflejara el comportamiento real
      del modelo en producción.

      Ahora: en cada fold se calibra el threshold sobre el fold de train
      con la misma estrategia que build_model() (ej. 'f1'), y se aplica
      ese threshold sobre el fold de validación. Garantiza que las métricas
      de CV sean homogéneas con las métricas de test.

    Adicionalmente, fit_params (ej. sample_weight para GradientBoosting)
    se subsetea por fold para garantizar coherencia con el índice de train.

    Por qué StratifiedKFold:
      Con una clase positiva minoritaria, KFold puede generar folds sin positivos.
      Stratified garantiza la proporción en cada fold.

    Args:
        name:                    nombre del modelo.
        estimator:               instancia sklearn ya ajustada (se clona por fold).
        X_train:                 features de entrenamiento (array o DataFrame).
        y_train:                 target de entrenamiento.
        n_splits:                número de folds.
        scoring:                 métrica principal (informativo, se calculan todas).
        threshold_strategy:      estrategia de calibración ('f1'|'recall'|'prior').
        threshold_min_precision: precisión mínima para estrategia 'recall'.
        fit_params:              kwargs adicionales para fit() (ej. sample_weight).

    Returns:
        dict con mean, std y scores por fold para accuracy, f1, precision,
        recall y roc_auc — todos evaluados con el threshold calibrado por fold.
    """
    from sklearn.base import clone

    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    x_arr = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    y_arr = y_train if isinstance(y_train, np.ndarray) else np.array(y_train)

    fold_scores = {m: [] for m in ["accuracy", "f1", "precision", "recall", "roc_auc"]}

    for tr_idx, val_idx in skf.split(x_arr, y_arr):
        x_tr,  x_val = x_arr[tr_idx], x_arr[val_idx]
        y_tr,  y_val = y_arr[tr_idx], y_arr[val_idx]

        est = clone(estimator)
        if fit_params:
            fp = {
                k: v[tr_idx] if (hasattr(v, "__len__") and len(v) == len(y_arr)) else v
                for k, v in fit_params.items()
            }
            est.fit(x_tr, y_tr, **fp)
        else:
            est.fit(x_tr, y_tr)

        y_proba_tr  = est.predict_proba(x_tr)[:, 1]
        y_proba_val = est.predict_proba(x_val)[:, 1]

        # Calibrar threshold sobre el fold de train (idéntico a build_model)
        thr = find_optimal_threshold(
            y_tr, y_proba_tr,
            strategy=threshold_strategy,
            min_precision=threshold_min_precision,
        )
        y_pred_val = (y_proba_val >= thr).astype(int)

        fold_scores["roc_auc"].append(roc_auc_score(y_val, y_proba_val))
        fold_scores["recall"].append(recall_score(y_val, y_pred_val, zero_division=0))
        fold_scores["precision"].append(precision_score(y_val, y_pred_val, zero_division=0))
        fold_scores["f1"].append(f1_score(y_val, y_pred_val, zero_division=0))
        fold_scores["accuracy"].append(accuracy_score(y_val, y_pred_val))


    results = {}
    for metric, scores in fold_scores.items():
        results[metric] = {
            "mean":   round(float(np.mean(scores)), 4),
            "std":    round(float(np.std(scores)),  4),
            "scores": [round(float(s), 4) for s in scores],
        }

    logger.info("-" * 60)
    logger.info("CV StratifiedKFold(k=%d) — %s  scoring='%s'  [threshold calibrado por fold]",
                n_splits, name, scoring)
    logger.info("-" * 60)
    for metric, r in results.items():
        logger.info("  %-12s mean=%.3f  std=%.3f", metric, r["mean"], r["std"])

    return {"model": name, "cv_folds": n_splits, "metrics": results}

# ──────────────────────────────────────────────────────────
#  Gráficos comparativos

def plot_roc_pr_curves(models_data, X_test, y_test, save_path):
    colors = ["#1D9E75", "#378ADD", "#BA7517", "#E24B4A"]
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, md in enumerate(models_data):
        y_prob = md["estimator"].predict_proba(X_test)[:, 1]
        color  = colors[i % len(colors)]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0].plot(fpr, tpr, color=color, linewidth=2,
                     label=f"{md['name']}  (AUC={roc_auc_score(y_test, y_prob):.3f})")

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(rec, prec, color=color, linewidth=2,
                     label=f"{md['name']}  (AP={average_precision_score(y_test, y_prob):.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", linewidth=1,
                    label=f"Random (AP={y_test.mean():.3f})")

    for ax, title, xlabel, ylabel in [
        (axes[0], "Curva ROC", "Tasa Falsos Positivos", "Tasa Verdaderos Positivos"),
        (axes[1], "Curva Precision-Recall", "Recall", "Precision"),
    ]:
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="lower right" if ax == axes[0] else "upper right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.suptitle("Comparación de Modelos — Curvas ROC y PR",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Curvas ROC/PR guardadas: %s", save_path)


def plot_metrics_comparison(metrics_list, save_path, event_label="event"):
    """Barras comparativas de métricas en test set."""
    test_df = pd.DataFrame(
        [m for m in metrics_list if m["split"] == "Test"]
    ).set_index("model")

    metric_cols   = ["roc_auc", "pr_auc", "recall_event", "f1_event"]
    metric_labels = ["ROC-AUC", "PR-AUC", f"Recall {event_label}", f"F1 {event_label}"]
    colors        = ["#1D9E75", "#378ADD", "#BA7517", "#E24B4A"]

    _, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, col, label, color in zip(axes, metric_cols, metric_labels, colors):
        vals = test_df[col].sort_values(ascending=True)
        bars = ax.barh(vals.index, vals.values, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        for bar, val in zip(bars, vals.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

    plt.suptitle("Comparación de Métricas — Test Set (threshold calibrado)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Comparacion metricas guardada: %s", save_path)

def plot_cv_comparison(cv_results_list, save_path, event_label="event"):
    """
    Boxplot de CV recall por modelo — muestra consistency.
    Replica la visualización del profe extendida a todos los modelos.
    """
    data   = []
    labels = []
    for cv in cv_results_list:
        data.append(cv["metrics"]["recall"]["scores"])
        labels.append(cv["model"])

    _, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    colors = ["#1D9E75", "#378ADD", "#BA7517", "#E24B4A"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)

    ax.set_title(f"Recall {event_label} — CV StratifiedKFold(5) por modelo",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel(f"Recall {event_label}")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("CV comparison guardada: %s", save_path)

def plot_confusion_matrices(models_data, thresholds, X_test, y_test, save_path, event_label="event", neg_label="Negativo"):
    """Matrices de confusión normalizadas con threshold calibrado."""
    n    = len(models_data)
    _, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, md in zip(axes, models_data):
        thr    = thresholds.get(md["name"], 0.5)
        y_prob = md["estimator"].predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= thr).astype(int)
        cm     = confusion_matrix(y_test, y_pred, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=[neg_label, event_label],
                    yticklabels=[neg_label, event_label],
                    ax=ax, cbar=False)
        ax.set_title(f"{md['name']}\n(thr={thr:.3f})", fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")

    plt.suptitle("Matrices de Confusion Normalizadas — Test Set",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Matrices confusion guardadas: %s", save_path)

def plot_feature_importance(model, feature_names, model_name, save_path, top_n=20):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Importancia de Features — {model_name}"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = f"Coeficientes Absolutos — {model_name}"
    else:
        return

    indices = np.argsort(importances)[-top_n:]
    _, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh([feature_names[i] for i in indices], importances[indices],
            color="#1D9E75", alpha=0.85, edgecolor="white")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance guardada: %s", save_path)

def plot_learning_curve_best(estimator, model_name, X_train, y_train,
                              save_path, scoring="recall", lc_cfg: dict = None):
    """Learning curve del mejor modelo — Consistency y Scalability."""
    from sklearn.model_selection import ShuffleSplit
    lc_cfg   = lc_cfg or {}
    n_splits = int(lc_cfg.get("n_splits", 30))
    test_size = float(lc_cfg.get("test_size", 0.2))
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv, scoring=scoring, n_jobs=1, return_times=True, random_state=42,
    )

    _, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Learning curve
    ax = axes[0]
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-",
            color="#1D9E75", label="Training score")
    ax.plot(train_sizes, test_scores.mean(axis=1), "o-",
            color="#E24B4A", label="Cross-validation score")
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1),
                    alpha=0.2, color="#1D9E75")
    ax.fill_between(train_sizes,
                    test_scores.mean(1) - test_scores.std(1),
                    test_scores.mean(1) + test_scores.std(1),
                    alpha=0.2, color="#E24B4A")
    ax.set_title(f"Learning Curve for {model_name}", fontweight="bold")
    ax.set_xlabel("Training examples"); ax.set_ylabel(scoring.capitalize())
    ax.legend(); ax.set_ylim(0, 1)

    # Scalability
    ax = axes[1]
    ax.plot(train_sizes, fit_times.mean(axis=1), "o-", color="#378ADD")
    ax.fill_between(train_sizes,
                    fit_times.mean(1) - fit_times.std(1),
                    fit_times.mean(1) + fit_times.std(1),
                    alpha=0.2, color="#378ADD")
    ax.set_title("Scalability — Fit Time", fontweight="bold")
    ax.set_xlabel("Training examples"); ax.set_ylabel("Fit time (s)")

    plt.suptitle(f"{model_name} — Consistency & Scalability",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Learning curve guardada: %s", save_path)

def print_summary_table(all_metrics):
    cols = ["model", "split", "threshold", "roc_auc", "pr_auc",
            "recall_event", "precision_event", "f1_event", "accuracy", "fit_time"]
    df = pd.DataFrame(all_metrics)[cols].sort_values(
        ["split", "roc_auc"], ascending=[True, False]
    )
    logger.info("\n%s", df.to_string(index=False))
    return df

# ──────────────────────────────────────────────────────────
#  Selección del mejor modelo

def select_best_model(models_data, cv_results_list, all_metrics, weights):
    """
    Score ponderado usando métricas de CV.

      Performance  → ROC-AUC CV (40%) y PR-AUC test (30%)
      Recall del evento  → recall CV mean (20%)
      Consistency  → penaliza gap roc_auc train-test (10%)
    """
    cv_map   = {r["model"]: r["metrics"] for r in cv_results_list}
    test_map = {m["model"]: m for m in all_metrics if m["split"] == "Test"}
    train_map= {m["model"]: m for m in all_metrics if m["split"] == "Train"}

    w_roc    = weights.get("roc_auc", 0.40)
    w_pr     = weights.get("pr_auc",  0.30)
    w_recall = weights.get("recall",  0.20)
    w_gap    = weights.get("gap",     0.10)

    scored = []
    for md in models_data:
        name = md["name"]
        cv   = cv_map.get(name, {})
        te   = test_map.get(name, {})
        tr   = train_map.get(name, {})

        roc_cv     = cv.get("roc_auc", {}).get("mean", 0)
        recall_cv  = cv.get("recall",  {}).get("mean", 0)
        pr_test    = te.get("pr_auc",  0)
        gap        = abs(tr.get("roc_auc", 0) - te.get("roc_auc", 0))

        score = (w_roc * roc_cv + w_pr * pr_test
                 + w_recall * recall_cv - w_gap * gap)

        scored.append({**md, "score": round(score, 4), "gap_auc": round(gap, 4)})
        logger.info(
            "Score %-25s → %.4f  (ROC_cv=%.3f PR_te=%.3f Recall_cv=%.3f gap=%.3f)",
            name, score, roc_cv, pr_test, recall_cv, gap,
        )

    best = max(scored, key=lambda x: x["score"])
    logger.info("Mejor modelo: %s  (score=%.4f)", best["name"], best["score"])
    return best


def _default_model_specs() -> list[dict]:
    return [
        {"name": "Logistic Regression", "type": "logistic_regression",
         "params": {"class_weight": "balanced", "max_iter": 1000, "C": 0.1, "solver": "lbfgs", "random_state": 42}},
        {"name": "Decision Tree", "type": "decision_tree",
         "params": {"class_weight": "balanced", "max_depth": 5, "min_samples_leaf": 20, "ccp_alpha": 0.0, "random_state": 42}},
        {"name": "Random Forest", "type": "random_forest",
         "params": {"class_weight": "balanced", "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10, "ccp_alpha": 0.0, "n_jobs": -1, "random_state": 42}},
        {"name": "Gradient Boosting", "type": "gradient_boosting",
         "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "min_samples_leaf": 10, "max_features": None, "subsample": 0.8, "random_state": 42},
         "fit_strategy": "sample_weight_positive"},
    ]


def build_model_definitions(train_cfg: dict, y_train: np.ndarray) -> list[tuple[str, object, dict | None]]:
    """Construye el catálogo de modelos desde config.training.models."""
    specs = train_cfg.get("models") or _default_model_specs()
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    gb_sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    out = []
    for spec in specs:
        model_type   = spec["type"]
        params       = dict(spec.get("params", {}))
        fit_strategy = spec.get("fit_strategy")
        name         = spec.get("name", model_type.replace("_", " ").title())

        # Garantizar reproducibilidad si config no incluye random_state
        params.setdefault("random_state", 42)
        rs = params.pop("random_state")

        if model_type == "logistic_regression":
            estimator = LogisticRegression(random_state=rs, **params)
        elif model_type == "decision_tree":
            estimator = DecisionTreeClassifier(
                random_state=rs,
                ccp_alpha=params.pop("ccp_alpha", 0.0),
                **params,
            )
        elif model_type == "random_forest":
            estimator = RandomForestClassifier(
                random_state=rs,
                ccp_alpha=params.pop("ccp_alpha", 0.0),
                min_samples_leaf=params.pop("min_samples_leaf", 1),
                max_features=params.pop("max_features", "sqrt"),
                **params,
            )
        elif model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                random_state=rs,
                max_features=params.pop("max_features", None),
                learning_rate=params.pop("learning_rate", 0.1),
                **params,
            )
        elif model_type == "hist_gradient_boosting":
            estimator = HistGradientBoostingClassifier(
                random_state=rs,
                learning_rate=params.pop("learning_rate", 0.1),
                **params,
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado en config.training.models: {model_type}")

        fit_params = {"sample_weight": gb_sample_weight} if fit_strategy == "sample_weight_positive" else None
        out.append((name, estimator, fit_params))

    logger.info("Catálogo de modelos cargado desde config: %d modelos", len(out))
    return out

# ──────────────────────────────────────────────────────────
#  Ejecución principal


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from ft_engineering import load_features_from_cache, load_config, resolve_cfg

    parser = argparse.ArgumentParser(
        description="model_training — entrenamiento y selección del mejor modelo"
    )
    parser.add_argument(
        "--use-case", type=str, dest="use_case", required=True,
        help="Caso de uso definido en config.json > use_cases.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Ruta opcional a config.json.",
    )
    args = parser.parse_args()

    # ── Config y directorios ──────────────────────────────
    cfg_global, _ = load_config(Path(args.config) if args.config else None)
    cfg = resolve_cfg(cfg_global, args.use_case)
    paths = resolve_runtime_paths(cfg_global, cfg)

    artifacts_dir  = Path(paths["artifacts_dir"])
    reports_dir    = Path(paths["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg.get("training", {})
    cv_folds            = train_cfg.get("cv_folds", 5)
    cv_scoring          = train_cfg.get("cv_scoring", "recall")
    threshold_strategy  = train_cfg.get("threshold_strategy", "f1")
    threshold_min_prec  = train_cfg.get("threshold_min_precision", 0.05)
    sel_weights         = train_cfg.get(
        "selection_weights",
        {"roc_auc": 0.4, "pr_auc": 0.3, "recall": 0.2, "gap": 0.1},
    )
    event_label = cfg["target"]["event_col"]
    neg_label   = cfg["target"].get("negative_class_label", "Negativo")

    # ── Features ─────────────────────────────────────────
    safe_use_case = str(args.use_case).replace('\n', ' ').replace('\r', ' ')
    logger.info("Cargando features desde caché para use_case='%s'...", safe_use_case)
    X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base, _, _ = load_features_from_cache(
        use_case=args.use_case,
        config_path=Path(args.config) if args.config else None,
    )
    feature_names = list(X_train.columns)
    logger.info("X_train: %s | X_test: %s", X_train.shape, X_test.shape)
    logger.info("%s train: %.2f%% | test: %.2f%%",
                event_label,
                y_train.mean() * 100, y_test.mean() * 100)

    # ── Baseline heurístico ───────────────────────────────
    baseline_path = reports_dir / "heuristic_baseline.json"
    baseline_roc  = 0.0
    if baseline_path.exists():
        with open(baseline_path) as f:
            bl = json.load(f)
        baseline_roc = bl["test"]["roc_auc"]
        logger.info("Baseline heurístico — ROC-AUC test: %.4f", baseline_roc)
    # ── Catálogo de modelos desde config ───────────────────
    model_definitions = build_model_definitions(train_cfg, y_train)

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO DE MODELOS | threshold_strategy='%s'", threshold_strategy)
    logger.info("=" * 60)

    models_data: list[dict] = []
    all_metrics: list[dict] = []
    thresholds:  dict       = {}

    for name, estimator, fit_params in model_definitions:
        fitted, m_train, m_test, fit_time, thr = build_model(
            name, estimator, X_train, y_train, X_test, y_test,
            threshold_strategy=threshold_strategy,
            threshold_min_precision=threshold_min_prec,
            fit_params=fit_params,
            event_label=event_label,
            neg_label=neg_label,
        )
        models_data.append({"name": name, "estimator": fitted})
        all_metrics.extend([m_train, m_test])
        thresholds[name] = thr

    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION StratifiedKFold(k=%d)  scoring='%s'",
                cv_folds, cv_scoring)
    logger.info("=" * 60)

    cv_results_list = []
    for i, (name, _, fit_params) in enumerate(model_definitions):
        cv_r = cross_validate_model(
            name, models_data[i]["estimator"],
            X_train, y_train,
            n_splits=cv_folds,
            scoring=cv_scoring,
            threshold_strategy=threshold_strategy,
            threshold_min_precision=threshold_min_prec,
            fit_params=fit_params,
        )
        cv_results_list.append(cv_r)

    logger.info("\n%s", "=" * 60)
    logger.info("TABLA RESUMEN")
    logger.info("=" * 60)
    summary_df = print_summary_table(all_metrics)

    if baseline_roc > 0:
        logger.info("=" * 60)
        logger.info("Comparacion vs baseline heuristico (ROC-AUC=%.4f)", baseline_roc)
        for md in models_data:
            te = next(m for m in all_metrics
                      if m["model"] == md["name"] and m["split"] == "Test")
            logger.info("  %-25s ROC=%.4f  supera baseline: %s",
                        md["name"], te["roc_auc"],
                        "SI" if te["roc_auc"] > baseline_roc else "NO")

    logger.info("=" * 60)
    logger.info("SELECCION DEL MEJOR MODELO  (CV-driven)")
    logger.info("=" * 60)
    best = select_best_model(models_data, cv_results_list, all_metrics, sel_weights)

    logger.info("Generando graficos...")
    plot_roc_pr_curves(models_data, X_test, y_test,
                       save_path=reports_dir / "curvas_roc_pr.png")
    plot_metrics_comparison(all_metrics,
                            save_path=reports_dir / "comparacion_metricas.png",
                            event_label=event_label)
    plot_cv_comparison(cv_results_list,
                       save_path=reports_dir / "cv_recall_comparison.png",
                       event_label=event_label)
    plot_confusion_matrices(models_data, thresholds, X_test, y_test,
                            save_path=reports_dir / "matrices_confusion.png",
                            event_label=event_label, neg_label=neg_label)
    plot_feature_importance(best["estimator"], feature_names, best["name"],
                            save_path=reports_dir / "feature_importance.png")
    plot_learning_curve_best(best["estimator"], best["name"],
                             X_train, y_train,
                             save_path=reports_dir / "learning_curve_best.png",
                             scoring=cv_scoring,
                             lc_cfg=train_cfg.get("learning_curve_config", {}))

    model_path = Path(paths["model_file"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["estimator"], model_path)
    logger.info("Mejor modelo guardado: %s", model_path)

    pipeline_ml_path = Path(paths["pipeline_ml_file"])
    pipeline_ml_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_ml, pipeline_ml_path)
    logger.info("pipeline_ml guardado: %s", pipeline_ml_path)

    pipeline_base_path = Path(paths["pipeline_base_file"])
    pipeline_base_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_base, pipeline_base_path)
    logger.info("pipeline_base guardado: %s", pipeline_base_path)

    best_test = next(m for m in all_metrics
                     if m["model"] == best["name"] and m["split"] == "Test")
    best_cv   = next(r for r in cv_results_list if r["model"] == best["name"])

    meta = {
        "use_case":         args.use_case,
        "event_col":        event_label,
        "model_name":       best["name"],
        "score":            best["score"],
        "gap_auc":          best["gap_auc"],
        "threshold":        thresholds[best["name"]],
        "threshold_strategy": threshold_strategy,
        "test_metrics":     best_test,
        "cv_metrics":       best_cv["metrics"],
        "baseline_roc":     baseline_roc,
        "feature_count":    len(feature_names),
        "train_size":       int(len(y_train)),
        "test_size":        int(len(y_test)),
        "train_mora_rate":  round(float(y_train.mean()), 4),
        "test_mora_rate":   round(float(y_test.mean()),  4),
        "split_cutoff":     cfg["split"].get("train_cutoff"),
        "paths":            paths,
    }
    meta_path = Path(paths["model_meta_file"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    all_metrics_ext = all_metrics + [
        {"model": r["model"], "split": "CV", **{
            k: r["metrics"][k]["mean"] for k in r["metrics"]
        }} for r in cv_results_list
    ]
    metrics_path = Path(paths["metrics_file"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_metrics_ext, f, indent=2)

    ref_path = Path(paths["train_reference_file"])
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_df   = X_train.copy()
    ref_df[event_label] = y_train
    ref_df.to_csv(ref_path, index=False)

    logger.info("=" * 60)
    logger.info("RESULTADO FINAL")
    logger.info("=" * 60)
    logger.info("use_case          : %s", args.use_case)
    logger.info("Mejor modelo      : %s", best["name"])
    logger.info("Score seleccion   : %.4f", best["score"])
    logger.info("Threshold         : %.4f", thresholds[best["name"]])
    logger.info("ROC-AUC CV mean   : %.4f", best_cv["metrics"]["roc_auc"]["mean"])
    logger.info("Recall CV mean    : %.4f", best_cv["metrics"]["recall"]["mean"])
    logger.info("ROC-AUC test      : %.4f", best_test["roc_auc"])
    logger.info("Recall %s test   : %.4f", event_label, best_test["recall_event"])
    logger.info("Gap AUC           : %.4f", best["gap_auc"])
    logger.info("=" * 60)

# ──────────────────────────────────────────────────