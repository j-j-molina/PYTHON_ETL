"""
heuristic_model.py
==================
Modelo heurístico de referencia (baseline) para el modelo de mora crediticia.

Un modelo heurístico es una regla de negocio simple basada en los hallazgos
del EDA. Su propósito NO es competir con ML sino establecer un piso mínimo
de desempeño que cualquier modelo entrenado DEBE superar para justificarse.

Reglas derivadas del EDA (IV y análisis por decil):
  R1: puntaje_datacredito < 760   → señal de riesgo  (IV=0.199, media mora=749)
  R2: huella_consulta     > 5     → señal de riesgo  (IV=0.146, media mora=5.2)
  R3: plazo_meses         > 12    → señal de riesgo  (IV=0.128, media mora=12.5)

Política de decisión:
  mora_pred = 1  si  suma_señales >= min_signals  (default: 2 de 3)

Evaluación del baseline (lineamientos del proyecto):
  Performance  → ROC-AUC, PR-AUC, recall mora
  Consistency  → KFold(10) cross-validation + learning curve (ShuffleSplit 50)
  Scalability  → fit time vs training size

Uso:
  python heuristic_model.py
  from heuristic_model import HeuristicMoraModel
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    cross_val_score,
    learning_curve,
)
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


# ──────────────────────────────────────────────────────────
#  Modelo Heurístico

class HeuristicMoraModel(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en reglas de negocio derivadas del EDA.
    Hereda BaseEstimator + ClassifierMixin → compatible con
    cross_val_score, Pipeline y learning_curve.

    Parámetros:
        threshold_puntaje:  puntaje_datacredito por debajo del cual hay riesgo.
        threshold_huella:   huella_consulta por encima del cual hay riesgo.
        threshold_plazo:    plazo_meses por encima del cual hay riesgo.
        min_signals:        mínimo de señales activas para predecir mora=1.
    """

    _COL_PUNTAJE = "numeric__puntaje_datacredito"
    _COL_HUELLA  = "numeric__huella_consulta"
    _COL_PLAZO   = "numeric__plazo_meses"

    def __init__(
        self,
        threshold_puntaje: float = 760.0,
        threshold_huella:  float = 5.0,
        threshold_plazo:   float = 12.0,
        min_signals:       int   = 2,
    ):
        self.threshold_puntaje = threshold_puntaje
        self.threshold_huella  = threshold_huella
        self.threshold_plazo   = threshold_plazo
        self.min_signals       = min_signals

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def _get_col(self, X: pd.DataFrame, col_name: str) -> pd.Series:
        if col_name in X.columns:
            return X[col_name]
        suffix  = col_name.split("__")[-1]
        matches = [c for c in X.columns if c == suffix or c.endswith(f"__{suffix}")]
        if matches:
            return X[matches[0]]
        raise KeyError(f"Columna '{col_name}' no encontrada.")

    def _senales(self, X: pd.DataFrame):
        """Activa señales con umbrales fijos derivados del EDA (medias del grupo mora).

        Los thresholds no dependen del batch de entrada — son constantes de negocio:
          puntaje_datacredito: media al día=782 vs mora=749  → threshold=760
          huella_consulta:     media al día=4.2 vs mora=5.2  → threshold=5
          plazo_meses:         media al día=10.5 vs mora=12.5 → threshold=12
        Comparar contra la media del batch era incorrecto: el baseline cambiaría
        con cada conjunto de datos y no reflejaría las reglas de negocio del EDA.
        """
        puntaje = self._get_col(X, self._COL_PUNTAJE)
        huella  = self._get_col(X, self._COL_HUELLA)
        plazo   = self._get_col(X, self._COL_PLAZO)
        s1 = (puntaje < self.threshold_puntaje).astype(float)
        s2 = (huella  > self.threshold_huella ).astype(float)
        s3 = (plazo   > self.threshold_plazo  ).astype(float)
        return s1, s2, s3

    def predict(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HeuristicMoraModel requiere un DataFrame pandas.")
        s1, s2, s3 = self._senales(X)
        return ((s1 + s2 + s3) >= self.min_signals).astype(int).values

    def predict_proba(self, X) -> np.ndarray:
        """Score continuo: proporcion de senales activas."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HeuristicMoraModel requiere un DataFrame pandas.")
        s1, s2, s3 = self._senales(X)
        score_mora = (s1 + s2 + s3) / 3.0
        return np.column_stack([(1 - score_mora).values, score_mora.values])

# ──────────────────────────────────────────────────────────
#  Evaluación en test

def evaluate_heuristic(
    model: HeuristicMoraModel,
    X: pd.DataFrame,
    y: np.ndarray,
    split_name: str = "Test",
) -> dict:
    """Evalua el modelo y retorna un diccionario de metricas."""
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, y_proba)
    pr_auc  = average_precision_score(y, y_proba)
    cm      = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y, y_pred, target_names=["Al dia", "Mora"],
        output_dict=True, zero_division=0,
    )

    logger.info("=" * 55)
    logger.info("Evaluacion heuristica — %s", split_name)
    logger.info("=" * 55)
    logger.info("ROC-AUC : %.4f  |  PR-AUC: %.4f", roc_auc, pr_auc)
    logger.info("Recall mora: %.4f  |  Precision mora: %.4f",
                report["Mora"]["recall"], report["Mora"]["precision"])
    logger.info("\n%s", classification_report(y, y_pred,
                target_names=["Al dia", "Mora"], zero_division=0))

    return {
        "model":          "HeuristicMoraModel",
        "split":          split_name,
        "roc_auc":        round(roc_auc, 4),
        "pr_auc":         round(pr_auc, 4),
        "precision_mora": round(report["Mora"]["precision"], 4),
        "recall_mora":    round(report["Mora"]["recall"], 4),
        "f1_mora":        round(report["Mora"]["f1-score"], 4),
        "accuracy":       round(report["accuracy"], 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "min_signals": model.min_signals,
    }

# ──────────────────────────────────────────────────────────
#  Cross-Validation  (Consistency)

def cross_validate_heuristic(
    model: HeuristicMoraModel,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = 10,
) -> tuple:
    """
    KFold(10) cross-validation. Retorna (cv_df, train_scores_dict).
    Replica el analisis del profe: variabilidad entre folds.
    """
    model_pipe      = Pipeline(steps=[("model", model)])
    kfold           = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring_metrics = ["accuracy", "f1", "precision", "recall"]

    cv_results   = {}
    train_scores = {}

    for metric in scoring_metrics:
        cv_results[metric] = cross_val_score(
            model_pipe, X_train, y_train, cv=kfold, scoring=metric
        )
        model_pipe.fit(X_train, y_train)
        train_scores[metric] = model_pipe.score(X_train, y_train)

    cv_df = pd.DataFrame(cv_results)

    logger.info("=" * 55)
    logger.info("Cross-Validation KFold(k=%d) — HeuristicMoraModel", n_splits)
    logger.info("=" * 55)
    for metric in scoring_metrics:
        logger.info("%-10s  CV mean: %.2f  std: %.2f  train: %.2f",
                    metric,
                    cv_df[metric].mean(),
                    cv_df[metric].std(),
                    train_scores[metric])

    return cv_df, train_scores

def plot_cv_boxplot(cv_df: pd.DataFrame, save_path: Path) -> None:
    """Boxplot de variabilidad entre folds — replica el grafico del profe."""
    fig, ax = plt.subplots(figsize=(8, 4))
    cv_df.plot.box(
        ax=ax,
        title="Cross-Validation Boxplot — HeuristicMoraModel",
        ylabel="Score",
        color={"boxes": "#378ADD", "whiskers": "#378ADD",
               "medians": "#E24B4A", "caps": "#378ADD"},
    )
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("CV Boxplot guardado: %s", save_path)

def plot_train_vs_cv(
    cv_df: pd.DataFrame,
    train_scores: dict,
    save_path: Path,
) -> None:
    """Barras Train Score vs CV Mean con barra de error — replica del profe."""
    metrics    = cv_df.columns.tolist()
    x          = np.arange(len(metrics))
    width      = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2,
           [train_scores[m] for m in metrics], width,
           label="Train Score", color="#1D9E75", alpha=0.85)
    ax.bar(x + width / 2,
           [cv_df[m].mean() for m in metrics], width,
           yerr=[cv_df[m].std() for m in metrics],
           capsize=4, label="CV Mean", color="#378ADD", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Training vs Cross-Validation — HeuristicMoraModel",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Train vs CV guardado: %s", save_path)

# ──────────────────────────────────────────────────────────
#  Learning Curve  (Consistency + Scalability)

def plot_learning_curve(
    model: HeuristicMoraModel,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    save_path: Path,
    scoring:    str = "recall",
    n_shuffles: int = 50,
) -> None:
    """
    Learning curve con ShuffleSplit(50) — identica al profe.
    Panel izquierdo : score vs tamano del dataset (Consistency).
    Panel derecho   : fit time vs tamano del dataset (Scalability).
    Metrica principal: recall (capturar moras es prioritario).
    """
    model_pipe = Pipeline(steps=[("model", model)])
    cv         = ShuffleSplit(n_splits=n_shuffles, test_size=0.2, random_state=42)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        model_pipe, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv, scoring=scoring,
        n_jobs=1, return_times=True,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    test_mean  = test_scores.mean(axis=1)
    test_std   = test_scores.std(axis=1)
    fit_mean   = fit_times.mean(axis=1)
    fit_std    = fit_times.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Learning curve
    ax = axes[0]
    ax.plot(train_sizes, train_mean, "o-", color="#1D9E75", label="Training score")
    ax.plot(train_sizes, test_mean,  "o-", color="#E24B4A", label="Cross-validation score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="#1D9E75")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color="#E24B4A")
    ax.set_title(f"Learning Curve for HeuristicMoraModel",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring.capitalize())
    ax.legend(loc="best")
    ax.set_ylim(0, 1)

    # Panel 2: Scalability
    ax = axes[1]
    ax.plot(train_sizes, fit_mean, "o-", color="#378ADD")
    ax.fill_between(train_sizes, fit_mean - fit_std, fit_mean + fit_std,
                    alpha=0.2, color="#378ADD")
    ax.set_title("Scalability — Fit Time", fontweight="bold", fontsize=11)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Fit time (s)")

    plt.suptitle("HeuristicMoraModel — Consistency & Scalability",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Learning curve guardada: %s", save_path)

    logger.info("Training Sizes  : %s", train_sizes.astype(int))
    logger.info("Train score mean: %s", np.round(train_mean, 3))
    logger.info("CV score mean   : %s", np.round(test_mean, 3))
    logger.info("Fit time mean   : %s s", np.round(fit_mean, 4))

# ──────────────────────────────────────────────────────────
#  Ejecucion standalone

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from ft_engineering import build_features, load_config

    cfg, repo_root = load_config()
    artifacts_dir  = repo_root / cfg["paths"]["artifacts_dir"]
    reports_dir    = repo_root / cfg["paths"]["reports_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Features
    logger.info("Cargando features...")
    X_train, X_test, y_train, y_test, _, _ = build_features(return_dataframe=True)

    # Modelo
    model = HeuristicMoraModel(min_signals=2)
    model.fit(X_train, y_train)

    # Evaluacion en train/test
    metrics_train = evaluate_heuristic(model, X_train, y_train, "Train")
    metrics_test  = evaluate_heuristic(model, X_test,  y_test,  "Test")

    # Cross-validation KFold(10)
    logger.info("Ejecutando cross-validation KFold(10)...")
    cv_df, train_scores = cross_validate_heuristic(model, X_train, y_train, n_splits=10)

    plot_cv_boxplot(
        cv_df,
        save_path=reports_dir / "heuristic_cv_boxplot.png",
    )
    plot_train_vs_cv(
        cv_df, train_scores,
        save_path=reports_dir / "heuristic_train_vs_cv.png",
    )

    # Learning curve (ShuffleSplit 50)
    logger.info("Generando learning curve (ShuffleSplit n=50)...")
    plot_learning_curve(
        model, X_train, y_train,
        save_path=reports_dir / "heuristic_learning_curve.png",
        scoring="recall",
        n_shuffles=50,
    )

    # Guardar baseline
    baseline_path = reports_dir / "heuristic_baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump({
            "train": metrics_train,
            "test":  metrics_test,
            "cv_summary": {
                m: {
                    "mean": round(float(cv_df[m].mean()), 4),
                    "std":  round(float(cv_df[m].std()),  4),
                }
                for m in cv_df.columns
            },
        }, f, indent=2)
    logger.info("Baseline guardado: %s", baseline_path)

    # Resumen
    logger.info("=" * 55)
    logger.info("BASELINE — ROC-AUC  test : %.4f", metrics_test["roc_auc"])
    logger.info("BASELINE — PR-AUC   test : %.4f", metrics_test["pr_auc"])
    logger.info("BASELINE — Recall   mora : %.4f", metrics_test["recall_mora"])
    logger.info("CV Recall KFold-10       : %.4f +/- %.4f",
                cv_df["recall"].mean(), cv_df["recall"].std())
    logger.info("Todo modelo ML debe superar estos valores para justificarse.")
    logger.info("=" * 55)

    # ──────────────────────────────────────────────────────────