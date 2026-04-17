"""
heuristic_model.py
==================
Modelo heurístico de referencia (baseline) para cualquier caso de uso del pipeline.

Un modelo heurístico es una regla de negocio simple derivada del EDA.
Su propósito NO es competir con ML sino establecer un piso mínimo de
desempeño que cualquier modelo entrenado DEBE superar para justificarse.

Diseño escalable
----------------
La clase HeuristicMoraModel es completamente genérica: no contiene ningún
nombre de columna ni threshold hardcodeado. Todo se lee desde config.json
bajo use_cases.<nombre>.heuristic.rules:

    "heuristic": {
        "min_signals": 2,
        "rules": [
            {"col": "puntaje_datacredito", "op": "<",  "threshold": 760},
            {"col": "huella_consulta",     "op": ">",  "threshold": 5},
            {"col": "plazo_meses",         "op": ">",  "threshold": 12}
        ]
    }

Operadores soportados: "<", "<=", ">", ">=", "==", "!="

Para un nuevo caso de uso basta con declarar sus reglas en config.json.
No se modifica código.

Evaluación del baseline (lineamientos del proyecto):
  Performance  → ROC-AUC, PR-AUC, recall mora
  Consistency  → KFold(10) cross-validation + learning curve (ShuffleSplit 50)
  Scalability  → fit time vs training size

Uso:
  python src/heuristic_model.py --use-case scoring_mora
  from heuristic_model import HeuristicRuleModel, build_heuristic_from_config
"""

from __future__ import annotations

import json
import logging
import operator
from pathlib import Path
from typing import Any

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

# Operadores soportados para las reglas
_OPS: dict[str, Any] = {
    "<":  operator.lt,
    "<=": operator.le,
    ">":  operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


# ──────────────────────────────────────────────────────────
#  Modelo Heurístico Genérico

class HeuristicRuleModel(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en reglas de negocio derivadas del EDA.
    Hereda BaseEstimator + ClassifierMixin → compatible con
    cross_val_score, Pipeline y learning_curve.

    Completamente genérico: las columnas, operadores y thresholds
    se declaran en config.json. No hay nombres de columna hardcodeados.

    Parámetros
    ----------
    rules : list[dict]
        Lista de reglas. Cada regla es un dict con:
            col       : nombre de la columna (o sufijo post-pipeline)
            op        : operador como string ("<", "<=", ">", ">=", "==", "!=")
            threshold : valor de corte numérico
        Ejemplo:
            [
                {"col": "puntaje_datacredito", "op": "<",  "threshold": 760},
                {"col": "huella_consulta",     "op": ">",  "threshold": 5},
                {"col": "plazo_meses",         "op": ">",  "threshold": 12}
            ]

    min_signals : int
        Mínimo de reglas activas simultáneamente para predecir clase positiva.
        Default: 2 de N reglas.
    """

    def __init__(
        self,
        rules: list[dict] | None = None,
        min_signals: int = 2,
    ):
        self.rules       = rules or []
        self.min_signals = min_signals

    def fit(self, _x=None, y=None):
        if not self.rules:
            raise ValueError(
                "HeuristicRuleModel.rules está vacío. "
                "Define las reglas en config.json > use_cases.<nombre>.heuristic.rules "
                "y usa build_heuristic_from_config() para instanciar el modelo."
            )
        self.classes_ = np.array([0, 1])
        return self

    def _resolve_col(self, X: pd.DataFrame, col_name: str) -> pd.Series:
        """
        Busca la columna por nombre exacto o por sufijo post-ColumnTransformer.
        Ejemplo: "puntaje_datacredito" encuentra "numeric__puntaje_datacredito".
        """
        if col_name in X.columns:
            return X[col_name]
        matches = [c for c in X.columns if c == col_name or c.endswith(f"__{col_name}")]
        if matches:
            return X[matches[0]]
        raise KeyError(
            f"Columna '{col_name}' no encontrada en el DataFrame.\n"
            f"Columnas disponibles: {sorted(X.columns.tolist())}"
        )

    def _evaluar_reglas(self, X: pd.DataFrame) -> np.ndarray:
        """
        Evalúa cada regla y retorna una matriz (n_samples, n_rules)
        donde 1.0 indica que la señal de riesgo está activa.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HeuristicRuleModel requiere un DataFrame pandas.")

        signals = []
        for rule in self.rules:
            col       = rule["col"]
            op_str    = rule["op"]
            threshold = rule["threshold"]

            if op_str not in _OPS:
                raise ValueError(
                    f"Operador '{op_str}' no soportado. "
                    f"Válidos: {list(_OPS.keys())}"
                )

            col_data = pd.to_numeric(self._resolve_col(X, col), errors="coerce")
            signal   = _OPS[op_str](col_data, threshold).astype(float)
            signals.append(signal.values)

        return np.column_stack(signals)  # (n_samples, n_rules)

    def predict(self, X) -> np.ndarray:
        signal_matrix = self._evaluar_reglas(X)
        return (signal_matrix.sum(axis=1) >= self.min_signals).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """Score continuo: proporción de señales activas sobre el total de reglas."""
        signal_matrix = self._evaluar_reglas(X)
        score_positivo = signal_matrix.mean(axis=1)
        return np.column_stack([1 - score_positivo, score_positivo])

    def describe_rules(self) -> str:
        """Descripción legible de las reglas activas."""
        lines = [f"HeuristicRuleModel — {len(self.rules)} reglas | min_signals={self.min_signals}"]
        for i, r in enumerate(self.rules, 1):
            lines.append(f"  R{i}: {r['col']} {r['op']} {r['threshold']}")
        return "\n".join(lines)


# Alias backward-compatible para no romper imports existentes
HeuristicMoraModel = HeuristicRuleModel

def build_heuristic_from_config(cfg: dict) -> HeuristicRuleModel:
    """
    Instancia HeuristicRuleModel desde la sección heuristic del cfg resuelto.

    Espera:
        cfg["heuristic"]["rules"]       → lista de reglas
        cfg["heuristic"]["min_signals"] → entero (default: 2)

    cfg debe haber sido resuelto por resolve_cfg() para que contenga
    los parámetros del use_case correcto.
    """
    heuristic_cfg = cfg.get("heuristic")
    if not heuristic_cfg:
        raise KeyError(
            "No se encontró 'heuristic' en el config resuelto. "
            "Agrega una sección 'heuristic' en use_cases.<nombre> en config.json."
        )

    rules       = heuristic_cfg.get("rules", [])
    min_signals = int(heuristic_cfg.get("min_signals", 2))

    if not rules:
        raise ValueError(
            "La sección 'heuristic.rules' está vacía. "
            "Define al menos una regla en config.json."
        )

    model = HeuristicRuleModel(rules=rules, min_signals=min_signals)
    logger.info(model.describe_rules())
    return model


# ──────────────────────────────────────────────────────────
#  Evaluación en test

def evaluate_heuristic(
    model: HeuristicRuleModel,
    X: pd.DataFrame,
    y: np.ndarray,
    split_name: str = "Test",
    event_label: str = "Event",
) -> dict:
    """
    Evalúa el modelo y retorna un diccionario de métricas.

    event_label: nombre de la clase positiva para el reporte (configurable
                 por use_case desde cfg["target"]["event_col"]).
    """
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, y_proba)
    pr_auc  = average_precision_score(y, y_proba)
    cm      = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    neg_label = "Negativo"
    report = classification_report(
        y, y_pred,
        target_names=[neg_label, event_label],
        output_dict=True,
        zero_division=0,
    )

    logger.info("=" * 55)
    logger.info("Evaluacion heuristica — %s", split_name)
    logger.info("=" * 55)
    logger.info("ROC-AUC : %.4f  |  PR-AUC: %.4f", roc_auc, pr_auc)
    logger.info("Recall %s: %.4f  |  Precision %s: %.4f",
                event_label, report[event_label]["recall"],
                event_label, report[event_label]["precision"])
    logger.info("\n%s", classification_report(
        y, y_pred, target_names=[neg_label, event_label], zero_division=0
    ))

    return {
        "model":               "HeuristicRuleModel",
        "split":               split_name,
        "roc_auc":             round(roc_auc, 4),
        "pr_auc":              round(pr_auc, 4),
        f"precision_{event_label}": round(report[event_label]["precision"], 4),
        f"recall_{event_label}":    round(report[event_label]["recall"],    4),
        f"f1_{event_label}":        round(report[event_label]["f1-score"],  4),
        "accuracy":            round(report["accuracy"], 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "min_signals":         model.min_signals,
        "n_rules":             len(model.rules),
    }


# ──────────────────────────────────────────────────────────
#  Cross-Validation  (Consistency)

def cross_validate_heuristic(
    model: HeuristicRuleModel,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """KFold(10) cross-validation. Retorna (cv_df, train_scores_dict)."""
    model_pipe      = Pipeline(steps=[("model", model)], memory=None)
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
    logger.info("Cross-Validation KFold(k=%d) — HeuristicRuleModel", n_splits)
    logger.info("=" * 55)
    for metric in scoring_metrics:
        logger.info("%-10s  CV mean: %.2f  std: %.2f  train: %.2f",
                    metric,
                    cv_df[metric].mean(),
                    cv_df[metric].std(),
                    train_scores[metric])

    return cv_df, train_scores


def plot_cv_boxplot(cv_df: pd.DataFrame, save_path: Path) -> None:
    """Boxplot de variabilidad entre folds."""
    _, ax = plt.subplots(figsize=(8, 4))
    cv_df.plot.box(
        ax=ax,
        title="Cross-Validation Boxplot — HeuristicRuleModel",
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
    """Barras Train Score vs CV Mean con barra de error."""
    metrics = cv_df.columns.tolist()
    x       = np.arange(len(metrics))
    width   = 0.35

    _, ax = plt.subplots(figsize=(8, 4))
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
    ax.set_title("Training vs Cross-Validation — HeuristicRuleModel", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Train vs CV guardado: %s", save_path)


# ──────────────────────────────────────────────────────────
#  Learning Curve  (Consistency + Scalability)

def plot_learning_curve(
    model: HeuristicRuleModel,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    save_path: Path,
    scoring:    str  = "recall",
    n_shuffles: int  = 50,
    test_size:  float = 0.2,
) -> None:
    """
    Learning curve con ShuffleSplit.
    Panel izquierdo : score vs tamaño del dataset (Consistency).
    Panel derecho   : fit time vs tamaño del dataset (Scalability).
    n_shuffles y test_size se leen de config.training.learning_curve_config.
    """
    model_pipe = Pipeline(steps=[("model", model)], memory=None)
    cv = ShuffleSplit(n_splits=n_shuffles, test_size=test_size, random_state=42)

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

    _, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(train_sizes, train_mean, "o-", color="#1D9E75", label="Training score")
    ax.plot(train_sizes, test_mean,  "o-", color="#E24B4A", label="Cross-validation score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="#1D9E75")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color="#E24B4A")
    ax.set_title("Learning Curve for HeuristicRuleModel", fontweight="bold", fontsize=11)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring.capitalize())
    ax.legend(loc="best")
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.plot(train_sizes, fit_mean, "o-", color="#378ADD")
    ax.fill_between(train_sizes, fit_mean - fit_std, fit_mean + fit_std,
                    alpha=0.2, color="#378ADD")
    ax.set_title("Scalability — Fit Time", fontweight="bold", fontsize=11)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Fit time (s)")

    plt.suptitle("HeuristicRuleModel — Consistency & Scalability",
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
#  Ejecución standalone

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from ft_engineering import build_features, load_config, resolve_cfg

    parser = argparse.ArgumentParser(
        description="Baseline heurístico basado en reglas — pipeline MLOps."
    )
    parser.add_argument(
        "--use-case", type=str, dest="use_case", default="scoring_mora",
        help="Caso de uso en config.json > use_cases (default: scoring_mora).",
    )
    args = parser.parse_args()

    # 1. Config resuelto por use_case → paths, heuristic y target correctos
    cfg_global, _ = load_config()
    cfg = resolve_cfg(cfg_global, args.use_case)

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    reports_dir   = Path(cfg["paths"]["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    event_label = cfg["target"]["event_col"]

    # 2. Features
    # return_base=True expone X_train_base / X_test_base: DataFrames en escala
    # original (salida de pipeline_base, antes de StandardScaler).
    # El modelo heurístico evalúa reglas con thresholds en escala original
    # (ej. puntaje_datacredito < 760). Si recibiera la salida de pipeline_ml
    # (z-scores), los thresholds serían imposibles (z < 760 siempre True,
    # z > 5 o z > 12 nunca True) → Recall = 0.
    logger.info("Cargando features para use_case='%s'...", args.use_case)
    X_train_ml, X_test_ml, y_train, y_test, _, _, X_train_base, X_test_base = build_features(
        use_case=args.use_case,
        return_dataframe=True,
        return_base=True,
    )

    # 3. Modelo construido desde config — sin hardcodeo
    model = build_heuristic_from_config(cfg)
    model.fit(X_train_base, y_train)

    # 4. Evaluación en train/test (escala original — reglas interpretables)
    metrics_train = evaluate_heuristic(model, X_train_base, y_train, "Train", event_label)
    metrics_test  = evaluate_heuristic(model, X_test_base,  y_test,  "Test",  event_label)

    # 5. Cross-validation KFold(n)
    train_cfg = cfg.get("training", {})
    heuristic_cv_splits = int(train_cfg.get("heuristic_cv_splits", 10))
    lc_cfg = train_cfg.get("learning_curve_config", {})

    logger.info("Ejecutando cross-validation KFold(%d)...", heuristic_cv_splits)
    cv_df, train_scores = cross_validate_heuristic(
        model, X_train_base, y_train, n_splits=heuristic_cv_splits
    )

    plot_cv_boxplot(cv_df, save_path=reports_dir / "heuristic_cv_boxplot.png")
    plot_train_vs_cv(cv_df, train_scores, save_path=reports_dir / "heuristic_train_vs_cv.png")

    # 6. Learning curve (ShuffleSplit — config-driven)
    n_shuffles = int(lc_cfg.get("n_splits", 50))
    lc_test_size = float(lc_cfg.get("test_size", 0.2))
    logger.info("Generando learning curve (ShuffleSplit n=%d)...", n_shuffles)
    plot_learning_curve(
        model, X_train_base, y_train,
        save_path=reports_dir / "heuristic_learning_curve.png",
        scoring="recall",
        n_shuffles=n_shuffles,
        test_size=lc_test_size,
    )

    # 7. Guardar baseline
    baseline_path = reports_dir / "heuristic_baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump({
            "use_case": args.use_case,
            "rules":    model.rules,
            "train":    metrics_train,
            "test":     metrics_test,
            "cv_summary": {
                m: {
                    "mean": round(float(cv_df[m].mean()), 4),
                    "std":  round(float(cv_df[m].std()),  4),
                }
                for m in cv_df.columns
            },
        }, f, indent=2)
    logger.info("Baseline guardado: %s", baseline_path)

    # 8. Resumen final
    logger.info("=" * 55)
    logger.info("use_case : %s", args.use_case)
    logger.info("BASELINE — ROC-AUC  test : %.4f", metrics_test["roc_auc"])
    logger.info("BASELINE — PR-AUC   test : %.4f", metrics_test["pr_auc"])
    logger.info("BASELINE — Recall   %s : %.4f",
                event_label, metrics_test.get(f"recall_{event_label}", float("nan")))
    logger.info("CV Recall KFold-10       : %.4f +/- %.4f",
                cv_df["recall"].mean(), cv_df["recall"].std())
    logger.info("Todo modelo ML debe superar estos valores para justificarse.")
    logger.info("=" * 55)
    
# ──────────────────────────────────────────────────────────