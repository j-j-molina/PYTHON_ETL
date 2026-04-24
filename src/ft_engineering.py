"""
ft_engineering.py
--------------------
Primera componente del flujo MLOps — Ingeniería de Features.

Cada paso de transformación es una clase independiente que
hereda de BaseEstimator + TransformerMixin, lo que hace
todo el pipeline compatible con GridSearchCV y cross_val_score.

Todo el comportamiento se controla desde config.json.

Estructura (corregida para evitar data leakage):
  ┌─────────────────────────────────────────────────────┐
  │           pipeline_stateless                        │
  │  (aplica sobre el DataFrame COMPLETO, pre-split)    │
  │  Pasos SIN estado — fit() es un no-op               │
  ├─────────────────────────────────────────────────────┤
  │  CrearFeaturesDerivadas → LimpiarTendenciaIngresos  │
  └─────────────────────────────────────────────────────┘
              ↓  temporal_split()
  ┌─────────────────────────────────────────────────────┐
  │           pipeline_base                             │
  │  (fit SOLO sobre train — evita data leakage)        │
  │  Pasos CON estado — aprenden estadísticos de train  │
  ├─────────────────────────────────────────────────────┤
  │  ImputacionSegmentada → Winsorizar →                │
  │  EliminarColumnas                                   │
  └─────────────────────────────────────────────────────┘
              ↓
  ┌─────────────────────────────────────────────────────┐
  │           pipeline_ml                               │
  │  (fit SOLO sobre train — encoding + escalado)       │
  ├─────────────────────────────────────────────────────┤
  │  ColumnTransformer:                                 │
  │    numeric   → SimpleImputer(median)+StandardScaler │
  │    categoric → SimpleImputer(mode)+OneHotEncoder    │
  │    cat_ord   → SimpleImputer(mode)+OrdinalEncoder   │
  └─────────────────────────────────────────────────────┘

Separar los pasos stateless de los stateful es la corrección
principal de esta versión: ImputacionSegmentada (medianas por
tipo_laboral) y Winsorizar (caps al p99) ahora se ajustan
ÚNICAMENTE sobre el conjunto de train, tal como requiere
cualquier pipeline libre de leakage.

Uso como módulo:
  from ft_engineering import build_features
  X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base = build_features()

Uso como script:
  python ft_engineering.py
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────
#  Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Utilidades: config y repo

def _find_repo_root(start: Path) -> Path:
    """Sube hasta encontrar la raíz del repositorio git (contiene .git/)."""
    p = start.resolve()
    for _ in range(8):
        if (p / ".git").exists():
            return p
        p = p.parent
    return start.resolve()

def load_config(config_path: Optional[Path] = None) -> tuple[dict, Path]:
    """
    Carga config.json buscando en rutas estándar del proyecto.
    Retorna (cfg, repo_root).
    """
    here = Path(__file__).resolve().parent
    repo_root = _find_repo_root(here)

    candidates = [
        config_path,
        here / "config.json",
        repo_root / "mlops_pipeline" / "src" / "config.json",
        repo_root / "config.json",
    ]
    path = next((p for p in candidates if p is not None and Path(p).exists()), None)
    if path is None:
        raise FileNotFoundError(
            "No se encontró config.json. "
            "Colócalo en mlops_pipeline/src/ o pasa la ruta explícita."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    logger.info("config.json cargado: %s", path)
    return cfg, repo_root


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge recursivo: los valores de override ganan en conflicto."""
    import copy
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_cfg(cfg: dict, use_case: str) -> dict:
    """
    Construye la configuración efectiva para un caso de uso específico.

    Estrategia de resolución (de menor a mayor precedencia):
      1. Secciones globales del config.json base
         (feature_engineering, split, target, paths, training, monitoring, deploy)
      2. Overrides declarados en config['use_cases'][use_case]
         (cualquier sección que el caso de uso quiera sobreescribir)

    Esto permite que:
      - scoring_mora use sus propias features, target y cutoff.
      - scoring_fraude defina features y target completamente distintos.
      - Ambos coexistan en el mismo config.json sin conflicto.
      - Los pipeline builders (build_pipeline_base, build_pipeline_ml, etc.)
        reciban siempre un cfg con la estructura esperada — no se modifican.

    Ejemplo de uso en config.json para un segundo caso de uso:
        "use_cases": {
            "scoring_mora": {
                "feature_engineering": { ... },
                "target": { "label_col": "Pago_atiempo", ... },
                "split":  { "train_cutoff": "2025-09" },
                "paths":  { "artifacts_dir": "mlops_pipeline/artifacts/scoring_mora" }
            },
            "scoring_fraude": {
                "feature_engineering": { ... },   # features distintas
                "target": { "label_col": "es_fraude", ... },
                "split":  { "type": "random", "test_size": 0.2 },
                "paths":  { "artifacts_dir": "mlops_pipeline/artifacts/scoring_fraude" }
            }
        }
    """
    use_cases = cfg.get("use_cases", {})
    if use_case not in use_cases:
        available = ", ".join(use_cases.keys()) if use_cases else "ninguno"
        raise KeyError(
            f"Caso de uso '{use_case}' no encontrado en config.json. "
            f"Disponibles: {available}"
        )

    uc_overrides = use_cases[use_case]
    resolved = _deep_merge(cfg, uc_overrides)
    resolved["_use_case"] = use_case
    logger.info("Config resuelto para use_case='%s'", use_case)
    return resolved

# ──────────────────────────────────────────────────────────
#  TRANSFORMADORES PERSONALIZADOS
#  Cada clase hereda BaseEstimator + TransformerMixin para
#  ser compatible con Pipeline, GridSearchCV y cross_val_score
# ──────────────────────────────────────────────────────────

class CrearFeaturesDerivadas(BaseEstimator, TransformerMixin):
    """
    Crea columnas derivadas según lo declarado en config.json.
    Es IDEMPOTENTE: si la columna ya existe no la sobreescribe.

    Paso SIN estado: fit() es un no-op. Seguro de aplicar sobre
    el dataset completo antes del split temporal.

    Completamente genérico — no contiene ningún nombre de columna
    hardcodeado. Todo se lee del cfg efectivo (ya resuelto por
    resolve_cfg) bajo "feature_engineering.derived_features":

      ratio_features   → col = numerator / denominator
                         "numerator_sum": [colA, colB] suma antes de dividir
      sum_features     → col = sum(cols)
      pct_features     → col = numerator / denominator_col (ya creada)
      temporal_col     → año, mes, día semana, antigüedad desde ref_date
      age_col/bins     → bucketing de cualquier variable continua

    Para un nuevo caso de uso basta con definir su bloque en config.json
    bajo use_cases.<nombre>.feature_engineering.derived_features.
    No se modifica código.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        derived  = self.cfg["feature_engineering"]["derived_features"]
        date_col = self.cfg["split"]["date_col"]

        # --- Ratios: name = numerator / denominator ------------------
        # "numerator_sum": [colA, colB] suma antes de dividir
        # "denominator_add": N  → suma N al denominador antes de dividir (evita /0)
        for r in derived.get("ratio_features", []):
            if r["name"] in X.columns:
                continue
            denom = pd.to_numeric(X[r["denominator"]], errors="coerce")
            add   = r.get("denominator_add", 0)
            denom = (denom + add) if add else denom.replace(0, np.nan)
            numer = sum(X[c] for c in r["numerator_sum"]) if "numerator_sum" in r else X[r["numerator"]]
            X[r["name"]] = numer / denom

        # --- Sumas: name = sum(cols) ---------------------------------
        for s in derived.get("sum_features", []):
            if s["name"] not in X.columns:
                X[s["name"]] = sum(X[c] for c in s["cols"])

        # --- Porcentajes sobre columna ya creada --------------------
        for p in derived.get("pct_features", []):
            if p["name"] not in X.columns:
                X[p["name"]] = X[p["numerator"]] / X[p["denominator"]].replace(0, np.nan)

        # --- Features temporales ------------------------------------
        temporal_col = derived.get("temporal_col") or date_col
        if temporal_col in X.columns:
            if isinstance(X[temporal_col].dtype, pd.DatetimeTZDtype):
                X[temporal_col] = X[temporal_col].dt.tz_localize(None)

            ref_date = pd.Timestamp(derived["antiguedad_ref_date"]).tz_localize(None)

            self._add(X, derived.get("year_col", "anio_prestamo"), X[temporal_col].dt.year)
            self._add(X, derived.get("month_col", "mes_prestamo"), X[temporal_col].dt.month)
            self._add(X, derived.get("dayname_col", "dia_semana_prestamo"), X[temporal_col].dt.day_name())
            self._add(
                X,
                derived.get("antiguedad_col", "antiguedad_prestamo_dias"),
                (ref_date - X[temporal_col]).dt.days)

        # --- Bucketing de edad (legado) ------------------------------
        age_col        = derived.get("age_col",        "edad_cliente")
        age_bucket_col = derived.get("age_bucket_col", "edad_bucket")
        age_bins       = derived.get("edad_bucket_bins")
        age_labels     = derived.get("edad_bucket_labels")
        if age_col in X.columns and age_bins and age_labels:
            self._add(
                X, age_bucket_col,
                pd.cut(X[age_col], bins=age_bins, labels=age_labels, right=False)
                  .astype(str).replace("nan", np.nan),
            )

        # --- Bucketing genérico: bucket_features ---------------------
        # [{"name": "col_bucket", "col": "col_orig", "bins": [...], "labels": [...]}]
        for b in derived.get("bucket_features", []):
            if b["name"] in X.columns or b["col"] not in X.columns:
                continue
            self._add(
                X, b["name"],
                pd.cut(X[b["col"]], bins=b["bins"], labels=b["labels"],
                       right=b.get("right", False))
                  .astype(str).replace("nan", np.nan),
            )

        logger.info("CrearFeaturesDerivadas: %d columnas en el DataFrame", X.shape[1])
        return X

    @staticmethod
    def _add(df: pd.DataFrame, col: str, values) -> None:
        """Agrega la columna solo si no existe (idempotencia)."""
        if col not in df.columns:
            df[col] = values


class LimpiarCategoricas(BaseEstimator, TransformerMixin):
    """
    Reemplaza por NaN valores fuera de catálogo en columnas categóricas.

    Completamente genérico — opera sobre cualquier columna/catálogo
    declarado en cfg["feature_engineering"]["categorical_cleaners"]:
        { "columna": ["valor_valido_1", "valor_valido_2", ...] }

    Reemplaza a LimpiarTendenciaIngresos (hardcodeada a una columna).
    Para un nuevo caso de uso basta con declarar su sección en config.
    No se modifica código.

    Paso SIN estado: fit() es un no-op.
    """

    def __init__(self, cleaners: dict):
        self.cleaners = cleaners

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, valid_values in self.cleaners.items():
            if col not in X.columns:
                logger.warning("LimpiarCategoricas: columna '%s' no encontrada, omitida.", col)
                continue
            valid = set(valid_values)
            mask  = X[col].notna() & ~X[col].isin(valid)
            n     = int(mask.sum())
            if n:
                X.loc[mask, col] = np.nan
                logger.info("LimpiarCategoricas: '%s' — %d valores fuera de catálogo → NaN", col, n)
        return X

class ImputacionSegmentada(BaseEstimator, TransformerMixin):
    """
    EDA: promedio_ingresos_datacredito tiene 27.2% de nulos y la mediana
    difiere por tipo_laboral → imputación segmentada.

    Paso CON estado — debe fitearse SOLO sobre train para evitar leakage.

    fit():       aprende las medianas por grupo sobre el conjunto de entrenamiento.
    transform(): aplica esas medianas; si el grupo no se vio en train, usa
                 la mediana global del train como fallback.

    Acepta un dict {columna: columna_grupo} para ser extensible.
    """

    def __init__(self, impute_map: dict):
        self.impute_map = impute_map

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_: dict = {}
        self.global_medians_: dict = {}
        for col, group_col in self.impute_map.items():
            if col not in X.columns:
                continue
            self.medians_[col]        = X.groupby(group_col)[col].median().to_dict()
            self.global_medians_[col] = float(X[col].median())
            logger.info("ImputacionSegmentada fit: medianas de '%s' por '%s' aprendidas",
                        col, group_col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, group_col in self.impute_map.items():
            if col not in X.columns or col not in self.medians_:
                continue
            n_null    = int(X[col].isna().sum())
            group_med = X[group_col].map(self.medians_[col])
            X[col]    = X[col].fillna(group_med).fillna(self.global_medians_[col])
            logger.info("ImputacionSegmentada transform: %d nulos imputados en '%s'",
                        n_null, col)
        return X

class Winsorizar(BaseEstimator, TransformerMixin):
    """
    Aplica cap superior (winsorización) al percentil definido.

    Paso CON estado — debe fitearse SOLO sobre train para evitar leakage.
    Si se ajusta sobre el dataset completo, los caps del percentil 99
    se calculan incluyendo datos del test, lo cual contamina el proceso
    de entrenamiento.

    fit():       aprende los caps sobre el conjunto de entrenamiento.
    transform(): aplica clip() con esos caps a cualquier conjunto.

    EDA (cell 20): outliers IQR relevantes en salario_cliente (6.7%),
    capital_prestado (5.1%), total_otros_prestamos y cuota_pactada.
    """

    def __init__(self, cols: list, quantile: float = 0.99):
        self.cols = cols
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y=None):
        self.caps_ = {}
        for col in self.cols:
            if col not in X.columns:
                continue

            s = pd.to_numeric(X[col], errors="coerce").astype("float64")
            cap = s.quantile(self.quantile)

            if pd.notna(cap):
                self.caps_[col] = float(cap)

        logger.info("Winsorizar fit: caps aprendidos → %s", self.caps_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, cap in self.caps_.items():
            if col not in X.columns:
                continue

            s = pd.to_numeric(X[col], errors="coerce").astype("float64")
            n = int((s > cap).sum())
            X[col] = s.clip(upper=float(cap))

            logger.info("Winsorizar: %-30s cap=%.2f  (%d capeados)", col, cap, n)
        return X

class EliminarColumnas(BaseEstimator, TransformerMixin):
    """
    Elimina columnas por nombre.
    errors='ignore' evita errores si alguna columna ya no existe.

    Se usa al final del pipeline_base para quitar columnas de leakage
    (puntaje, Pago_atiempo) y la columna de fecha, una vez que ya
    se han creado los features temporales derivados de ella.
    """

    def __init__(self, cols_to_drop: list):
        self.cols_to_drop = cols_to_drop

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dropped = [c for c in self.cols_to_drop if c in X.columns]
        logger.info("EliminarColumnas: %s", dropped)
        return X.drop(columns=self.cols_to_drop, errors="ignore")


# ──────────────────────────────────────────────────────────
#  Helper: aplicar pipeline step-by-step sin check_is_fitted

def apply_pipeline_steps(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica los transformadores de un Pipeline serializados uno a uno,
    evitando el check_is_fitted interno de sklearn.Pipeline.transform().

    Útil en producción cuando el Pipeline fue serializado con joblib/pickle
    y sklearn no detecta el estado fitted de transformadores personalizados.

    Args:
        pipeline: Pipeline de sklearn con pasos ya ajustados (fitted).
        X:        DataFrame de entrada.

    Returns:
        DataFrame transformado.
    """
    for _, step in pipeline.steps:
        X = step.transform(X)
    return X

# ──────────────────────────────────────────────────────────
#  PIPELINE STATELESS
#  Pasos SIN estado — no aprenden estadísticos de los datos.
#  Seguro de aplicar sobre el dataset completo antes del split.
#  No se guarda en disco (fit() es un no-op, no hay parámetros aprendidos).
# ──────────────────────────────────────────────────────────

def build_pipeline_stateless(cfg: dict) -> Pipeline:
    """
    Pipeline sin estado — recibe cfg ya resuelto por resolve_cfg().

    Genérico: lee todo de cfg["feature_engineering"], que puede ser
    global o específico del caso de uso dependiendo de cómo se resolvió.
    No se serializa (fit() es no-op en todos sus pasos).
    """
    fe_cfg = cfg["feature_engineering"]
    return Pipeline(steps=[
        ("crear_features",     CrearFeaturesDerivadas(cfg=cfg)),
        ("limpiar_categoricas", LimpiarCategoricas(
            cleaners=fe_cfg.get("categorical_cleaners", {}),
        )),
    ])

# ──────────────────────────────────────────────────────────
#  PIPELINE BASE (stateful)
#  Pasos CON estado — aprenden estadísticos de los datos.
#  DEBE fitearse SOLO sobre el conjunto de train para evitar
#  que los caps de winsorización y las medianas de imputación
#  estén contaminados con información del conjunto de test.
#  Se guarda como pipeline_base.pkl para producción.
# ──────────────────────────────────────────────────────────

def build_pipeline_base(cfg: dict) -> Pipeline:
    """
    Construye el pipeline de pasos con estado desde config.json.

    Incluye:
      - ImputacionSegmentada: medianas de promedio_ingresos_datacredito
        segmentadas por tipo_laboral (EDA: 27.2% de nulos, medianas difieren)
      - Winsorizar: cap al p99 para salario_cliente, capital_prestado,
        total_otros_prestamos, cuota_pactada (EDA cell 20: outliers IQR)
      - EliminarColumnas: leakage_cols + drop_cols + date_col
        EDA (cell 33): saldo_mora, saldo_total, saldo_principal,
        saldo_mora_codeudor son leakage post-evento — se eliminan aquí
        explícitamente (antes solo caían por remainder='drop' en pipeline_ml,
        pero eso los deja presentes durante pipeline_base, lo cual es incorrecto).

    Requiere fit SOLO sobre train. Se serializa como pipeline_base.pkl.
    """
    fe_cfg       = cfg["feature_engineering"]
    leakage_cols = fe_cfg.get("leakage_cols", [])
    drop_cols    = fe_cfg.get("drop_cols", [])
    date_col     = cfg["split"]["date_col"]

    # Unión deduplicada: leakage + drop_cols explícitos + columna de fecha
    # dict.fromkeys preserva orden y elimina duplicados
    all_drop = list(dict.fromkeys(leakage_cols + drop_cols + [date_col]))

    return Pipeline(steps=[
        ("imputacion",    ImputacionSegmentada(
            impute_map=fe_cfg["impute_grouped"],
        )),
        ("winsorizar",    Winsorizar(
            cols=fe_cfg["winsorize_cols"],
            quantile=float(fe_cfg["winsorize_quantile"]),
        )),
        ("eliminar_cols", EliminarColumnas(
            cols_to_drop=all_drop,
        )),
    ])

def build_pipeline_basemodel(cfg: dict) -> Pipeline:
    """
    Alias de compatibilidad: retorna un pipeline de 5 pasos (stateless + stateful).

    NOTA: Este pipeline combinado NO debe usarse para fit en producción,
    ya que mezcla pasos sin estado con pasos con estado.
    Usa build_pipeline_stateless() + build_pipeline_base() por separado.
    """
    fe_cfg       = cfg["feature_engineering"]
    leakage_cols = fe_cfg["leakage_cols"]
    date_col     = cfg["split"]["date_col"]

    return Pipeline(steps=[
        ("crear_features",    CrearFeaturesDerivadas(cfg=cfg)),
        ("limpiar_categoricas", LimpiarCategoricas(
            cleaners=fe_cfg.get("categorical_cleaners", {}),
        )),
        ("imputacion",        ImputacionSegmentada(
            impute_map=fe_cfg["impute_grouped"],
        )),
        ("winsorizar",        Winsorizar(
            cols=fe_cfg["winsorize_cols"],
            quantile=float(fe_cfg["winsorize_quantile"]),
        )),
        ("eliminar_cols",     EliminarColumnas(
            cols_to_drop=leakage_cols + [date_col],
        )),
    ])

# ──────────────────────────────────────────────────────────
#  PIPELINE ML (ColumnTransformer)
#  Se fitea sobre train para evitar data leakage.

def build_pipeline_ml(cfg: dict) -> Pipeline:
    """Construye el pipeline ML (encoding + escalado) desde config.json."""
    fe_cfg = cfg["feature_engineering"]

    numeric_features     = fe_cfg["numeric_cols"]
    categorical_features = fe_cfg["categorical_cols"]
    ordinal_dict: dict   = fe_cfg["ordinal_cols"]
    ordinal_features     = list(ordinal_dict.keys())
    ordinal_categories   = [ordinal_dict[c] for c in ordinal_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric",   numeric_pipeline,      numeric_features),
            ("categoric", categorical_pipeline,  categorical_features),
            ("cat_ord",   ordinal_pipeline,       ordinal_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        "pipeline_ml — numeric: %d | nominal: %d | ordinal: %d",
        len(numeric_features), len(categorical_features), len(ordinal_features),
    )
    return Pipeline(steps=[("preprocessor", preprocessor)])


# ──────────────────────────────────────────────────────────
#  SPLIT TEMPORAL

def temporal_split(
    df: pd.DataFrame,
    cfg: dict,
    date_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split de datos. Estrategia leída de cfg["split"]["type"]:

      "temporal" (default recomendado para créditos):
          train <= train_cutoff / test > train_cutoff
          Evita leakage por efecto de maduración de cartera.

      "random":
          Split aleatorio estratificado por target.
          Requiere cfg["split"]["test_size"] (default 0.2).
          Útil para casos de uso sin componente temporal fuerte.

    Recibe cfg ya resuelto por resolve_cfg(), por lo que cada caso de
    uso puede definir su propia estrategia de split sin modificar código.
    """
    from sklearn.model_selection import train_test_split

    split_cfg = cfg["split"]
    event_col = cfg["target"]["event_col"]
    split_type = split_cfg.get("type", "temporal")

    if split_type == "temporal":
        cutoff        = split_cfg["train_cutoff"]
        cutoff_period = pd.Period(cutoff, freq="M")
        periods       = df[date_col].dt.to_period("M")
        train_df = df[periods <= cutoff_period].copy()
        test_df  = df[periods >  cutoff_period].copy()
        logger.info(
            "Split temporal — cutoff: %s | train: %d (%.1f%%) | test: %d (%.1f%%)",
            cutoff,
            len(train_df), len(train_df) / len(df) * 100,
            len(test_df),  len(test_df)  / len(df) * 100,
        )

    elif split_type == "random":
        test_size  = float(split_cfg.get("test_size", 0.2))
        random_state = int(split_cfg.get("random_state", 42))
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[event_col],
            random_state=random_state,
        )
        train_df = train_df.copy()
        test_df  = test_df.copy()
        logger.info(
            "Split aleatorio — test_size: %.0f%% | train: %d | test: %d",
            test_size * 100, len(train_df), len(test_df),
        )

    else:
        raise ValueError(
            f"cfg['split']['type'] no reconocido: '{split_type}'. "
            "Valores válidos: 'temporal', 'random'."
        )

    logger.info(
        "Tasa %s → train: %.2f%%  |  test: %.2f%%",
        event_col,
        train_df[event_col].mean() * 100,
        test_df[event_col].mean()  * 100,
    )
    return train_df, test_df

# ──────────────────────────────────────────────────────────
#  FUNCION PRINCIPAL

def load_raw_data(
    cfg: dict,
    use_case: str,
    raw_path_override: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Carga los datos crudos producidos por Cargar_datos.py.

    Recibe cfg ya resuelto por resolve_cfg(). La ruta se toma de
    cfg["paths"]["raw_data"] (que ya fue fusionada desde el use_case).

    Fallback legacy: cfg["paths"]["base_data_csv"] para compatibilidad
    con ejecuciones que apuntan directamente al CSV original.

    Soporta .parquet y .csv según extensión.
    """
    if raw_path_override is not None:
        raw_path = Path(raw_path_override)
    else:
        raw_str = cfg["paths"].get("raw_data") or cfg["paths"].get("base_data_csv")
        if not raw_str:
            raise FileNotFoundError(
                "No se encontró 'paths.raw_data' ni 'paths.base_data_csv' en config.json."
            )
        raw_path = Path(raw_str)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Archivo de datos no encontrado: {raw_path}\n"
            f"Ejecuta primero: python src/Cargar_datos.py --use-case {use_case}"
        )

    ext = raw_path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(raw_path)
    elif ext == ".csv":
        df = pd.read_csv(raw_path)
    else:
        raise ValueError(f"Formato no soportado: '{ext}'. Usar .parquet o .csv.")

    logger.info("Datos cargados: %s  |  shape: %s", raw_path, df.shape)
    return df


def build_features(
    use_case: str = "scoring_mora",
    raw_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    return_dataframe: bool = False,
    return_base: bool = False,
) -> tuple:
    """
    Ejecuta el pipeline completo de ingeniería de features.

    Resuelve la configuración efectiva del caso de uso con resolve_cfg(),
    de modo que cada use_case puede tener su propio feature_engineering,
    target, split y artifacts_dir sin conflicto con otros use_cases.

    Flujo libre de data leakage:
      1. Cargar config y resolver cfg efectivo para use_case
      2. Cargar datos crudos (.parquet o .csv)
      3. pipeline_stateless.fit_transform(df_completo)  [pasos sin estado]
      4. split (temporal o random según cfg["split"]["type"])
      5. pipeline_base.fit_transform(X_train) / transform(X_test)  [solo train]
      6. pipeline_ml.fit_transform(X_train) / transform(X_test)    [solo train]

    Args:
        use_case:         Nombre del caso de uso en config.json > use_cases.
        raw_path:         Ruta explícita al archivo de datos (sobreescribe config).
        config_path:      Ruta al config.json (opcional).
        return_dataframe: True → DataFrames con nombres de columna.
                          False → arrays numpy (default).
        return_base:      True → retorna 8 elementos incluyendo X_train_base y
                          X_test_base (DataFrames pre-scaling, escala original).
                          Necesario para el modelo heurístico, que aplica
                          thresholds en la escala original de las variables.
                          False → retorna 6 elementos (comportamiento default,
                          compatible con versiones anteriores).

    Returns (return_base=False, default):
        X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base

    Returns (return_base=True):
        X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base,
        X_train_base, X_test_base
        donde X_train_base / X_test_base son DataFrames en escala original
        (salida de pipeline_base, antes de StandardScaler / OHE).
    """
    logger.info("=" * 60)
    logger.info("INICIO  ft_engineering.py — build_features() | use_case: %s", use_case)
    logger.info("=" * 60)

    # 1. Config global + resolución por use_case
    cfg_global, repo_root = load_config(config_path)
    cfg = resolve_cfg(cfg_global, use_case)   # cfg efectivo: global + overrides del use_case

    date_col  = cfg["split"]["date_col"]
    label_col = cfg["target"]["label_col"]
    event_col = cfg["target"]["event_col"]

    # 2. Carga de datos (Parquet desde Cargar_datos.py, o CSV legacy)
    df = load_raw_data(cfg, use_case=use_case, raw_path_override=raw_path)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce", utc=True).dt.tz_localize(None)
    df[event_col] = (df[label_col] == 0).astype(int)

    logger.info("Tasa de %s: %.2f%%  (%d positivos / %d total)",
                event_col, df[event_col].mean() * 100, df[event_col].sum(), len(df))

    # 3. Pasos SIN estado (cfg ya resuelto → builders usan features del use_case)
    pipeline_stateless = build_pipeline_stateless(cfg)
    df_pre = pipeline_stateless.fit_transform(df)

    # 4. Split (temporal o random según cfg["split"]["type"])
    train_df, test_df = temporal_split(df_pre, cfg, date_col)

    # 5. Separar X / y
    y_train     = train_df[event_col].values
    y_test      = test_df[event_col].values
    X_train_raw = train_df.drop(columns=[event_col], errors="ignore")
    X_test_raw  = test_df.drop(columns=[event_col],  errors="ignore")

    # 6. Pasos CON estado: fit SOLO sobre train
    pipeline_base = build_pipeline_base(cfg)
    X_train_base = pipeline_base.fit_transform(X_train_raw)
    X_test_base  = apply_pipeline_steps(pipeline_base, X_test_raw)

    # Conservar como DataFrames con nombres de columna para el modelo heurístico
    # (necesita escala original; pipeline_ml aplica StandardScaler encima)
    if not isinstance(X_train_base, pd.DataFrame):
        X_train_base = pd.DataFrame(X_train_base)
    if not isinstance(X_test_base, pd.DataFrame):
        X_test_base = pd.DataFrame(X_test_base)

    _validate_columns(X_train_base, cfg)

    # 7. Pipeline ML: fit SOLO sobre train
    pipeline_ml = build_pipeline_ml(cfg)
    X_train = pipeline_ml.fit_transform(X_train_base)
    X_test  = pipeline_ml.transform(X_test_base)

    try:
        feature_names = pipeline_ml.named_steps["preprocessor"].get_feature_names_out()
        pipeline_ml.feature_names_out_ = feature_names
    except Exception as exc:
        logger.warning("No se pudieron obtener feature names: %s", exc)
        feature_names = None

    logger.info(
        "X_train: %s  |  X_test: %s  |  %s train: %.2f%%  |  %s test: %.2f%%",
        X_train.shape, X_test.shape,
        event_col, y_train.mean() * 100, event_col, y_test.mean() * 100,
    )
    if feature_names is not None:
        logger.info("Features generadas: %d", len(feature_names))

    if return_dataframe and feature_names is not None:
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test  = pd.DataFrame(X_test,  columns=feature_names)

    logger.info("=" * 60)
    logger.info("FIN     ft_engineering.py — build_features() | use_case: %s", use_case)
    logger.info("=" * 60)

    if return_base:
        # X_train_base / X_test_base: DataFrames en escala original (pre-pipeline_ml).
        # Son la salida de pipeline_base: imputados, winsorizados y con columnas de
        # leakage eliminadas, pero SIN StandardScaler ni encoding.
        # Indispensable para el modelo heurístico, cuyos thresholds están en la
        # escala interpretable original (ej. puntaje_datacredito < 760).
        return X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base, \
               X_train_base, X_test_base

    return X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base

def _validate_columns(df: pd.DataFrame, cfg: dict) -> None:
    """Valida columnas del cfg ya resuelto — aplica al use_case correcto."""
    fe_cfg   = cfg["feature_engineering"]
    expected = (
        fe_cfg["numeric_cols"]
        + fe_cfg["categorical_cols"]
        + list(fe_cfg["ordinal_cols"].keys())
    )
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas del config no encontradas en el DataFrame:\n  {missing}\n"
            f"Disponibles: {sorted(df.columns.tolist())}"
        )
    logger.info("Validacion de columnas OK — %d features presentes.", len(expected))

def load_features_from_cache(
    use_case: str = "scoring_mora",
    config_path: Optional[Path] = None,
) -> tuple:
    """
    Carga los artefactos de features pre-computados por ft_engineering.py.

    Devuelve la misma tupla de 8 elementos que build_features(return_base=True):
        X_train, X_test, y_train, y_test,
        pipeline_ml, pipeline_base,
        X_train_base, X_test_base

    Lanza FileNotFoundError si los artefactos no existen (ejecuta ft_engineering
    primero).
    """
    cfg_global, _ = load_config(config_path)
    cfg = resolve_cfg(cfg_global, use_case)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    event_col = cfg["target"]["event_col"]

    required = [
        artifacts_dir / "X_train.csv",
        artifacts_dir / "X_test.csv",
        artifacts_dir / "X_train_base.csv",
        artifacts_dir / "X_test_base.csv",
        artifacts_dir / "y_train.csv",
        artifacts_dir / "y_test.csv",
        artifacts_dir / "pipeline_ml.pkl",
        artifacts_dir / "pipeline_base.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Caché de features incompleta. Ejecuta ft_engineering.py primero.\n"
            f"Faltan: {missing}"
        )

    X_train      = pd.read_csv(artifacts_dir / "X_train.csv")
    X_test       = pd.read_csv(artifacts_dir / "X_test.csv")
    X_train_base = pd.read_csv(artifacts_dir / "X_train_base.csv")
    X_test_base  = pd.read_csv(artifacts_dir / "X_test_base.csv")
    y_train      = pd.read_csv(artifacts_dir / "y_train.csv")[event_col].values
    y_test       = pd.read_csv(artifacts_dir / "y_test.csv")[event_col].values

    import joblib as _joblib
    pipeline_ml   = _joblib.load(artifacts_dir / "pipeline_ml.pkl")
    pipeline_base = _joblib.load(artifacts_dir / "pipeline_base.pkl")

    logger.info(
        "Features cargadas desde caché: %s  |  X_train=%s  X_test=%s",
        artifacts_dir, X_train.shape, X_test.shape,
    )
    return X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base, X_train_base, X_test_base


# ──────────────────────────────────────────────────────────
#  EJECUCION STANDALONE

if __name__ == "__main__":
    import argparse
    import inspect
    import sys as _sys

    # Fijar __module__ de las clases antes de hacer pickle, para que se
    # serialicen como 'ft_engineering.Clase' en lugar de '__main__.Clase'.
    # Así los pkl son cargables por cualquier script que importe ft_engineering.
    _this_mod = _sys.modules[__name__]
    for _cls_name, _cls in inspect.getmembers(_this_mod, inspect.isclass):
        if _cls.__module__ == "__main__":
            _cls.__module__ = "ft_engineering"
    _sys.modules.setdefault("ft_engineering", _this_mod)

    parser = argparse.ArgumentParser(description="Feature engineering — pipeline MLOps.")
    parser.add_argument(
        "--use-case", type=str, dest="use_case", default="scoring_mora",
        help="Caso de uso en config.json > use_cases (default: scoring_mora).",
    )
    parser.add_argument(
        "--raw-path", type=str, dest="raw_path", default=None,
        help="Ruta explícita al archivo de datos (sobreescribe config).",
    )
    args = parser.parse_args()

    # artifacts_dir resuelto por use_case → dos use_cases no se sobreescriben
    cfg_main, _ = load_config()
    cfg_resolved = resolve_cfg(cfg_main, args.use_case)
    artifacts_dir = Path(cfg_resolved["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (X_train, X_test, y_train, y_test,
     pipeline_ml, pipeline_base,
     X_train_base, X_test_base) = build_features(
        use_case=args.use_case,
        raw_path=Path(args.raw_path) if args.raw_path else None,
        return_dataframe=True,
        return_base=True,
    )

    X_train.to_csv(artifacts_dir / "X_train.csv", index=False)
    X_test.to_csv(artifacts_dir  / "X_test.csv",  index=False)
    X_train_base.to_csv(artifacts_dir / "X_train_base.csv", index=False)
    X_test_base.to_csv(artifacts_dir  / "X_test_base.csv",  index=False)
    event_col = cfg_resolved["target"]["event_col"]
    pd.Series(y_train, name=event_col).to_csv(artifacts_dir / "y_train.csv", index=False)
    pd.Series(y_test,  name=event_col).to_csv(artifacts_dir / "y_test.csv",  index=False)

    with open(artifacts_dir / "pipeline_ml.pkl", "wb") as fh:
        pickle.dump(pipeline_ml, fh)

    with open(artifacts_dir / "pipeline_base.pkl", "wb") as fh:
        pickle.dump(pipeline_base, fh)

    logger.info("Artefactos guardados en: %s", artifacts_dir)
    logger.info("  X_train.csv  (%d filas, %d cols)", *X_train.shape)
    logger.info("  X_test.csv   (%d filas, %d cols)", *X_test.shape)
    logger.info("  y_train.csv  (%d registros)",      len(y_train))
    logger.info("  y_test.csv   (%d registros)",      len(y_test))
    logger.info("  pipeline_ml.pkl")
    logger.info("  pipeline_base.pkl  [ajustado solo sobre train]")

# ──────────────────────────────────────────────────────────