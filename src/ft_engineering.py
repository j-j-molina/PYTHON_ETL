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
    """Sube hasta encontrar la raíz del repositorio (contiene mlops_pipeline/)."""
    p = start.resolve()
    for _ in range(8):
        if (p / "mlops_pipeline").exists():
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

# ──────────────────────────────────────────────────────────
#  TRANSFORMADORES PERSONALIZADOS
#  Cada clase hereda BaseEstimator + TransformerMixin para
#  ser compatible con Pipeline, GridSearchCV y cross_val_score
# ──────────────────────────────────────────────────────────

class CrearFeaturesDerivadas(BaseEstimator, TransformerMixin):
    """
    Crea las columnas derivadas que no vienen en el CSV base.
    Es IDEMPOTENTE: si la columna ya existe no la sobreescribe,
    por lo que funciona con el CSV de 24 o de 32 columnas.

    Paso SIN estado: fit() es un no-op. Seguro de aplicar sobre
    el dataset completo antes del split temporal.

    Columnas creadas:
      Ratios financieros (EDA: poder predictivo confirmado):
        ratio_cuota_salario           = cuota_pactada / salario_cliente
        ratio_capital_salario         = capital_prestado / salario_cliente
        ratio_otros_prestamos_salario = total_otros_prestamos / salario_cliente
        dti_aprox                     = (cuota_pactada + otros) / salario_cliente

      Sector crediticio:
        creditos_sector_total         = Financiero + Cooperativo + Real
        pct_creditos_sector*          = participacion por sector

      Temporales (desde fecha_prestamo):
        anio_prestamo, mes_prestamo, dia_semana_prestamo
        antiguedad_prestamo_dias      = (ref_date - fecha_prestamo).days

      Segmentacion de edad:
        edad_bucket  bins: 18-25, 26-35, 36-45, 46-55, 56-65, 66+
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, y=None):
        # Sin estado: no aprende ningún parámetro de los datos.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        derived  = self.cfg["feature_engineering"]["derived_features"]
        date_col = self.cfg["split"]["date_col"]

        ref_date   = pd.Timestamp(derived["antiguedad_ref_date"])
        age_bins   = derived["edad_bucket_bins"]
        age_labels = derived["edad_bucket_labels"]

        sal = pd.to_numeric(X["salario_cliente"], errors="coerce").replace(0, np.nan)

        # Ratios financieros (EDA: IV débil-medio, p-valor significativo)
        self._add(X, "ratio_cuota_salario",
                  X["cuota_pactada"] / sal)
        self._add(X, "ratio_capital_salario",
                  X["capital_prestado"] / sal)
        self._add(X, "ratio_otros_prestamos_salario",
                  X["total_otros_prestamos"] / sal)
        self._add(X, "dti_aprox",
                  (X["cuota_pactada"] + X["total_otros_prestamos"]) / sal)

        # Sector crediticio
        sec_total = (X["creditos_sectorFinanciero"]
                     + X["creditos_sectorCooperativo"]
                     + X["creditos_sectorReal"])
        self._add(X, "creditos_sector_total", sec_total)
        sec_denom = sec_total.replace(0, np.nan)
        self._add(X, "pct_creditos_sectorFinanciero",
                  X["creditos_sectorFinanciero"] / sec_denom)
        self._add(X, "pct_creditos_sectorCooperativo",
                  X["creditos_sectorCooperativo"] / sec_denom)
        self._add(X, "pct_creditos_sectorReal",
                  X["creditos_sectorReal"] / sec_denom)

        # Temporales
        self._add(X, "anio_prestamo",        X[date_col].dt.year)
        self._add(X, "mes_prestamo",         X[date_col].dt.month)
        self._add(X, "dia_semana_prestamo",  X[date_col].dt.day_name())
        self._add(X, "antiguedad_prestamo_dias",
                  (ref_date - X[date_col]).dt.days)

        # Edad bucket (OrdinalEncoder en pipeline_ml)
        self._add(
            X, "edad_bucket",
            pd.cut(X["edad_cliente"], bins=age_bins,
                   labels=age_labels, right=False)
              .astype(str).replace("nan", np.nan),
        )

        logger.info("CrearFeaturesDerivadas: %d columnas en el DataFrame", X.shape[1])
        return X

    @staticmethod
    def _add(df: pd.DataFrame, col: str, values) -> None:
        """Agrega la columna solo si no existe (idempotencia)."""
        if col not in df.columns:
            df[col] = values

class LimpiarTendenciaIngresos(BaseEstimator, TransformerMixin):
    """
    EDA detectó 58 registros con valores numéricos en tendencia_ingresos.
    Reemplaza cualquier valor fuera del catálogo válido por NaN para que
    el SimpleImputer del pipeline_ml los impute por moda.

    Paso SIN estado: fit() es un no-op. Seguro de aplicar sobre
    el dataset completo antes del split temporal.
    """

    def __init__(self, valid_values: list):
        self.valid_values = valid_values

    def fit(self, X: pd.DataFrame, y=None):
        # Sin estado: solo necesita el catálogo válido, no aprende de los datos.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        valid = set(self.valid_values)
        mask  = X["tendencia_ingresos"].notna() & ~X["tendencia_ingresos"].isin(valid)
        n = int(mask.sum())
        if n:
            X.loc[mask, "tendencia_ingresos"] = np.nan
            logger.info("LimpiarTendenciaIngresos: %d valores sucios → NaN", n)
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
        self.cols     = cols
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y=None):
        self.caps_: dict = {
            col: float(X[col].quantile(self.quantile))
            for col in self.cols if col in X.columns
        }
        logger.info("Winsorizar fit: caps aprendidos → %s", self.caps_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, cap in self.caps_.items():
            if col not in X.columns:
                continue
            n      = int((X[col] > cap).sum())
            X[col] = X[col].clip(upper=cap)
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
    Construye el pipeline de pasos sin estado desde config.json.

    Incluye:
      - CrearFeaturesDerivadas: ratios, temporales, edad_bucket
      - LimpiarTendenciaIngresos: reemplaza valores fuera del catálogo por NaN

    Al no aprender ningún parámetro de los datos, es seguro aplicarlo
    sobre el dataset completo antes del split temporal. No se serializa.
    """
    fe_cfg = cfg["feature_engineering"]
    return Pipeline(steps=[
        ("crear_features",    CrearFeaturesDerivadas(cfg=cfg)),
        ("limpiar_tendencia", LimpiarTendenciaIngresos(
            valid_values=fe_cfg["tendencia_valid_values"],
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
      - EliminarColumnas: leakage_cols (puntaje, Pago_atiempo) + fecha_prestamo

    Requiere fit SOLO sobre train. Se serializa como pipeline_base.pkl.
    """
    fe_cfg       = cfg["feature_engineering"]
    leakage_cols = fe_cfg["leakage_cols"]
    date_col     = cfg["split"]["date_col"]

    return Pipeline(steps=[
        ("imputacion",    ImputacionSegmentada(
            impute_map=fe_cfg["impute_grouped"],
        )),
        ("winsorizar",    Winsorizar(
            cols=fe_cfg["winsorize_cols"],
            quantile=float(fe_cfg["winsorize_quantile"]),
        )),
        ("eliminar_cols", EliminarColumnas(
            cols_to_drop=leakage_cols + [date_col],
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
        ("limpiar_tendencia", LimpiarTendenciaIngresos(
            valid_values=fe_cfg["tendencia_valid_values"],
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
) -> tuple:
    """
    Split TEMPORAL basado en fecha de originación.
    Corte: train <= sep-2025 / test > sep-2025.

    EDA: split aleatorio introduce leakage temporal por efecto de maduración.
    Los créditos más recientes tienen tasas de mora artificialmente bajas
    porque aún no han tenido tiempo de caer en mora.
    """
    cutoff        = cfg["split"]["train_cutoff"]
    cutoff_period = pd.Period(cutoff, freq="M")
    periods       = df[date_col].dt.to_period("M")
    train_df = df[periods <= cutoff_period].copy()
    test_df  = df[periods >  cutoff_period].copy()
    event_col = cfg["target"]["event_col"]
    logger.info(
        "Split temporal — cutoff: %s | train: %d (%.1f%%) | test: %d (%.1f%%)",
        cutoff,
        len(train_df), len(train_df) / len(df) * 100,
        len(test_df),  len(test_df)  / len(df) * 100,
    )
    logger.info(
        "Tasa mora → train: %.2f%%  |  test: %.2f%%",
        train_df[event_col].mean() * 100,
        test_df[event_col].mean()  * 100,
    )
    return train_df, test_df

# ──────────────────────────────────────────────────────────
#  FUNCION PRINCIPAL

def build_features(
    csv_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    return_dataframe: bool = False,
) -> tuple:
    """
    Ejecuta el pipeline completo de ingeniería de features.

    Flujo libre de data leakage:
      1. Cargar config.json y CSV
      2. Crear target 'mora'
      3. pipeline_stateless.fit_transform(df_completo)
         → CrearFeaturesDerivadas + LimpiarTendenciaIngresos
         → Seguro en todo el dataset (pasos sin estado)
      4. temporal_split()  →  train / test
      5. pipeline_base.fit_transform(X_train_raw)  [fit SOLO en train]
         pipeline_base.transform(X_test_raw)
         → ImputacionSegmentada + Winsorizar + EliminarColumnas
         → Caps y medianas aprendidos ÚNICAMENTE del train
      6. pipeline_ml.fit_transform(X_train_base)  [fit SOLO en train]
         pipeline_ml.transform(X_test_base)
         → ColumnTransformer: encoding + escalado

    Args:
        csv_path:         Ruta al CSV (opcional).
        config_path:      Ruta al config.json (opcional).
        return_dataframe: True → DataFrames con nombres de features.
                          False → arrays numpy (default).

    Returns:
        X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base
        donde pipeline_base está ajustado SOLO sobre train.
    """
    logger.info("=" * 60)
    logger.info("INICIO  ft_engineering.py — build_features()")
    logger.info("=" * 60)

    # 1. Config
    cfg, repo_root = load_config(config_path)
    date_col  = cfg["split"]["date_col"]
    label_col = cfg["target"]["label_col"]
    event_col = cfg["target"]["event_col"]

    # 2. Carga de datos
    candidates = [
        csv_path,
        repo_root / cfg["paths"]["base_data_csv"],
        repo_root / "mlops_pipeline" / "src" / cfg["paths"]["base_data_csv"],
        Path(cfg["paths"]["base_data_csv"]),
    ]
    csv_found = next((p for p in candidates if p is not None and Path(p).exists()), None)
    if csv_found is None:
        raise FileNotFoundError(f"No se encontro {cfg['paths']['base_data_csv']}.")

    df = pd.read_csv(csv_found)
    df[date_col]  = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")
    df[event_col] = (df[label_col] == 0).astype(int)

    logger.info("CSV cargado: %s  |  shape: %s", csv_found, df.shape)
    logger.info("Tasa de mora: %.2f%%  (%d mora / %d total)",
                df[event_col].mean() * 100, df[event_col].sum(), len(df))

    # 3. Pasos SIN estado sobre el dataset completo
    #    (CrearFeaturesDerivadas + LimpiarTendenciaIngresos no aprenden de los datos)
    pipeline_stateless = build_pipeline_stateless(cfg)
    df_pre = pipeline_stateless.fit_transform(df)
    # df_pre conserva: fecha_prestamo, mora, todas las features derivadas

    # 4. Split temporal (fecha_prestamo aún presente para el corte)
    train_df, test_df = temporal_split(df_pre, cfg, date_col)

    # 5. Separar X / y
    #    Mantener fecha_prestamo en X_*_raw porque pipeline_base la eliminará.
    y_train     = train_df[event_col].values
    y_test      = test_df[event_col].values
    X_train_raw = train_df.drop(columns=[event_col], errors="ignore")
    X_test_raw  = test_df.drop(columns=[event_col],  errors="ignore")

    # 6. Pasos CON estado: fit SOLO sobre train, transform sobre ambos
    #    Corrige el leakage de la versión anterior donde ImputacionSegmentada
    #    y Winsorizar aprendían sobre el dataset completo (train + test).
    pipeline_base = build_pipeline_base(cfg)
    X_train_base = pipeline_base.fit_transform(X_train_raw)  # aprende de train
    X_test_base  = pipeline_base.transform(X_test_raw)        # aplica parámetros de train

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
        "X_train: %s  |  X_test: %s  |  mora train: %.2f%%  |  mora test: %.2f%%",
        X_train.shape, X_test.shape,
        y_train.mean() * 100, y_test.mean() * 100,
    )
    if feature_names is not None:
        logger.info("Features generadas: %d", len(feature_names))

    if return_dataframe and feature_names is not None:
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test  = pd.DataFrame(X_test,  columns=feature_names)

    logger.info("=" * 60)
    logger.info("FIN     ft_engineering.py — build_features()")
    logger.info("=" * 60)

    return X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base

def _validate_columns(df: pd.DataFrame, cfg: dict) -> None:
    """Valida que todas las columnas del config estén en el DataFrame."""
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

# ──────────────────────────────────────────────────────────
#  EJECUCION STANDALONE

if __name__ == "__main__":
    cfg_main, repo_root_main = load_config()
    artifacts_dir = repo_root_main / cfg_main["paths"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, pipeline_ml, pipeline_base = build_features(
        return_dataframe=True
    )

    X_train.to_csv(artifacts_dir / "X_train.csv", index=False)
    X_test.to_csv(artifacts_dir  / "X_test.csv",  index=False)
    pd.Series(y_train, name="mora").to_csv(artifacts_dir / "y_train.csv", index=False)
    pd.Series(y_test,  name="mora").to_csv(artifacts_dir / "y_test.csv",  index=False)

    with open(artifacts_dir / "pipeline_ml.pkl", "wb") as fh:
        pickle.dump(pipeline_ml, fh)

    # pipeline_base: ImputacionSegmentada + Winsorizar + EliminarColumnas
    # Ajustado SOLO sobre train — caps y medianas libres de leakage.
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
"""
ft_engineering.py
--------------------
Primera componente del flujo MLOps — Ingeniería de Features.

Cada paso de transformación es una clase independiente que 
hereda de BaseEstimator + TransformerMixin, lo que hace
todo el pipeline compatible con GridSearchCV y cross_val_score.

Todo el comportamiento se controla desde config.json.

Estructura:
  ┌─────────────────────────────────────────────────────┐
  │              pipeline_basemodel                     │
  │  (se aplica sobre el DataFrame COMPLETO, pre-split) │
  ├─────────────────────────────────────────────────────┤
  │  CrearFeaturesDerivadas → LimpiarTendencia →        │
  │  ImputacionSegmentada → Winsorizar →                │
  │  EliminarColumnas                                   │
  └─────────────────────────────────────────────────────┘
              ↓  temporal_split()
  ┌─────────────────────────────────────────────────────┐
  │              pipeline_ml                            │
  │  (fit sobre train, transform sobre train+test) │
  ├─────────────────────────────────────────────────────┤
  │  ColumnTransformer:                                 │
  │    numeric   → SimpleImputer(median)+StandardScaler │
  │    categoric → SimpleImputer(mode)+OneHotEncoder    │
  │    cat_ord   → SimpleImputer(mode)+OrdinalEncoder   │
  └─────────────────────────────────────────────────────┘

Uso como módulo:
  from ft_engineering import build_features
  X_train, X_test, y_train, y_test, pipeline_ml = build_features()

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
    """Sube hasta encontrar la raíz del repositorio (contiene mlops_pipeline/)."""
    p = start.resolve()
    for _ in range(8):
        if (p / "mlops_pipeline").exists():
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

# ──────────────────────────────────────────────────────────
#  TRANSFORMADORES PERSONALIZADOS
#  Cada clase hereda BaseEstimator + TransformerMixin para
#  ser compatible con Pipeline, GridSearchCV y cross_val_score
# ──────────────────────────────────────────────────────────

class CrearFeaturesDerivadas(BaseEstimator, TransformerMixin):
    """
    Crea las columnas derivadas que no vienen en el CSV base.
    Es IDEMPOTENTE: si la columna ya existe no la sobreescribe,
    por lo que funciona con el CSV de 24 o de 32 columnas.

    Columnas creadas:
      Ratios financieros (EDA: poder predictivo confirmado):
        ratio_cuota_salario           = cuota_pactada / salario_cliente
        ratio_capital_salario         = capital_prestado / salario_cliente
        ratio_otros_prestamos_salario = total_otros_prestamos / salario_cliente
        dti_aprox                     = (cuota_pactada + otros) / salario_cliente

      Sector crediticio:
        creditos_sector_total         = Financiero + Cooperativo + Real
        pct_creditos_sector*          = participacion por sector

      Temporales (desde fecha_prestamo):
        anio_prestamo, mes_prestamo, dia_semana_prestamo
        antiguedad_prestamo_dias      = (ref_date - fecha_prestamo).days

      Segmentacion de edad:
        edad_bucket  bins: 18-25, 26-35, 36-45, 46-55, 56-65, 66+
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        derived  = self.cfg["feature_engineering"]["derived_features"]
        date_col = self.cfg["split"]["date_col"]

        ref_date   = pd.Timestamp(derived["antiguedad_ref_date"])
        age_bins   = derived["edad_bucket_bins"]
        age_labels = derived["edad_bucket_labels"]

        sal = pd.to_numeric(X["salario_cliente"], errors="coerce").replace(0, np.nan)

        # Ratios financieros
        self._add(X, "ratio_cuota_salario",
                  X["cuota_pactada"] / sal)
        self._add(X, "ratio_capital_salario",
                  X["capital_prestado"] / sal)
        self._add(X, "ratio_otros_prestamos_salario",
                  X["total_otros_prestamos"] / sal)
        self._add(X, "dti_aprox",
                  (X["cuota_pactada"] + X["total_otros_prestamos"]) / sal)

        # Sector crediticio
        sec_total = (X["creditos_sectorFinanciero"]
                     + X["creditos_sectorCooperativo"]
                     + X["creditos_sectorReal"])
        self._add(X, "creditos_sector_total", sec_total)
        sec_denom = sec_total.replace(0, np.nan)
        self._add(X, "pct_creditos_sectorFinanciero",
                  X["creditos_sectorFinanciero"] / sec_denom)
        self._add(X, "pct_creditos_sectorCooperativo",
                  X["creditos_sectorCooperativo"] / sec_denom)
        self._add(X, "pct_creditos_sectorReal",
                  X["creditos_sectorReal"] / sec_denom)

        # Temporales
        self._add(X, "anio_prestamo",        X[date_col].dt.year)
        self._add(X, "mes_prestamo",         X[date_col].dt.month)
        self._add(X, "dia_semana_prestamo",  X[date_col].dt.day_name())
        self._add(X, "antiguedad_prestamo_dias",
                  (ref_date - X[date_col]).dt.days)

        # Edad bucket
        self._add(
            X, "edad_bucket",
            pd.cut(X["edad_cliente"], bins=age_bins,
                   labels=age_labels, right=False)
              .astype(str).replace("nan", np.nan),
        )

        logger.info("CrearFeaturesDerivadas: %d columnas en el DataFrame", X.shape[1])
        return X

    @staticmethod
    def _add(df: pd.DataFrame, col: str, values) -> None:
        """Agrega la columna solo si no existe (idempotencia)."""
        if col not in df.columns:
            df[col] = values

class LimpiarTendenciaIngresos(BaseEstimator, TransformerMixin):
    """
    EDA detecto 58 registros con valores numericos en tendencia_ingresos.
    Reemplaza cualquier valor fuera del catalogo valido por NaN para que
    el SimpleImputer del pipeline_ml los impute por moda.
    """

    def __init__(self, valid_values: list):
        self.valid_values = valid_values

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        valid = set(self.valid_values)
        mask  = X["tendencia_ingresos"].notna() & ~X["tendencia_ingresos"].isin(valid)
        n = int(mask.sum())
        if n:
            X.loc[mask, "tendencia_ingresos"] = np.nan
            logger.info("LimpiarTendenciaIngresos: %d valores sucios → NaN", n)
        return X

class ImputacionSegmentada(BaseEstimator, TransformerMixin):
    """
    EDA: promedio_ingresos_datacredito tiene 27.2% de nulos y la mediana
    difiere por tipo_laboral → imputacion segmentada.

    fit():      aprende las medianas por grupo SOLO sobre train.
    transform(): aplica esas medianas, con fallback a mediana global.

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
    Aplica cap superior (winsorizacion) al percentil definido.

    fit():      aprende los caps sobre train (evita leakage).
    transform(): aplica clip() con esos caps a cualquier conjunto.

    EDA: outliers IQR relevantes en salario_cliente, capital_prestado,
    total_otros_prestamos y cuota_pactada.
    """

    def __init__(self, cols: list, quantile: float = 0.99):
        self.cols     = cols
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y=None):
        self.caps_: dict = {
            col: float(X[col].quantile(self.quantile))
            for col in self.cols if col in X.columns
        }
        logger.info("Winsorizar fit: caps aprendidos → %s", self.caps_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, cap in self.caps_.items():
            if col not in X.columns:
                continue
            n      = int((X[col] > cap).sum())
            X[col] = X[col].clip(upper=cap)
            logger.info("Winsorizar: %-30s cap=%.2f  (%d capeados)", col, cap, n)
        return X

class EliminarColumnas(BaseEstimator, TransformerMixin):
    """
    Elimina columnas por nombre.
    errors='ignore' evita errores si alguna columna ya no existe.
    Se usa al final del pipeline_basemodel para quitar leakage
    (puntaje, Pago_atiempo) y la columna de fecha.
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
#  PIPELINE BASE
#  Se aplica sobre el DataFrame COMPLETO (antes del split).
#  Incluye: creacion de features, limpieza, imputacion y
#  winsorizacion. NO incluye encoding ni escalado.
#
#  NOTA sobre el fit pre-split: ImputacionSegmentada y
#  Winsorizar aprenden sobre el dataset completo por
#  simplicidad. El split es TEMPORAL (futuro nunca
#  contamina el pasado), lo que mitiga el riesgo.
#  En produccion pura estos transformers deben fittearse
#  solo sobre train.
# ──────────────────────────────────────────────────────────

def build_pipeline_basemodel(cfg: dict) -> Pipeline:
    """Construye el pipeline base desde config.json."""
    fe_cfg       = cfg["feature_engineering"]
    leakage_cols = fe_cfg["leakage_cols"]
    date_col     = cfg["split"]["date_col"]
    event_col    = cfg["target"]["event_col"]

    # mora (event_col) se mantiene hasta despues del split para
    # poder separar X / y; se elimina en build_features() post-split.
    return Pipeline(steps=[
        ("crear_features",    CrearFeaturesDerivadas(cfg=cfg)),
        ("limpiar_tendencia", LimpiarTendenciaIngresos(
            valid_values=fe_cfg["tendencia_valid_values"],
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
) -> tuple:
    """
    Split TEMPORAL basado en fecha de originacion.
    Corte: train <= sep-2025 / test > sep-2025.
    EDA: split aleatorio introduce leakage temporal por efecto de maduracion.
    """
    cutoff        = cfg["split"]["train_cutoff"]
    cutoff_period = pd.Period(cutoff, freq="M")
    periods       = df[date_col].dt.to_period("M")
    train_df = df[periods <= cutoff_period].copy()
    test_df  = df[periods >  cutoff_period].copy()
    event_col = cfg["target"]["event_col"]
    logger.info(
        "Split temporal — cutoff: %s | train: %d (%.1f%%) | test: %d (%.1f%%)",
        cutoff,
        len(train_df), len(train_df) / len(df) * 100,
        len(test_df),  len(test_df)  / len(df) * 100,
    )
    logger.info(
        "Tasa mora → train: %.2f%%  |  test: %.2f%%",
        train_df[event_col].mean() * 100,
        test_df[event_col].mean()  * 100,
    )
    return train_df, test_df

# ──────────────────────────────────────────────────────────
#  FUNCION PRINCIPAL

def build_features(
    csv_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    return_dataframe: bool = False,
) -> tuple:
    """
    Ejecuta el pipeline completo de ingenieria de features.

    Flujo:
      1. Cargar config.json
      2. Cargar CSV y crear target 'mora'
      3. pipeline_basemodel.fit_transform(df_completo)
      4. Split temporal
      5. Separar X / y
      6. pipeline_ml.fit(X_train) → transform(X_train, X_test)

    Args:
        csv_path:         Ruta al CSV (opcional).
        config_path:      Ruta al config.json (opcional).
        return_dataframe: True → DataFrames. False → arrays numpy.

    Returns:
        X_train, X_test, y_train, y_test, pipeline_ml
    """
    logger.info("=" * 60)
    logger.info("INICIO  ft_engineering.py — build_features()")
    logger.info("=" * 60)

    # 1. Config
    cfg, repo_root = load_config(config_path)
    date_col  = cfg["split"]["date_col"]
    label_col = cfg["target"]["label_col"]
    event_col = cfg["target"]["event_col"]

    # 2. Carga de datos
    candidates = [
        csv_path,
        repo_root / cfg["paths"]["base_data_csv"],
        repo_root / "mlops_pipeline" / "src" / cfg["paths"]["base_data_csv"],
        Path(cfg["paths"]["base_data_csv"]),
    ]
    csv_found = next((p for p in candidates if p is not None and Path(p).exists()), None)
    if csv_found is None:
        raise FileNotFoundError(f"No se encontro {cfg['paths']['base_data_csv']}.")

    df = pd.read_csv(csv_found)
    df[date_col]  = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")
    df[event_col] = (df[label_col] == 0).astype(int)

    logger.info("CSV cargado: %s  |  shape: %s", csv_found, df.shape)
    logger.info("Tasa de mora: %.2f%%  (%d mora / %d total)",
                df[event_col].mean() * 100, df[event_col].sum(), len(df))

    # Guardar la fecha antes de que EliminarColumnas la elimine
    fecha_original = df[date_col].copy()

    # 3. Pipeline base sobre todo el dataset
    pipeline_base = build_pipeline_basemodel(cfg)
    df_clean = pipeline_base.fit_transform(df)

    # Reincorporar fecha solo para el split temporal
    df_clean[date_col] = fecha_original.values

    # 4. Split temporal
    train_df, test_df = temporal_split(df_clean, cfg, date_col)

    # 5. Separar X / y
    y_train = train_df[event_col].values
    y_test  = test_df[event_col].values
    X_train_raw = train_df.drop(columns=[event_col, date_col], errors="ignore")
    X_test_raw  = test_df.drop(columns=[event_col, date_col],  errors="ignore")

    _validate_columns(X_train_raw, cfg)

    # 6. Pipeline ML: fit sobre train
    pipeline_ml = build_pipeline_ml(cfg)
    X_train = pipeline_ml.fit_transform(X_train_raw)
    X_test  = pipeline_ml.transform(X_test_raw)

    try:
        feature_names = pipeline_ml.named_steps["preprocessor"].get_feature_names_out()
        pipeline_ml.feature_names_out_ = feature_names
    except Exception as exc:
        logger.warning("No se pudieron obtener feature names: %s", exc)
        feature_names = None

    logger.info(
        "X_train: %s  |  X_test: %s  |  mora train: %.2f%%  |  mora test: %.2f%%",
        X_train.shape, X_test.shape,
        y_train.mean() * 100, y_test.mean() * 100,
    )
    if feature_names is not None:
        logger.info("Features generadas: %d", len(feature_names))

    if return_dataframe and feature_names is not None:
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test  = pd.DataFrame(X_test,  columns=feature_names)
    
    logger.info("=" * 60)
    logger.info("FIN     ft_engineering.py — build_features()")
    logger.info("=" * 60)

    return X_train, X_test, y_train, y_test, pipeline_ml

def _validate_columns(df: pd.DataFrame, cfg: dict) -> None:
    """Valida que todas las columnas del config esten en el DataFrame."""
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

# ──────────────────────────────────────────────────────────
#  EJECUCION STANDALONE

if __name__ == "__main__":
    cfg_main, repo_root_main = load_config()
    artifacts_dir = repo_root_main / cfg_main["paths"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, pipeline_ml = build_features(
        return_dataframe=True
    )

    X_train.to_csv(artifacts_dir / "X_train.csv", index=False)
    X_test.to_csv(artifacts_dir  / "X_test.csv",  index=False)
    pd.Series(y_train, name="mora").to_csv(artifacts_dir / "y_train.csv", index=False)
    pd.Series(y_test,  name="mora").to_csv(artifacts_dir / "y_test.csv",  index=False)

    with open(artifacts_dir / "pipeline_ml.pkl", "wb") as fh:
        pickle.dump(pipeline_ml, fh)

    logger.info("Artefactos guardados en: %s", artifacts_dir)
    logger.info("  X_train.csv  (%d filas, %d cols)", *X_train.shape)
    logger.info("  X_test.csv   (%d filas, %d cols)", *X_test.shape)
    logger.info("  y_train.csv  (%d registros)",      len(y_train))
    logger.info("  y_test.csv   (%d registros)",      len(y_test))
    logger.info("  pipeline_ml.pkl")

# ──────────────────────────────────────────────────────────