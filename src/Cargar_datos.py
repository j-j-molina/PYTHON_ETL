"""
Cargar_datos.py  — v3
---------------------
Ingesta genérica y escalable para el pipeline MLOps.

Soporta múltiples fuentes de datos configurables desde config.json sin
modificar código. Para agregar una nueva fuente:
  1. Subclasear DataSource e implementar fetch().
  2. Registrar el tipo en _SOURCE_REGISTRY.
  3. En config.json: "source": {"type": "mi_fuente", ...params...}

Fuentes incluidas:
  - bigquery  → Google BigQuery (productivo)
  - csv       → CSV local (desarrollo/pruebas)
  - parquet   → Parquet local (desarrollo/pruebas)

Uso
---
    python src/Cargar_datos.py --use-case scoring_mora
    python src/Cargar_datos.py --use-case scoring_mora --since 2025-06-01
    python src/Cargar_datos.py --use-case scoring_mora --limit 500
    python src/Cargar_datos.py --list-cases
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Logging — nivel configurable por env var o config
# ---------------------------------------------------------------------------
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Cargar_datos")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "src" / "config.json"


# ---------------------------------------------------------------------------
# Capa de abstracción: fuentes de datos
# ---------------------------------------------------------------------------

class DataSource(ABC):
    """
    Interfaz común para todas las fuentes de datos del pipeline.
    Subclases implementan fetch(); el resto del pipeline no conoce
    el tipo concreto de la fuente.
    """

    @abstractmethod
    def fetch(
        self,
        columns: list[str],
        where_clause: str = "",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Extrae datos y los retorna como DataFrame.
        columns:      columnas a seleccionar (nunca SELECT *).
        where_clause: cláusula WHERE preformateada (puede ser "").
        limit:        número máximo de filas (None = sin límite).
        """

    def describe(self) -> str:
        return self.__class__.__name__


class BigQuerySource(DataSource):
    """
    Fuente Google BigQuery.
    Autenticación: GOOGLE_APPLICATION_CREDENTIALS → ADC → automático en GCP.
    config.json: {"type":"bigquery","project":"...","dataset":"...","table":"..."}
    """

    def __init__(self, project: str, dataset: str, table: str):
        self.project = project
        self.dataset = dataset
        self.table   = table

    def describe(self) -> str:
        return f"BigQuery({self.project}.{self.dataset}.{self.table})"

    def fetch(
        self,
        columns: list[str],
        where_clause: str = "",
        limit: int | None = None,
    ) -> pd.DataFrame:
        try:
            from google.cloud import bigquery
            from google.cloud.exceptions import GoogleCloudError
        except ImportError as exc:
            raise ImportError(
                "google-cloud-bigquery no está instalado. "
                "Ejecuta: pip install google-cloud-bigquery"
            ) from exc

        full_table    = f"`{self.project}.{self.dataset}.{self.table}`"
        select_clause = ", ".join(columns)
        query         = f"SELECT {select_clause}\nFROM {full_table}"
        if where_clause:
            query += f"\n{where_clause}"
        if limit:
            query += f"\nLIMIT {limit}"

        logger.info(
            "Conectando a BigQuery — %s | columnas: %d | limit: %s",
            self.describe(), len(columns),
            limit if limit else "sin límite",
        )
        logger.info("Query:\n%s", query)

        try:
            client = bigquery.Client(project=self.project)
            df = client.query(query).to_dataframe()
            logger.info("Ingesta exitosa — %d filas, %d columnas.", *df.shape)
            return df
        except Exception as exc:
            logger.error("Error en BigQuery fetch: %s", exc)
            raise


class CSVSource(DataSource):
    """
    Fuente CSV local — ideal para desarrollo y pruebas.
    config.json: {"type":"csv","path":"data/raw/dataset.csv","sep":",","encoding":"utf-8"}
    Nota: where_clause e ingesta incremental no se aplican; se carga todo el archivo.
    """

    def __init__(self, path: str, sep: str = ",", encoding: str = "utf-8"):
        self.path     = Path(path)
        self.sep      = sep
        self.encoding = encoding

    def describe(self) -> str:
        return f"CSV({self.path})"

    def fetch(
        self,
        columns: list[str],
        where_clause: str = "",
        limit: int | None = None,
    ) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {self.path}")
        df = pd.read_csv(
            self.path,
            usecols=columns,
            encoding=self.encoding,
            sep=self.sep,
            nrows=limit,
        )
        logger.info("CSV cargado: %s | shape: %s", self.path, df.shape)
        if where_clause:
            logger.warning("CSVSource: where_clause ignorado (no aplica a archivos locales).")
        return df


class ParquetSource(DataSource):
    """
    Fuente Parquet local — ideal para desarrollo y pruebas.
    config.json: {"type":"parquet","path":"data/raw/dataset.parquet"}
    Nota: where_clause e ingesta incremental no se aplican; se carga todo el archivo.
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def describe(self) -> str:
        return f"Parquet({self.path})"

    def fetch(
        self,
        columns: list[str],
        where_clause: str = "",
        limit: int | None = None,
    ) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet no encontrado: {self.path}")
        df = pd.read_parquet(self.path, columns=columns)
        if limit:
            df = df.head(limit)
        logger.info("Parquet cargado: %s | shape: %s", self.path, df.shape)
        if where_clause:
            logger.warning("ParquetSource: where_clause ignorado (no aplica a archivos locales).")
        return df


# Registry: agrega nuevas fuentes aquí sin tocar ningún otro código
_SOURCE_REGISTRY: dict[str, type[DataSource]] = {
    "bigquery": BigQuerySource,
    "csv":      CSVSource,
    "parquet":  ParquetSource,
}


def get_data_source(source_cfg: dict[str, Any]) -> DataSource:
    """
    Factory: instancia el DataSource correcto desde la sección source del config.

    Ejemplo:
        source = get_data_source(uc_config["source"])
        df = source.fetch(columns, where_clause, limit)
    """
    source_type = source_cfg.get("type")
    if not source_type:
        raise KeyError(
            "Falta 'type' en la configuración de la fuente. "
            f"Tipos disponibles: {list(_SOURCE_REGISTRY.keys())}"
        )

    source_cls = _SOURCE_REGISTRY.get(source_type.lower())
    if source_cls is None:
        raise NotImplementedError(
            f"Fuente '{source_type}' no implementada. "
            f"Disponibles: {list(_SOURCE_REGISTRY.keys())}. "
            "Para agregar una nueva: subclasea DataSource, implementa fetch() "
            "y regístrala en _SOURCE_REGISTRY en Cargar_datos.py."
        )

    params = {k: v for k, v in source_cfg.items() if k not in ("type", "dev_limit")}
    instance = source_cls(**params)
    logger.info("DataSource: %s", instance.describe())
    return instance


def register_source(type_name: str, source_cls: type[DataSource]) -> None:
    """
    Registra una implementación de DataSource en tiempo de ejecución.
    Útil para fuentes personalizadas sin modificar este archivo.

    Ejemplo:
        from Cargar_datos import register_source, DataSource

        class MyS3Source(DataSource):
            def fetch(self, columns, where_clause="", limit=None): ...

        register_source("s3", MyS3Source)
    """
    _SOURCE_REGISTRY[type_name.lower()] = source_cls
    logger.info("DataSource registrado: '%s' → %s", type_name, source_cls.__name__)


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró config.json en: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_use_case_config(config: dict[str, Any], use_case: str) -> dict[str, Any]:
    use_cases = config.get("use_cases", {})
    if not use_cases:
        raise KeyError(
            "La sección 'use_cases' no existe en config.json. "
            "Agrega al menos un caso de uso antes de ejecutar la ingesta."
        )
    if use_case not in use_cases:
        available = ", ".join(use_cases.keys())
        raise KeyError(
            f"Caso de uso '{use_case}' no encontrado en config.json. "
            f"Disponibles: {available}"
        )
    return use_cases[use_case]


def get_source_config(uc_config: dict[str, Any]) -> dict[str, Any]:
    """
    Resuelve la configuración de la fuente de datos.
    Soporta formato nuevo (source.type) y legado (bigquery) para compatibilidad.
    """
    if "source" in uc_config:
        return uc_config["source"]

    if "bigquery" in uc_config:
        bq = uc_config["bigquery"]
        return {
            "type":      "bigquery",
            "project":   bq["project"],
            "dataset":   bq["dataset"],
            "table":     bq["table"],
            "dev_limit": bq.get("dev_limit"),
        }

    raise KeyError(
        "No se encontró ni 'source' ni 'bigquery' en la configuración del caso de uso."
    )


# ---------------------------------------------------------------------------
# Estado incremental
# ---------------------------------------------------------------------------
def get_state_path(uc_config: dict[str, Any], use_case: str) -> Path:
    state_path = (
        uc_config.get("paths", {}).get("state_file")
        or f"data/state/{use_case}_ingestion_state.json"
    )
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Query incremental
# ---------------------------------------------------------------------------
def _sql_quote(value: Any) -> str:
    """Escapa un valor para incluirlo de forma segura en una cláusula WHERE."""
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    value = str(value).replace("'", "\\'")
    return f"'{value}'"


def get_select_columns(uc_config: dict[str, Any]) -> list[str]:
    ingestion_cfg = uc_config.get("ingestion", {})
    cols = ingestion_cfg.get("select_columns") or ingestion_cfg.get("expected_columns", [])
    if not cols:
        raise ValueError(
            "Debes definir 'ingestion.select_columns' o 'ingestion.expected_columns' "
            "en la configuración del caso de uso. No se usa SELECT *."
        )
    return cols


def build_incremental_where(
    uc_config: dict[str, Any],
    state: dict[str, Any],
    since: str | None = None,
) -> str:
    ingestion_cfg    = uc_config.get("ingestion", {})
    incremental_key  = ingestion_cfg.get("incremental_key")
    incremental_start = ingestion_cfg.get("incremental_start")
    lookback_days    = ingestion_cfg.get("lookback_days", 0)

    if not incremental_key:
        return ""

    effective_since = since or state.get("last_max_incremental_value") or incremental_start
    if not effective_since:
        return ""

    try:
        dt = datetime.fromisoformat(str(effective_since))
        effective_since = (dt - timedelta(days=lookback_days)).isoformat()
    except ValueError:
        pass

    return f"WHERE {incremental_key} >= {_sql_quote(effective_since)}"


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame, uc_config: dict[str, Any], use_case: str) -> None:
    if df.empty:
        raise ValueError(
            f"[{use_case}] El DataFrame está vacío. "
            "Verificar la tabla de origen, los filtros incrementales o la fecha de corte."
        )

    ingestion_cfg = uc_config.get("ingestion", {})
    expected_cols = ingestion_cfg.get("expected_columns", [])
    if expected_cols:
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"[{use_case}] Columnas faltantes en la extracción: {missing}\n"
                "Revisar 'select_columns' o el esquema de la tabla."
            )

    unique_key = ingestion_cfg.get("unique_key")
    if unique_key and unique_key in df.columns:
        dup_count = df.duplicated(subset=[unique_key]).sum()
        if dup_count > 0:
            logger.warning(
                "[%s] %d duplicados en unique_key='%s'. Revisar deduplicación upstream.",
                use_case, dup_count, unique_key,
            )

    high_null = df.isnull().mean()
    high_null = high_null[high_null > 0.5]
    if not high_null.empty:
        logger.warning(
            "[%s] Columnas con >50%% de nulos (revisar calidad en origen):\n%s",
            use_case, high_null.to_string(),
        )

    logger.info("[%s] Validación OK — %d filas, %d columnas.", use_case, *df.shape)


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def get_output_format(uc_config: dict[str, Any]) -> str:
    ingestion_cfg = uc_config.get("ingestion", {})
    fmt = ingestion_cfg.get("write_format")

    if not fmt:
        raw_path = uc_config.get("paths", {}).get("raw_data", "")
        ext = Path(raw_path).suffix.replace(".", "").lower()
        fmt = ext if ext in {"csv", "parquet"} else "parquet"

    fmt = fmt.lower()
    if fmt not in {"csv", "parquet"}:
        raise ValueError(f"write_format no soportado: '{fmt}'. Usar 'csv' o 'parquet'.")
    return fmt


def save_raw(df: pd.DataFrame, uc_config: dict[str, Any], use_case: str) -> Path:
    raw_path_str = uc_config.get("paths", {}).get("raw_data")
    if not raw_path_str:
        raise KeyError(
            f"[{use_case}] Falta 'paths.raw_data' en la configuración del caso de uso."
        )

    raw_path = Path(raw_path_str)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = get_output_format(uc_config)
    if fmt == "parquet":
        df.to_parquet(raw_path, index=False)
    else:
        df.to_csv(raw_path, index=False, encoding="utf-8")

    logger.info("[%s] Datos guardados en: %s  (formato: %s)", use_case, raw_path, fmt)
    return raw_path


# ---------------------------------------------------------------------------
# Estado incremental
# ---------------------------------------------------------------------------
def update_incremental_state(
    df: pd.DataFrame,
    uc_config: dict[str, Any],
    state_path: Path,
) -> None:
    incremental_key = uc_config.get("ingestion", {}).get("incremental_key")
    if not incremental_key or incremental_key not in df.columns or df.empty:
        return

    max_value = df[incremental_key].max()
    if pd.isna(max_value):
        return

    state = {"last_max_incremental_value": str(max_value)}
    save_state(state_path, state)
    logger.info("Estado incremental actualizado: %s = %s", incremental_key, max_value)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingesta genérica y escalable — pipeline MLOps.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--use-case", type=str, dest="use_case",
        help="Nombre del caso de uso definido en config.json > 'use_cases'.",
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help=(
            "Sobreescribe el punto de inicio incremental para esta ejecución.\n"
            "Formato ISO: 2025-06-01 o 2025-06-01T00:00:00"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limitar número de filas extraídas (útil para pruebas rápidas).",
    )
    parser.add_argument(
        "--list-cases", action="store_true",
        help="Listar casos de uso disponibles en config.json y salir.",
    )
    args = parser.parse_args()

    # Ajustar nivel de logging desde config si está disponible
    config = load_config()
    log_cfg = config.get("logging", {})
    level_str = os.environ.get("LOG_LEVEL") or log_cfg.get("level", "INFO")
    logging.getLogger().setLevel(getattr(logging, level_str.upper(), logging.INFO))

    if args.list_cases:
        cases = config.get("use_cases", {})
        if not cases:
            print("No hay casos de uso definidos en config.json.")
        else:
            print("Casos de uso disponibles:")
            for name, uc in cases.items():
                desc = uc.get("description", "sin descripción")
                src  = get_source_config(uc)
                print(f"  • {name}: {desc}")
                print(f"    → fuente: {src.get('type')} | {src.get('project','')}.{src.get('dataset','')}.{src.get('table', src.get('path',''))}")
        sys.exit(0)

    if not args.use_case:
        parser.error("Debes especificar --use-case <nombre> o usar --list-cases.")

    use_case = args.use_case
    logger.info("=== Cargar_datos v3 | caso de uso: %s ===", use_case)

    uc_config  = get_use_case_config(config, use_case)
    source_cfg = get_source_config(uc_config)
    limit      = args.limit if args.limit is not None else source_cfg.get("dev_limit")

    # Estado incremental
    state_path   = get_state_path(uc_config, use_case)
    state        = load_state(state_path)
    columns      = get_select_columns(uc_config)
    where_clause = build_incremental_where(uc_config, state, since=args.since)

    # Instanciar fuente de datos desde config — sin if-else por tipo
    source = get_data_source(source_cfg)
    df = source.fetch(columns, where_clause, limit)

    validate(df, uc_config, use_case)
    save_raw(df, uc_config, use_case)
    update_incremental_state(df, uc_config, state_path)

    logger.info("=== Cargar_datos completado | caso de uso: %s ===", use_case)


if __name__ == "__main__":
    main()
