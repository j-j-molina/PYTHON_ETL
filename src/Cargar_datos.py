"""
Cargar_datos.py  — v2
---------------------
Ingesta genérica y escalable desde BigQuery hacia el pipeline MLOps.
Reemplaza Cargar_datos.ipynb (carga desde CSV local no productivo).

Mejoras respecto a v1
---------------------
- SELECT explícito de columnas (no SELECT *).
- Ingesta incremental por columna de fecha, con estado persistido.
  El argumento --since sobreescribe el estado guardado puntualmente.
- Salida configurable: CSV o Parquet (por defecto Parquet).
- Detección de duplicados por clave única configurable.
- Compatibilidad con la sección 'bigquery' existente en config.json
  además de la nueva sección 'source', para no romper configs previos.

Uso
---
    # Extracción completa desde el inicio configurado
    python src/Cargar_datos.py --use-case scoring_mora

    # Sobreescribir fecha de inicio puntualmente
    python src/Cargar_datos.py --use-case scoring_mora --since 2025-06-01

    # Muestra reducida para pruebas rápidas
    python src/Cargar_datos.py --use-case scoring_mora --limit 500

    # Ver casos disponibles
    python src/Cargar_datos.py --list-cases
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Cargar_datos")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "src" / "config.json"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró config.json en: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_use_case_config(config: dict[str, Any], use_case: str) -> dict[str, Any]:
    """
    Extrae la configuración del caso de uso desde config.json.
    Falla con mensaje claro si el caso no existe.
    """
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

    Soporta dos formatos para compatibilidad:
      - Nuevo:   use_cases.<caso>.source  { type, project, dataset, table }
      - Legado:  use_cases.<caso>.bigquery { project, dataset, table }

    El campo 'type' permite preparar el terreno para otras fuentes
    (GCS, S3, SQL, etc.) sin cambiar el script.
    """
    if "source" in uc_config:
        return uc_config["source"]

    if "bigquery" in uc_config:
        bq = uc_config["bigquery"]
        return {
            "type": "bigquery",
            "project": bq["project"],
            "dataset": bq["dataset"],
            "table":   bq["table"],
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
# Query
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
    """
    Devuelve las columnas a seleccionar.

    Prioridad:
      1. ingestion.select_columns  → lista explícita (recomendado)
      2. ingestion.expected_columns → fallback para compatibilidad

    Nunca cae en SELECT * — si no hay columnas definidas, falla explícito.
    """
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
    """
    Construye la cláusula WHERE para ingesta incremental.

    El valor efectivo de inicio se resuelve en este orden:
      1. --since pasado por CLI  (sobreescribe todo, útil para backfills)
      2. last_max_incremental_value en el estado persistido
      3. ingestion.incremental_start del config (arranque inicial)

    lookback_days retrocede N días del punto de inicio para cubrir
    registros que podrían haber llegado tarde al DWH.

    Si no hay incremental_key definida, retorna string vacío
    y se hace una extracción completa.
    """
    ingestion_cfg    = uc_config.get("ingestion", {})
    incremental_key  = ingestion_cfg.get("incremental_key")
    incremental_start = ingestion_cfg.get("incremental_start")
    lookback_days    = ingestion_cfg.get("lookback_days", 0)

    if not incremental_key:
        return ""

    effective_since = since or state.get("last_max_incremental_value") or incremental_start
    if not effective_since:
        return ""

    # Aplicar lookback si el valor es parseable como fecha
    try:
        dt = datetime.fromisoformat(str(effective_since))
        effective_since = (dt - timedelta(days=lookback_days)).isoformat()
    except ValueError:
        pass  # Si no es fecha ISO, se usa tal cual

    return f"WHERE {incremental_key} >= {_sql_quote(effective_since)}"


def build_query(
    project: str,
    dataset: str,
    table: str,
    columns: list[str],
    where_clause: str = "",
    limit: int | None = None,
) -> str:
    """
    Construye la query de extracción.
    Separada de la conexión para que sea testeable de forma unitaria.
    """
    full_table    = f"`{project}.{dataset}.{table}`"
    select_clause = ", ".join(columns)
    query         = f"SELECT {select_clause}\nFROM {full_table}"

    if where_clause:
        query += f"\n{where_clause}"
    if limit:
        query += f"\nLIMIT {limit}"

    return query


# ---------------------------------------------------------------------------
# Ingesta
# ---------------------------------------------------------------------------
def fetch_from_bigquery(
    project: str,
    dataset: str,
    table: str,
    columns: list[str],
    where_clause: str = "",
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Extrae datos de BigQuery y los retorna como DataFrame.

    Autenticación (en orden de precedencia):
      1. Variable de entorno GOOGLE_APPLICATION_CREDENTIALS
         → JSON de Service Account (CI/CD, contenedores, GitHub Actions)
      2. Application Default Credentials (ADC)
         → `gcloud auth application-default login` en local
         → automático en Cloud Run, GCE, Vertex AI

    No se hardcodean credenciales en el código.
    """
    try:
        client = bigquery.Client(project=project)
        query  = build_query(project, dataset, table, columns, where_clause, limit)

        logger.info(
            "Conectando a BigQuery — %s.%s.%s | columnas: %d | limit: %s",
            project, dataset, table, len(columns),
            limit if limit else "sin límite",
        )
        logger.info("Query:\n%s", query)

        df = client.query(query).to_dataframe()
        logger.info("Ingesta exitosa — %d filas, %d columnas.", *df.shape)
        return df

    except GoogleCloudError as exc:
        logger.error("Error de BigQuery: %s", exc)
        raise
    except Exception as exc:
        logger.error("Error inesperado durante la ingesta: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame, uc_config: dict[str, Any], use_case: str) -> None:
    """
    Validación rápida antes de persistir.
    Falla explícito para no contaminar artefactos downstream.
    """
    if df.empty:
        raise ValueError(
            f"[{use_case}] El DataFrame está vacío. "
            "Verificar la tabla de origen, los filtros incrementales o la fecha de corte."
        )

    ingestion_cfg  = uc_config.get("ingestion", {})
    expected_cols  = ingestion_cfg.get("expected_columns", [])
    if expected_cols:
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"[{use_case}] Columnas faltantes en la extracción: {missing}\n"
                "Revisar 'select_columns' o el esquema de la tabla en BigQuery."
            )

    # Duplicados por clave única (no bloquea, pero avisa)
    unique_key = ingestion_cfg.get("unique_key")
    if unique_key and unique_key in df.columns:
        dup_count = df.duplicated(subset=[unique_key]).sum()
        if dup_count > 0:
            logger.warning(
                "[%s] %d duplicados en unique_key='%s'. Revisar deduplicación upstream.",
                use_case, dup_count, unique_key,
            )

    # Columnas con alta tasa de nulos (no bloquea, avisa)
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
    """
    Resuelve el formato de salida en este orden:
      1. ingestion.write_format en config
      2. Extensión del archivo en paths.raw_data
      3. Parquet por defecto
    """
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
    """
    Persiste el DataFrame en la ruta definida en paths.raw_data.

    Los pasos downstream (ft_engineering.py, comprension_eda.ipynb, etc.)
    leen desde esa ruta sin saber ni importarles el origen de los datos.
    """
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
# Actualización de estado incremental
# ---------------------------------------------------------------------------
def update_incremental_state(
    df: pd.DataFrame,
    uc_config: dict[str, Any],
    state_path: Path,
) -> None:
    """
    Persiste el valor máximo de la columna incremental para la próxima ejecución.
    Solo actúa si incremental_key está definida y presente en el DataFrame.
    """
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
        description="Ingesta genérica y escalable desde BigQuery — pipeline MLOps.",
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
            "Formato ISO: 2025-06-01 o 2025-06-01T00:00:00\n"
            "Útil para backfills puntuales sin alterar el estado persistido."
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

    # 1. Cargar config
    config = load_config()

    # 2. Modo informativo
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
                print(f"    → {src.get('project')}.{src.get('dataset')}.{src.get('table')}")
        sys.exit(0)

    if not args.use_case:
        parser.error("Debes especificar --use-case <nombre> o usar --list-cases.")

    use_case = args.use_case
    logger.info("=== Cargar_datos v2 | caso de uso: %s ===", use_case)

    # 3. Configuración del caso de uso
    uc_config   = get_use_case_config(config, use_case)
    source_cfg  = get_source_config(uc_config)

    if source_cfg.get("type") != "bigquery":
        raise NotImplementedError(
            f"Fuente no soportada aún: '{source_cfg.get('type')}'. "
            "Implementar un fetcher equivalente a fetch_from_bigquery()."
        )

    project = source_cfg["project"]
    dataset = source_cfg["dataset"]
    table   = source_cfg["table"]
    limit   = args.limit if args.limit is not None else source_cfg.get("dev_limit")

    # 4. Estado incremental
    state_path  = get_state_path(uc_config, use_case)
    state       = load_state(state_path)

    # 5. Columnas y filtro incremental
    columns      = get_select_columns(uc_config)
    where_clause = build_incremental_where(uc_config, state, since=args.since)

    # 6. Ingesta
    df = fetch_from_bigquery(project, dataset, table, columns, where_clause, limit)

    # 7. Validación
    validate(df, uc_config, use_case)

    # 8. Persistencia
    save_raw(df, uc_config, use_case)

    # 9. Actualizar estado incremental para la próxima ejecución
    update_incremental_state(df, uc_config, state_path)

    logger.info("=== Cargar_datos completado | caso de uso: %s ===", use_case)


if __name__ == "__main__":
    main()