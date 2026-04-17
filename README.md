# CDP_Credito - Pipeline MLOps de Scoring de Mora
# Autor: **Juan José Molina Zapata**

**Ciencia de Datos en Producción — Entregable 3**  
Docente: Juan Sebastián Parra Sánchez · Universidad Pontificia Bolivariana

---

## Descripción

Pipeline MLOps end-to-end para predecir la probabilidad de mora en créditos nuevos. El objetivo es generar un modelo predictivo que permita identificar clientes con alto riesgo de incumplimiento en el momento del desembolso, antes de que ocurra el evento.

El dataset contiene **10.763 créditos** con una tasa de mora del **4.75%** — fuertemente desbalanceado, lo que guía decisiones clave del pipeline: split temporal, threshold calibrado, class_weight y criterio de selección basado en CV.

---

## Estructura del Repositorio

```
mlops_pipeline/
├── src/
│   ├── Cargar_datos.py             # Carga de datos desde BigQuery o CSV local
│   ├── comprension_eda.ipynb       # EDA completo: IV, WoE, outliers, leakage
│   ├── ft_engineering.py           # Ingeniería de features + pipelines sklearn
│   ├── heuristic_model.py          # Modelo heurístico baseline (reglas IV del EDA)
│   ├── model_training.py           # Entrenamiento, CV con threshold calibrado, selección
│   ├── model_evaluation.py         # Reporte de métricas del modelo desplegado
│   ├── model_monitoring.py         # PSI + data drift por ventana temporal
│   ├── model_deploy.py             # Artefactos Docker + API Flask batch
│   └── config.json                 # Configuración centralizada de todo el pipeline
├── artifacts/                      # Modelos y pipelines serializados (generados)
├── reports/                        # Gráficos y reportes HTML/JSON (generados)
├── Dockerfile                      # Imagen de producción (generado por model_deploy.py)
├── requirements.txt                # Dependencias de desarrollo
├── requirements_deploy.txt         # Dependencias mínimas de producción
├── set_up.sh / set_up.bat          # Creación del entorno virtual
├── run_pipeline.sh / run_pipeline.bat  # Ejecución del pipeline completo
├── .gitignore
└── README.md
Base_de_datos.csv                   # Dataset (raíz del repo)
```

---

## Instalación

### Requisitos previos
- Python 3.10+
- Git

### Setup del entorno (Mac/Linux)

```bash
# Desde la raíz del repositorio (mlops_pipeline/)
bash set_up.sh
```

### Setup del entorno (Windows)

```bat
set_up.bat
```

Esto crea el entorno virtual `cdp_credito-venv`, instala `requirements.txt` y registra el kernel de Jupyter.

---

## Ejecución del Pipeline

```bash
# Mac/Linux — desde mlops_pipeline/
bash run_pipeline.sh

# Windows
run_pipeline.bat
```

El pipeline ejecuta los 6 pasos en orden, abortando si alguno falla:

| Paso | Script | Output principal |
|------|--------|-----------------|
| 1 | `ft_engineering.py` | `X_train/test.csv`, `pipeline_ml.pkl`, `pipeline_base.pkl` |
| 2 | `heuristic_model.py` | `heuristic_baseline.json`, gráficos CV |
| 3 | `model_training.py` | `best_model.joblib`, `best_model_meta.json`, gráficos comparativos |
| 4 | `model_evaluation.py` | `evaluation_report.html`, `evaluation_report.json` |
| 5 | `model_monitoring.py` | `drift_report.csv`, `monitoring_report.html` |
| 6 | `model_deploy.py` | `Dockerfile`, `requirements_deploy.txt`, `deploy_summary.json` |

---

## Diseño del Pipeline

### Feature Engineering (`ft_engineering.py`)

El pipeline está separado en tres capas para evitar data leakage:

```
pipeline_stateless  →  split temporal  →  pipeline_base (fit train)  →  pipeline_ml (fit train)
```

**`pipeline_stateless`** (pasos sin estado — aplicados sobre el dataset completo antes del split):
- `CrearFeaturesDerivadas`: ratios financieros, features temporales desde `fecha_prestamo`, `edad_bucket`
- `LimpiarTendenciaIngresos`: reemplaza los 58 valores numéricos sucios por NaN

**`pipeline_base`** (pasos con estado — ajustados **solo sobre train**):
- `ImputacionSegmentada`: medianas de `promedio_ingresos_datacredito` por `tipo_laboral` (EDA: 27.2% de nulos, medianas difieren por segmento)
- `Winsorizar`: caps al p99 para `salario_cliente`, `capital_prestado`, `total_otros_prestamos`, `cuota_pactada` (EDA: outliers IQR relevantes)
- `EliminarColumnas`: elimina leakage (`puntaje`, `Pago_atiempo`) y `fecha_prestamo`

**`pipeline_ml`** (ColumnTransformer — ajustado **solo sobre train**):
- Numéricas: `SimpleImputer(median)` + `StandardScaler`
- Categóricas nominales: `SimpleImputer(mode)` + `OneHotEncoder(drop='first')`
- Categóricas ordinales: `SimpleImputer(mode)` + `OrdinalEncoder` (orden definido en `config.json`)

**Split temporal:** train ≤ sep-2025 / test > sep-2025 (9.938 / 825 registros). El split aleatorio introduciría leakage temporal por efecto de maduración de cartera.

### Modelo Heurístico Baseline (`heuristic_model.py`)

Tres reglas derivadas del EDA (IV y pruebas t):

| Regla | Threshold | IV | p-valor |
|-------|-----------|-----|---------|
| `puntaje_datacredito < 760` | Media mora=749, al día=782 | 0.199 | <0.001 |
| `huella_consulta > 5` | Media mora=5.2, al día=4.2 | 0.146 | <0.001 |
| `plazo_meses > 12` | Media mora=12.5, al día=10.5 | 0.128 | <0.001 |

`mora = 1` si al menos 2 de 3 señales activas. ROC-AUC test = 0.418.

### Entrenamiento y Selección (`model_training.py`)

Modelos comparados: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.

**Threshold calibrado:** `find_optimal_threshold()` busca el umbral que maximiza F1-mora sobre el train set. Con mora=4.85%, el threshold 0.50 predice siempre clase 0.

**Cross-validation con threshold consistente:** en cada fold se calibra el threshold sobre el fold de train con la misma estrategia (`'f1'`), y se aplica sobre el fold de validación. Garantiza que las métricas de CV sean homogéneas con las métricas de test.

**GradientBoosting:** no acepta `class_weight`. Se usa `sample_weight` con `pos_weight ≈ 19x` para compensar el desbalance.

**Criterio de selección** (ponderado, config-driven):

| Componente | Fuente | Peso |
|---|---|---|
| ROC-AUC | CV mean | 40% |
| PR-AUC | test set | 30% |
| Recall mora | CV mean | 20% |
| Gap AUC | penalización | 10% |

**Resultados:**

| Modelo | ROC-AUC CV | Recall CV | ROC-AUC test |
|--------|-----------|-----------|-------------|
| **Logistic Regression** ✓ | **0.692** | **0.614** | 0.509 |
| Random Forest | 0.710 | 0.201 | 0.524 |
| Gradient Boosting | 0.706 | 0.039 | 0.519 |
| Decision Tree | 0.631 | 0.518 | 0.509 |

El gap train-test (0.20) refleja que el test tiene solo 29 eventos de mora — las métricas de test son inestables estadísticamente. El CV sobre train es el criterio principal de selección.

### Despliegue (`model_deploy.py`)

API Flask batch con tres endpoints:
- `GET /health` — estado del servicio
- `GET /metrics` — métricas del modelo + PSI de drift
- `POST /predict` — predicción batch (JSON o CSV)

Flujo de predicción en producción:
```
df_raw  →  pipeline_stateless (fresh)  →  pipeline_base.pkl (caps train)  →  pipeline_ml.pkl  →  scores
```

Genera `Dockerfile` y `requirements_deploy.txt` listos para CI/CD.

### Monitoreo (`model_monitoring.py`)

- **PSI** sobre el score de salida del modelo
- **PSI por feature** para detectar drift en variables de entrada
- **Performance por ventana temporal** (Q1–Q4) del log de predicciones

Umbral de alerta: PSI > 0.2 (configurable en `config.json`).

---

## Configuración (`config.json`)

Todo el comportamiento del pipeline está centralizado:

```json
{
  "project_code": "cdp_credito",
  "paths": { ... },
  "target": { "label_col": "Pago_atiempo", "event_col": "mora" },
  "feature_engineering": { "winsorize_quantile": 0.99, ... },
  "split": { "type": "temporal", "train_cutoff": "2025-09" },
  "training": {
    "cv_folds": 5,
    "cv_scoring": "recall",
    "threshold_strategy": "f1",
    "selection_weights": { "roc_auc": 0.4, "pr_auc": 0.3, "recall": 0.2, "gap": 0.1 }
  },
  "monitoring": { "psi_threshold": 0.2 },
  "deploy": { "docker_image": "cdp_credito-scoring", "port": 5000 }
}
```

---

## Docker

```bash
# 1. Generar artefactos primero (solo la primera vez o tras reentrenar)
bash run_pipeline.sh        # Mac/Linux
# run_pipeline.bat          # Windows

# 2. Build desde mlops_pipeline/
docker build -t cdp_credito-scoring .

# 3. Run
docker run -p 5000:5000 cdp_credito-scoring

# Health check
curl http://localhost:5000/health

# Predicción batch
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '[{"capital_prestado": 2000000, "plazo_meses": 24}]'
```

---

## Variables y Poder Predictivo (EDA)

| Variable | IV | p-valor | Acción |
|---|---|---|---|
| `puntaje_datacredito` | 0.199 (Medio) | <0.001 | Incluida |
| `huella_consulta` | 0.146 (Medio) | <0.001 | Incluida |
| `plazo_meses` | 0.128 (Medio) | <0.001 | Incluida |
| `promedio_ingresos_datacredito` | 0.107 (Medio) | <0.001 | Incluida + imputación segmentada |
| `edad_cliente` | 0.089 (Débil) | <0.001 | Incluida (bucketizada) |
| `puntaje` | — | — | **Excluida — leakage confirmado** (clientes al día: 95.23 fijo) |
| `salario_cliente` | 0.024 (Débil) | 0.680 (ns) | Incluida con winsorización p99 |

---

## Limitaciones conocidas

- El conjunto de test tiene **29 eventos de mora** (3.52%), lo que hace que las métricas puntuales de test sean estadísticamente inestables. Una predicción correcta adicional mueve el recall ~3.4 puntos.
- El gap train-test (ROC-AUC: 0.69 vs 0.51) es alto, pero refleja esta inestabilidad más que overfitting real.
- Con más datos de test o una ventana temporal más amplia, las métricas de generalización serían más confiables.

---

## Dependencias principales

| Paquete | Versión mínima | Uso |
|---|---|---|
| scikit-learn | 1.2 | Pipelines, modelos, métricas |
| pandas | 2.0 | Transformaciones y EDA |
| numpy | 1.24 | Operaciones vectoriales |
| joblib | 1.3 | Serialización de modelos |
| matplotlib / seaborn | 3.7 / 0.13 | Visualizaciones |
| flask | 3.0 | API de inferencia |
| scipy | 1.10 | Tests estadísticos (EDA) |

Ver `requirements.txt` para desarrollo, `requirements_deploy.txt` para producción (imagen Docker).