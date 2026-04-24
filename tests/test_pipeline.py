"""
test_pipeline.py
================
Suite de pruebas unitarias para el pipeline MLOps de mora crediticia.

Cobertura:
  - ft_engineering: CrearFeaturesDerivadas, LimpiarTendenciaIngresos,
                    ImputacionSegmentada, Winsorizar, EliminarColumnas
  - model_training:  find_optimal_threshold, summarize_classification, build_model
  - heuristic_model: HeuristicMoraModel (predict, predict_proba, get_params)

Ejecutar:
  cd mlops_pipeline
  pytest tests/ --cov=src --cov-report=xml:coverage.xml -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# ── Asegura que src/ esté en el path ─────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para tests de plots
import matplotlib.pyplot as plt

from ft_engineering import (
    CrearFeaturesDerivadas,
    EliminarColumnas,
    ImputacionSegmentada,
    LimpiarCategoricas,
    Winsorizar,
    load_config,
    resolve_cfg,
    _deep_merge,
    build_pipeline_stateless,
    build_pipeline_base,
    build_pipeline_ml,
    temporal_split,
    _validate_columns,
)
LimpiarTendenciaIngresos = LimpiarCategoricas  # alias para tests heredados
from heuristic_model import (
    HeuristicMoraModel, evaluate_heuristic, HeuristicRuleModel,
    plot_learning_curve,
)
from model_training import (
    build_model, find_optimal_threshold, summarize_classification,
    build_model_definitions,
)
from model_evaluation import (
    plot_score_distribution,
    get_event_metadata as eval_get_event_metadata,
    decile_analysis,
)
from model_monitoring import (
    build_monitoring_html,
    plot_performance_over_time,
    plot_score_drift,
    get_event_metadata as mon_get_event_metadata,
    compute_psi_feature,
)
from model_deploy import (
    get_event_metadata as deploy_get_event_metadata,
    resolve_runtime_paths as deploy_resolve_runtime_paths,
    _make_risk_classifier,
    _load_batch_input,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures compartidos

CFG_MINIMAL = {
    "feature_engineering": {
        "derived_features": {
            "antiguedad_ref_date": "2025-10-01",
            "edad_bucket_bins": [18, 26, 36, 46, 56, 66, 120],
            "edad_bucket_labels": ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
            "ratio_features": [
                {"name": "ratio_cuota_ingresos",
                 "numerator": "cuota_pactada",
                 "denominator": "promedio_ingresos_datacredito"},
                {"name": "ratio_capital_ingresos",
                 "numerator": "capital_prestado",
                 "denominator": "promedio_ingresos_datacredito"},
                {"name": "ratio_otros_prestamos_ingresos",
                 "numerator": "total_otros_prestamos",
                 "denominator": "promedio_ingresos_datacredito"},
                {"name": "dti_aprox",
                 "numerator_sum": ["cuota_pactada", "total_otros_prestamos"],
                 "denominator": "promedio_ingresos_datacredito"},
            ],
        }
    },
    "split": {"date_col": "fecha_prestamo"},
}

def _make_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """DataFrame mínimo con las columnas base del CSV."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2023-01-01", periods=n, freq="30D")
    return pd.DataFrame(
        {
            "fecha_prestamo": fechas,
            "capital_prestado": rng.integers(500_000, 5_000_000, n).astype(float),
            "plazo_meses": rng.integers(6, 60, n),
            "edad_cliente": rng.integers(22, 65, n),
            "tipo_laboral": rng.choice(["EMPLEADO", "INDEPENDIENTE", "PENSIONADO"], n),
            "salario_cliente": rng.integers(1_200_000, 10_000_000, n).astype(float),
            "total_otros_prestamos": rng.integers(0, 5_000_000, n).astype(float),
            "cuota_pactada": rng.integers(100_000, 800_000, n).astype(float),
            "puntaje": rng.uniform(400, 950, n),
            "puntaje_datacredito": rng.uniform(400, 950, n),
            "huella_consulta": rng.integers(0, 15, n),
            "tendencia_ingresos": rng.choice(
                ["ESTABLE", "CRECIENTE", "DECRECIENTE", "999", None], n
            ),
            "creditos_sectorFinanciero": rng.integers(0, 5, n),
            "creditos_sectorCooperativo": rng.integers(0, 3, n),
            "creditos_sectorReal": rng.integers(0, 2, n),
            "promedio_ingresos_datacredito": np.where(
                rng.random(n) < 0.3, np.nan, rng.uniform(1e6, 8e6, n)
            ),
            "mora": rng.choice([0, 1], n, p=[0.95, 0.05]),
        }
    )

# ─────────────────────────────────────────────────────────────────────────────
#  CrearFeaturesDerivadas

class TestCrearFeaturesDerivadas:
    def setup_method(self):
        self.transformer = CrearFeaturesDerivadas(cfg=CFG_MINIMAL)
        self.df = _make_df(20)

    def test_fit_returns_self(self):
        result = self.transformer.fit(self.df)
        assert result is self.transformer

    def test_ratios_financieros_creados(self):
        out = self.transformer.transform(self.df)
        for col in [
            "ratio_cuota_ingresos",
            "ratio_capital_ingresos",
            "ratio_otros_prestamos_ingresos",
            "dti_aprox",
        ]:
            assert col in out.columns, f"Falta columna: {col}"

    def test_edad_bucket_ordinal_creado(self):
        out = self.transformer.transform(self.df)
        assert "edad_bucket" in out.columns

    def test_temporales_creados(self):
        out = self.transformer.transform(self.df)
        for col in ["anio_prestamo", "mes_prestamo", "dia_semana_prestamo",
                    "antiguedad_prestamo_dias"]:
            assert col in out.columns, f"Falta columna: {col}"

    def test_idempotente(self):
        out1 = self.transformer.transform(self.df)
        out2 = self.transformer.transform(out1)
        assert list(out1.columns) == list(out2.columns)

    def test_ratio_cuota_ingresos_valor(self):
        out = self.transformer.transform(self.df)
        denom = self.df["promedio_ingresos_datacredito"].replace(0, np.nan)
        expected = self.df["cuota_pactada"] / denom
        pd.testing.assert_series_equal(
            out["ratio_cuota_ingresos"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_dti_aprox_valor(self):
        out = self.transformer.transform(self.df)
        denom = self.df["promedio_ingresos_datacredito"].replace(0, np.nan)
        expected = (self.df["cuota_pactada"] + self.df["total_otros_prestamos"]) / denom
        pd.testing.assert_series_equal(
            out["dti_aprox"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_no_modifica_dataframe_original(self):
        cols_before = list(self.df.columns)
        self.transformer.transform(self.df)
        assert list(self.df.columns) == cols_before

    def test_antiguedad_no_negativa(self):
        out = self.transformer.transform(self.df)
        assert (out["antiguedad_prestamo_dias"] >= 0).all()

# ─────────────────────────────────────────────────────────────────────────────
#  LimpiarTendenciaIngresos

class TestLimpiarTendenciaIngresos:
    VALID = ["ESTABLE", "CRECIENTE", "DECRECIENTE"]

    def setup_method(self):
        self.transformer = LimpiarCategoricas(
            cleaners={"tendencia_ingresos": self.VALID}
        )

    def _df(self, values):
        return pd.DataFrame({"tendencia_ingresos": values})

    def test_fit_returns_self(self):
        df = self._df(["ESTABLE"])
        assert self.transformer.fit(df) is self.transformer

    def test_valores_validos_se_conservan(self):
        df = self._df(["ESTABLE", "CRECIENTE", "DECRECIENTE"])
        out = self.transformer.transform(df)
        assert list(out["tendencia_ingresos"]) == ["ESTABLE", "CRECIENTE", "DECRECIENTE"]

    def test_valores_invalidos_se_nullifican(self):
        df = self._df(["ESTABLE", "999", "1.5", "CRECIENTE"])
        out = self.transformer.transform(df)
        assert pd.isna(out["tendencia_ingresos"].iloc[1])
        assert pd.isna(out["tendencia_ingresos"].iloc[2])

    def test_nulos_se_conservan(self):
        df = self._df([None, "ESTABLE"])
        out = self.transformer.transform(df)
        assert pd.isna(out["tendencia_ingresos"].iloc[0])

    def test_todos_invalidos(self):
        df = self._df(["abc", "123", "xyz"])
        out = self.transformer.transform(df)
        assert out["tendencia_ingresos"].isna().all()

    def test_no_modifica_df_original(self):
        df = self._df(["999", "ESTABLE"])
        self.transformer.transform(df)
        assert df["tendencia_ingresos"].iloc[0] == "999"

# ─────────────────────────────────────────────────────────────────────────────
#  ImputacionSegmentada

class TestImputacionSegmentada:
    def setup_method(self):
        self.impute_map = {"promedio_ingresos_datacredito": "tipo_laboral"}
        self.transformer = ImputacionSegmentada(impute_map=self.impute_map)

    def _df_train(self):
        return pd.DataFrame(
            {
                "tipo_laboral": ["EMPLEADO", "EMPLEADO", "INDEPENDIENTE", "PENSIONADO"],
                "promedio_ingresos_datacredito": [2e6, 3e6, 1.5e6, np.nan],
            }
        )

    def test_fit_aprende_medianas(self):
        self.transformer.fit(self._df_train())
        assert "promedio_ingresos_datacredito" in self.transformer.medians_
        # mediana EMPLEADO = 2.5e6
        assert abs(
            self.transformer.medians_["promedio_ingresos_datacredito"]["EMPLEADO"] - 2.5e6
        ) < 1

    def test_transform_imputa_nulos(self):
        self.transformer.fit(self._df_train())
        df_test = pd.DataFrame(
            {
                "tipo_laboral": ["EMPLEADO", "PENSIONADO"],
                "promedio_ingresos_datacredito": [np.nan, np.nan],
            }
        )
        out = self.transformer.transform(df_test)
        assert out["promedio_ingresos_datacredito"].isna().sum() == 0

    def test_fallback_grupo_nuevo(self):
        self.transformer.fit(self._df_train())
        df_test = pd.DataFrame(
            {
                "tipo_laboral": ["DESCONOCIDO"],
                "promedio_ingresos_datacredito": [np.nan],
            }
        )
        out = self.transformer.transform(df_test)
        assert not pd.isna(out["promedio_ingresos_datacredito"].iloc[0])

    def test_no_nulos_quedan(self):
        self.transformer.fit(self._df_train())
        df = pd.DataFrame(
            {
                "tipo_laboral": ["EMPLEADO", "INDEPENDIENTE", "PENSIONADO"],
                "promedio_ingresos_datacredito": [np.nan, np.nan, np.nan],
            }
        )
        out = self.transformer.transform(df)
        assert out["promedio_ingresos_datacredito"].isna().sum() == 0

    def test_valores_existentes_no_se_modifican(self):
        self.transformer.fit(self._df_train())
        df = pd.DataFrame(
            {
                "tipo_laboral": ["EMPLEADO"],
                "promedio_ingresos_datacredito": [9_999_999.0],
            }
        )
        out = self.transformer.transform(df)
        assert out["promedio_ingresos_datacredito"].iloc[0] == pytest.approx(9_999_999.0)

# ─────────────────────────────────────────────────────────────────────────────
#  Winsorizar

class TestWinsorizar:
    def setup_method(self):
        self.cols = ["salario_cliente", "capital_prestado"]
        self.transformer = Winsorizar(cols=self.cols, quantile=0.99)

    def _df(self, vals_sal, vals_cap):
        return pd.DataFrame(
            {"salario_cliente": vals_sal, "capital_prestado": vals_cap}
        )

    def test_fit_aprende_caps(self):
        df = self._df(list(range(1, 101)), list(range(100, 200)))
        self.transformer.fit(df)
        assert "salario_cliente" in self.transformer.caps_
        assert "capital_prestado" in self.transformer.caps_

    def test_transform_clipea_outliers(self):
        df_train = self._df(list(range(1, 101)), list(range(1, 101)))
        self.transformer.fit(df_train)
        cap_sal = self.transformer.caps_["salario_cliente"]

        df_test = self._df([cap_sal * 10, cap_sal * 0.5], [50, 50])
        out = self.transformer.transform(df_test)
        assert out["salario_cliente"].iloc[0] <= cap_sal + 1e-6

    def test_valores_bajo_cap_no_cambian(self):
        df_train = self._df(list(range(1, 101)), list(range(1, 101)))
        self.transformer.fit(df_train)
        cap_sal = self.transformer.caps_["salario_cliente"]

        df_test = self._df([cap_sal * 0.5], [50])
        out = self.transformer.transform(df_test)
        assert abs(out["salario_cliente"].iloc[0] - cap_sal * 0.5) < 1e-6

    def test_no_modifica_df_original(self):
        df_train = self._df(list(range(1, 101)), list(range(1, 101)))
        self.transformer.fit(df_train)

        vals_orig = [1_000_000_000]
        df_test = self._df(vals_orig, [50])
        self.transformer.transform(df_test)
        assert df_test["salario_cliente"].iloc[0] == vals_orig[0]

    def test_columna_inexistente_se_ignora(self):
        transformer = Winsorizar(cols=["col_inexistente"], quantile=0.99)
        df = self._df(list(range(1, 11)), list(range(1, 11)))
        transformer.fit(df)
        out = transformer.transform(df)
        assert list(df.columns) == list(out.columns)

# ─────────────────────────────────────────────────────────────────────────────
#  EliminarColumnas

class TestEliminarColumnas:
    def test_elimina_columnas_existentes(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        out = EliminarColumnas(cols_to_drop=["a", "b"]).transform(df)
        assert list(out.columns) == ["c"]

    def test_ignora_columnas_inexistentes(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        out = EliminarColumnas(cols_to_drop=["x", "y"]).transform(df)
        assert list(out.columns) == ["a", "b"]

    def test_fit_returns_self(self):
        t = EliminarColumnas(cols_to_drop=["a"])
        assert t.fit(pd.DataFrame({"a": [1]})) is t

    def test_no_modifica_df_original(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        EliminarColumnas(cols_to_drop=["a"]).transform(df)
        assert "a" in df.columns

# ─────────────────────────────────────────────────────────────────────────────
#  find_optimal_threshold

class TestFindOptimalThreshold:
    def _balanced_probas(self):
        """Probas perfectamente separadas → threshold óptimo cercano a 0.5."""
        y = np.array([0] * 40 + [1] * 10)
        probas = np.concatenate([np.linspace(0.01, 0.4, 40), np.linspace(0.6, 0.99, 10)])
        return y, probas

    def test_retorna_float(self):
        y, probas = self._balanced_probas()
        thr = find_optimal_threshold(y, probas, strategy="f1")
        assert isinstance(thr, float)

    def test_threshold_en_rango_valido(self):
        y, probas = self._balanced_probas()
        thr = find_optimal_threshold(y, probas, strategy="f1")
        assert 0.0 <= thr <= 1.0

    def test_estrategia_recall(self):
        y, probas = self._balanced_probas()
        thr = find_optimal_threshold(y, probas, strategy="recall")
        assert 0.0 <= thr <= 1.0

    def test_estrategia_prior(self):
        y, probas = self._balanced_probas()
        thr = find_optimal_threshold(y, probas, strategy="prior")
        assert 0.0 <= thr <= 1.0

# ─────────────────────────────────────────────────────────────────────────────
#  summarize_classification

class TestSummarizeClassification:
    def _data(self):
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 50 + [1] * 10)
        probas = np.concatenate(
            [rng.uniform(0.0, 0.4, 50), rng.uniform(0.5, 1.0, 10)]
        )
        y_pred = (probas >= 0.5).astype(int)
        return y_true, y_pred, probas

    def test_retorna_dict(self):
        y, yp, p = self._data()
        result = summarize_classification(y, yp, p, model_name="test", verbose=False)
        assert isinstance(result, dict)

    def test_contiene_metricas_clave(self):
        y, yp, p = self._data()
        result = summarize_classification(y, yp, p, model_name="test", verbose=False)
        for key in ["roc_auc", "pr_auc", "recall_event", "precision_event"]:
            assert key in result, f"Falta métrica: {key}"

    def test_roc_auc_en_rango(self):
        y, yp, p = self._data()
        result = summarize_classification(y, yp, p, model_name="test", verbose=False)
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_recall_event_en_rango(self):
        y, yp, p = self._data()
        result = summarize_classification(y, yp, p, model_name="test", verbose=False)
        assert 0.0 <= result["recall_event"] <= 1.0

# ─────────────────────────────────────────────────────────────────────────────
#  build_model

class TestBuildModel:
    def _data(self):
        rng = np.random.default_rng(99)
        X = rng.standard_normal((100, 5))
        y = (X[:, 0] + rng.standard_normal(100) > 0).astype(int)
        # split simple 80/20
        return X[:80], y[:80], X[80:], y[80:]

    def test_retorna_tupla(self):
        x_tr, y_tr, x_te, y_te = self._data()
        result = build_model("lr_test", LogisticRegression(max_iter=200, random_state=42),
                             x_tr, y_tr, x_te, y_te)
        assert isinstance(result, tuple)

    def test_estimador_puede_predecir(self):
        x_tr, y_tr, x_te, y_te = self._data()
        estimator, *_ = build_model("lr_test", LogisticRegression(max_iter=200, random_state=42),
                                    x_tr, y_tr, x_te, y_te)
        preds = estimator.predict(x_te)
        assert len(preds) == len(y_te)

    def test_preds_son_binarias(self):
        x_tr, y_tr, x_te, y_te = self._data()
        estimator, *_ = build_model("lr_test", LogisticRegression(max_iter=200, random_state=42),
                                    x_tr, y_tr, x_te, y_te)
        preds = estimator.predict(x_te)
        assert set(preds).issubset({0, 1})

# ─────────────────────────────────────────────────────────────────────────────
#  HeuristicMoraModel

_RULES_STD = [
    {"col": "puntaje_datacredito", "op": "<",  "threshold": 760},
    {"col": "huella_consulta",     "op": ">",  "threshold": 5},
    {"col": "plazo_meses",         "op": ">",  "threshold": 12},
]

class TestHeuristicMoraModel:
    def _df(self):
        return pd.DataFrame(
            {
                "puntaje_datacredito": [800, 700, 800, 700],
                "huella_consulta": [2, 8, 8, 8],
                "plazo_meses": [6, 6, 18, 18],
            }
        )

    def _model(self):
        return HeuristicRuleModel(rules=_RULES_STD, min_signals=2)

    def test_fit_returns_self(self):
        model = self._model()
        assert model.fit(self._df(), np.array([0, 1, 0, 1])) is model

    def test_predict_retorna_array_binario(self):
        preds = self._model().predict(self._df())
        assert set(preds).issubset({0, 1})

    def test_predict_longitud_correcta(self):
        assert len(self._model().predict(self._df())) == len(self._df())

    def test_predict_proba_suma_uno(self):
        probas = self._model().predict_proba(self._df())
        np.testing.assert_allclose(probas.sum(axis=1), 1.0)

    def test_predict_proba_shape(self):
        assert self._model().predict_proba(self._df()).shape == (4, 2)

    def test_todas_senales_activas_predice_mora(self):
        """puntaje<760 + huella>5 + plazo>12 → mora=1 con min_signals=2."""
        df = pd.DataFrame({
            "puntaje_datacredito": [700],
            "huella_consulta": [8],
            "plazo_meses": [18],
        })
        assert self._model().predict(df)[0] == 1

    def test_ninguna_senal_predice_no_mora(self):
        df = pd.DataFrame({
            "puntaje_datacredito": [900],
            "huella_consulta": [1],
            "plazo_meses": [6],
        })
        assert self._model().predict(df)[0] == 0

    def test_get_params(self):
        model = HeuristicRuleModel(rules=_RULES_STD, min_signals=1)
        params = model.get_params()
        assert "rules" in params
        assert params["min_signals"] == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Nuevas features de ft_engineering: bucket_features y denominator_add

_CFG_NEW_FEATURES = {
    "feature_engineering": {
        "derived_features": {
            "antiguedad_ref_date": "2025-10-01",
            "ratio_features": [
                {
                    "name": "ratio_cuota_ingresos",
                    "numerator": "cuota_pactada",
                    "denominator": "promedio_ingresos_datacredito",
                },
                {
                    "name": "ratio_consultas",
                    "numerator": "huella_consulta",
                    "denominator": "cant_creditosvigentes",
                    "denominator_add": 1,
                },
            ],
            "bucket_features": [
                {
                    "name": "score_bucket",
                    "col": "puntaje_datacredito",
                    "bins": [0, 700, 800, 1000],
                    "labels": ["alto", "medio", "bajo"],
                    "right": False,
                }
            ],
        }
    },
    "split": {"date_col": "fecha_prestamo"},
}


def _make_df_ext(n: int = 20, seed: int = 42) -> pd.DataFrame:
    base = _make_df(n, seed)
    rng = np.random.default_rng(seed)
    base["cant_creditosvigentes"] = rng.integers(0, 10, n)
    return base


class TestBucketFeatures:
    def setup_method(self):
        self.t = CrearFeaturesDerivadas(cfg=_CFG_NEW_FEATURES)
        self.df = _make_df_ext()

    def test_score_bucket_creado(self):
        out = self.t.transform(self.df)
        assert "score_bucket" in out.columns

    def test_score_bucket_valores_validos(self):
        out = self.t.transform(self.df)
        labels = {"alto", "medio", "bajo"}
        assert set(out["score_bucket"].dropna().unique()).issubset(labels)

    def test_bucket_idempotente(self):
        out1 = self.t.transform(self.df)
        out2 = self.t.transform(out1)
        assert list(out1.columns) == list(out2.columns)


class TestDenominatorAdd:
    def setup_method(self):
        self.t = CrearFeaturesDerivadas(cfg=_CFG_NEW_FEATURES)

    def test_denominator_add_evita_nan_con_cero(self):
        df = _make_df_ext(10)
        df["cant_creditosvigentes"] = 0
        out = self.t.transform(df)
        assert "ratio_consultas" in out.columns
        assert out["ratio_consultas"].isna().sum() == 0

    def test_sin_denominator_add_cero_da_nan(self):
        cfg = {
            "feature_engineering": {
                "derived_features": {
                    "antiguedad_ref_date": "2025-10-01",
                    "ratio_features": [
                        {"name": "r", "numerator": "huella_consulta",
                         "denominator": "cant_creditosvigentes"},
                    ],
                }
            },
            "split": {"date_col": "fecha_prestamo"},
        }
        t = CrearFeaturesDerivadas(cfg=cfg)
        df = _make_df_ext(5)
        df["cant_creditosvigentes"] = 0
        out = t.transform(df)
        assert out["r"].isna().all()


# ─────────────────────────────────────────────────────────────────────────────
#  neg_label en summarize_classification y build_model

class TestNegLabelSummarize:
    def _data(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 50 + [1] * 10)
        p = np.concatenate([rng.uniform(0.0, 0.4, 50), rng.uniform(0.5, 1.0, 10)])
        return y, (p >= 0.5).astype(int), p

    def test_neg_label_custom_acepta(self):
        y, yp, p = self._data()
        result = summarize_classification(
            y, yp, p, model_name="m", event_label="mora",
            neg_label="Al dia", verbose=False,
        )
        assert isinstance(result, dict)
        assert 0.0 <= result["recall_event"] <= 1.0

    def test_build_model_neg_label(self):
        rng = np.random.default_rng(99)
        X = rng.standard_normal((100, 4))
        y = (X[:, 0] > 0).astype(int)
        est, *_ = build_model(
            "lr", LogisticRegression(max_iter=200, random_state=42),
            X[:80], y[:80], X[80:], y[80:],
            event_label="mora", neg_label="Al dia",
        )
        assert hasattr(est, "predict")


# ─────────────────────────────────────────────────────────────────────────────
#  neg_label en evaluate_heuristic

class TestEvaluateHeuristicNegLabel:
    def _model_and_data(self):
        rules = [{"col": "puntaje_datacredito", "op": "<", "threshold": 760}]
        model = HeuristicRuleModel(rules=rules, min_signals=1)
        df = pd.DataFrame({
            "puntaje_datacredito": [700, 800, 650, 900],
        })
        y = np.array([1, 0, 1, 0])
        return model, df, y

    def test_neg_label_custom(self):
        model, df, y = self._model_and_data()
        metrics = evaluate_heuristic(
            model, df, y, split_name="Test",
            event_label="mora", neg_label="Al dia",
        )
        assert isinstance(metrics, dict)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_neg_label_default(self):
        model, df, y = self._model_and_data()
        metrics = evaluate_heuristic(model, df, y, event_label="mora")
        assert "roc_auc" in metrics


# ─────────────────────────────────────────────────────────────────────────────
#  neg_label en plot_score_distribution

class TestPlotScoreDistributionNegLabel:
    def test_neg_label_custom_no_falla(self, tmp_path):
        rng = np.random.default_rng(0)
        y = np.array([0] * 30 + [1] * 10)
        p = np.concatenate([rng.uniform(0, 0.5, 30), rng.uniform(0.5, 1, 10)])
        save = tmp_path / "dist.png"
        plot_score_distribution(y, p, 0.5, "modelo", save, neg_label="Al dia")
        plt.close("all")
        assert save.exists()

    def test_neg_label_default_no_falla(self, tmp_path):
        rng = np.random.default_rng(1)
        y = np.array([0] * 20 + [1] * 5)
        p = np.concatenate([rng.uniform(0, 0.4, 20), rng.uniform(0.6, 1, 5)])
        save = tmp_path / "dist2.png"
        plot_score_distribution(y, p, 0.5, "modelo", save)
        plt.close("all")
        assert save.exists()


# ─────────────────────────────────────────────────────────────────────────────
#  event_title en funciones de model_monitoring

class TestMonitoringEventTitle:
    def _perf_df(self):
        return pd.DataFrame({
            "window": ["Q1", "Q2"],
            "score_mean": [0.4, 0.45],
            "score_std": [0.1, 0.12],
            "pct_pred_event": [0.2, 0.25],
            "event_rate_real": [0.04, 0.05],
        })

    def _psi_df(self):
        return pd.DataFrame({
            "feature": ["feat_a", "feat_b"],
            "psi": [0.05, 0.25],
            "status": ["Estable", "Drift"],
        })

    def test_plot_score_drift_event_title(self, tmp_path):
        rng = np.random.default_rng(0)
        save = tmp_path / "drift.png"
        plot_score_drift(
            rng.uniform(0, 1, 100), rng.uniform(0, 1, 80),
            "ModeloTest", 0.05, save, event_title="Mora",
        )
        plt.close("all")
        assert save.exists()

    def test_plot_score_drift_default_event_title(self, tmp_path):
        rng = np.random.default_rng(1)
        save = tmp_path / "drift2.png"
        plot_score_drift(
            rng.uniform(0, 1, 50), rng.uniform(0, 1, 50),
            "ModeloTest", 0.10, save,
        )
        plt.close("all")
        assert save.exists()

    def test_plot_performance_event_title(self, tmp_path):
        save = tmp_path / "perf.png"
        plot_performance_over_time(self._perf_df(), save, event_title="Mora")
        plt.close("all")
        assert save.exists()

    def test_plot_performance_default_event_title(self, tmp_path):
        save = tmp_path / "perf2.png"
        plot_performance_over_time(self._perf_df(), save)
        plt.close("all")
        assert save.exists()

    def test_build_monitoring_html_event_title(self):
        html = build_monitoring_html(
            model_name="RF",
            psi_df=self._psi_df(),
            psi_score=0.05,
            perf_df=self._perf_df(),
            n_ref=1000,
            n_prod=200,
            psi_threshold=0.2,
            event_title="Mora",
        )
        assert "Mora" in html
        assert isinstance(html, str)

    def test_build_monitoring_html_default_event_title(self):
        html = build_monitoring_html(
            model_name="RF",
            psi_df=self._psi_df(),
            psi_score=0.05,
            perf_df=self._perf_df(),
            n_ref=1000,
            n_prod=200,
            psi_threshold=0.2,
        )
        assert isinstance(html, str)


# ─────────────────────────────────────────────────────────────────────────────
#  ft_engineering: load_config, resolve_cfg, _deep_merge

class TestFtEngineeringConfig:
    def test_load_config_returns_dict(self):
        cfg, _ = load_config()
        assert isinstance(cfg, dict)
        assert "use_cases" in cfg

    def test_resolve_cfg_scoring_mora(self):
        cfg_global, _ = load_config()
        cfg = resolve_cfg(cfg_global, "scoring_mora")
        assert cfg["_use_case"] == "scoring_mora"
        assert "feature_engineering" in cfg

    def test_resolve_cfg_unknown_raises(self):
        cfg_global, _ = load_config()
        with pytest.raises(KeyError):
            resolve_cfg(cfg_global, "use_case_inexistente")

    def test_deep_merge_overrides_nested(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        result = _deep_merge(base, override)
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["a"] == 1

    def test_deep_merge_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base["a"]["b"] == 1

    def test_validate_columns_ok(self):
        cfg_global, _ = load_config()
        cfg = resolve_cfg(cfg_global, "scoring_mora")
        fe = cfg["feature_engineering"]
        cols = fe["numeric_cols"] + fe["categorical_cols"] + list(fe["ordinal_cols"].keys())
        df = pd.DataFrame({c: [0] for c in cols})
        _validate_columns(df, cfg)  # should not raise

    def test_validate_columns_missing_raises(self):
        cfg_global, _ = load_config()
        cfg = resolve_cfg(cfg_global, "scoring_mora")
        with pytest.raises(ValueError):
            _validate_columns(pd.DataFrame({"col_x": [1]}), cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  ft_engineering: pipeline builders

def _make_scoring_mora_df(n: int = 30, seed: int = 7) -> pd.DataFrame:
    """DataFrame completo para el pipeline scoring_mora."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2024-01-01", periods=n, freq="10D")
    return pd.DataFrame({
        "tipo_credito":              rng.choice(["CONSUMO", "COMERCIAL", "HIPOTECARIO"], n),
        "fecha_prestamo":            fechas,
        "capital_prestado":          rng.uniform(5e5, 5e6, n),
        "plazo_meses":               rng.integers(6, 60, n),
        "edad_cliente":              rng.integers(22, 65, n),
        "tipo_laboral":              rng.choice(["Empleado", "Independiente", "Pensionado"], n),
        "salario_cliente":           rng.uniform(1.2e6, 10e6, n),
        "total_otros_prestamos":     rng.uniform(0, 5e6, n),
        "cuota_pactada":             rng.uniform(1e5, 8e5, n),
        "puntaje":                   rng.uniform(400, 950, n),
        "puntaje_datacredito":       rng.uniform(400, 950, n),
        "cant_creditosvigentes":     rng.integers(0, 10, n),
        "huella_consulta":           rng.integers(0, 15, n),
        "saldo_mora":                rng.uniform(0, 1e6, n),
        "saldo_total":               rng.uniform(0, 1e7, n),
        "saldo_principal":           rng.uniform(0, 1e7, n),
        "saldo_mora_codeudor":       rng.uniform(0, 5e5, n),
        "creditos_sectorFinanciero": rng.integers(0, 5, n),
        "creditos_sectorCooperativo":rng.integers(0, 3, n),
        "creditos_sectorReal":       rng.integers(0, 2, n),
        "promedio_ingresos_datacredito": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(1e6, 8e6, n)),
        "tendencia_ingresos":        rng.choice(["Creciente", "Estable", "Decreciente", None], n),
        "Pago_atiempo":              rng.choice([0, 1], n, p=[0.05, 0.95]),
    })


class TestBuildPipelines:
    def setup_method(self):
        cfg_global, _ = load_config()
        self.cfg = resolve_cfg(cfg_global, "scoring_mora")

    def test_build_pipeline_stateless_steps(self):
        pipe = build_pipeline_stateless(self.cfg)
        assert len(pipe.steps) == 2

    def test_build_pipeline_stateless_adds_derived_features(self):
        df = _make_scoring_mora_df()
        pipe = build_pipeline_stateless(self.cfg)
        out = pipe.fit_transform(df)
        assert "ratio_cuota_ingresos" in out.columns
        assert "edad_bucket" in out.columns

    def test_build_pipeline_base_steps(self):
        pipe = build_pipeline_base(self.cfg)
        assert len(pipe.steps) == 3

    def test_build_pipeline_base_removes_leakage(self):
        df = _make_scoring_mora_df()
        stateless = build_pipeline_stateless(self.cfg)
        df_pre = stateless.fit_transform(df)
        base = build_pipeline_base(self.cfg)
        out = base.fit_transform(df_pre)
        for col in ["puntaje", "Pago_atiempo", "saldo_mora"]:
            assert col not in out.columns

    def test_build_pipeline_ml_output_shape(self):
        df = _make_scoring_mora_df(40)
        stateless = build_pipeline_stateless(self.cfg)
        df_pre = stateless.fit_transform(df)
        df_pre["mora"] = (df_pre["Pago_atiempo"] == 0).astype(int)
        base = build_pipeline_base(self.cfg)
        df_base = base.fit_transform(df_pre.drop(columns=["mora"], errors="ignore"))
        ml = build_pipeline_ml(self.cfg)
        out = ml.fit_transform(df_base)
        assert out.shape[0] == 40
        assert out.shape[1] > 0


class TestTemporalSplitRandom:
    def test_random_split_sizes(self):
        df = _make_df(100)
        df["mora"] = np.random.default_rng(0).choice([0, 1], 100, p=[0.95, 0.05])
        cfg = {"split": {"type": "random", "test_size": 0.2, "date_col": "fecha_prestamo"},
               "target": {"event_col": "mora"}}
        train_df, test_df = temporal_split(df, cfg, "fecha_prestamo")
        assert len(train_df) + len(test_df) == 100

    def test_unknown_split_type_raises(self):
        df = _make_df(10)
        df["mora"] = 0
        cfg = {"split": {"type": "invalid", "date_col": "fecha_prestamo"},
               "target": {"event_col": "mora"}}
        with pytest.raises(ValueError):
            temporal_split(df, cfg, "fecha_prestamo")


# ─────────────────────────────────────────────────────────────────────────────
#  model_training: build_model_definitions (new explicit hyperparams)

class TestBuildModelDefinitions:
    def _y(self):
        return np.array([0] * 95 + [1] * 5)

    def _spec(self, model_type, params):
        return {"name": model_type, "type": model_type, "params": params}

    def test_random_forest_explicit_min_samples_leaf(self):
        cfg = {"models": [self._spec("random_forest", {
            "n_estimators": 10, "max_depth": 3,
            "min_samples_leaf": 7, "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        })]}
        defs = build_model_definitions(cfg, self._y())
        _, est, _ = defs[0]
        assert est.min_samples_leaf == 7
        assert est.max_features == "sqrt"

    def test_gradient_boosting_explicit_learning_rate(self):
        cfg = {"models": [self._spec("gradient_boosting", {
            "n_estimators": 10, "learning_rate": 0.05,
            "max_depth": 2, "min_samples_leaf": 5, "subsample": 0.8,
        })]}
        defs = build_model_definitions(cfg, self._y())
        _, est, _ = defs[0]
        assert est.learning_rate == pytest.approx(0.05)

    def test_hist_gradient_boosting_explicit_learning_rate(self):
        cfg = {"models": [self._spec("hist_gradient_boosting", {
            "max_iter": 20, "learning_rate": 0.07, "max_depth": 3,
            "class_weight": "balanced",
        })]}
        defs = build_model_definitions(cfg, self._y())
        _, est, _ = defs[0]
        assert est.learning_rate == pytest.approx(0.07)

    def test_logistic_regression_built(self):
        cfg = {"models": [self._spec("logistic_regression", {
            "C": 1.0, "max_iter": 100, "class_weight": "balanced",
        })]}
        defs = build_model_definitions(cfg, self._y())
        assert len(defs) == 1

    def test_unknown_model_type_raises(self):
        cfg = {"models": [self._spec("xgboost", {})]}
        with pytest.raises(ValueError):
            build_model_definitions(cfg, self._y())

    def test_returns_list_of_tuples(self):
        cfg = {"models": [
            self._spec("logistic_regression", {"max_iter": 50}),
            self._spec("random_forest", {"n_estimators": 5, "min_samples_leaf": 1, "max_features": "sqrt"}),
        ]}
        defs = build_model_definitions(cfg, self._y())
        assert isinstance(defs, list)
        assert len(defs) == 2
        assert all(isinstance(d, tuple) and len(d) == 3 for d in defs)


# ─────────────────────────────────────────────────────────────────────────────
#  heuristic_model: plot_learning_curve (new random_state=42 line)

class TestPlotLearningCurveHeuristic:
    def test_plot_learning_curve_creates_file(self, tmp_path):
        rules = [{"col": "puntaje_datacredito", "op": "<", "threshold": 760}]
        model = HeuristicRuleModel(rules=rules, min_signals=1)
        df = pd.DataFrame({
            "puntaje_datacredito": np.random.default_rng(0).uniform(400, 950, 80),
        })
        y = np.random.default_rng(0).choice([0, 1], 80, p=[0.95, 0.05])
        save = tmp_path / "lc.png"
        plot_learning_curve(model, df, y, save, n_shuffles=3)
        plt.close("all")
        assert save.exists()


# ─────────────────────────────────────────────────────────────────────────────
#  model_evaluation: get_event_metadata, decile_analysis

class TestModelEvaluationHelpers:
    def _cfg(self):
        return {"target": {"event_col": "mora"}}

    def test_get_event_metadata_keys(self):
        meta = eval_get_event_metadata(self._cfg())
        assert meta["event_col"] == "mora"
        assert "score_col" in meta
        assert "pred_col" in meta

    def test_decile_analysis_shape(self):
        rng = np.random.default_rng(42)
        y = np.array([0] * 90 + [1] * 10)
        p = np.concatenate([rng.uniform(0, 0.4, 90), rng.uniform(0.6, 1, 10)])
        result = decile_analysis(y, p)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "event_rate" in result.columns

    def test_decile_analysis_event_rates_valid(self):
        rng = np.random.default_rng(1)
        y = rng.choice([0, 1], 200, p=[0.9, 0.1])
        p = rng.uniform(0, 1, 200)
        result = decile_analysis(y, p)
        assert (result["event_rate"] >= 0).all()
        assert (result["event_rate"] <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
#  model_monitoring: get_event_metadata, compute_psi_feature

class TestModelMonitoringHelpers:
    def test_get_event_metadata_keys(self):
        cfg = {"target": {"event_col": "mora"}}
        meta = mon_get_event_metadata(cfg)
        assert meta["event_col"] == "mora"
        assert "score_col" in meta
        assert "pred_col" in meta

    def test_compute_psi_identical_distributions(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 500)
        psi = compute_psi_feature(data, data)
        assert psi < 0.05

    def test_compute_psi_different_distributions(self):
        rng = np.random.default_rng(0)
        ref  = rng.normal(0, 1, 500)
        prod = rng.normal(3, 1, 500)
        psi = compute_psi_feature(ref, prod)
        assert psi > 0.1

    def test_compute_psi_returns_float(self):
        rng = np.random.default_rng(42)
        psi = compute_psi_feature(rng.uniform(0, 1, 100), rng.uniform(0, 1, 100))
        assert isinstance(psi, float)


# ─────────────────────────────────────────────────────────────────────────────
#  model_deploy: funciones puras sin Flask

class TestModelDeployHelpers:
    def _cfg(self):
        return {"target": {"event_col": "mora"}}

    def _paths_cfg(self):
        cfg_global = {"paths": {
            "model_file":           "artifacts/best_model.joblib",
            "model_meta_file":      "artifacts/best_model_meta.json",
            "train_reference_file": "artifacts/train_reference.csv",
            "logs_file":            "artifacts/prediction_logs.csv",
            "pipeline_ml_file":     "artifacts/pipeline_ml.pkl",
            "pipeline_base_file":   "artifacts/pipeline_base.pkl",
            "deploy_summary_file":  "artifacts/deploy_summary.json",
            "metrics_file":         "reports/metrics_latest.json",
            "drift_report_file":    "reports/drift_report.csv",
        }}
        cfg = {"paths": {
            "artifacts_dir": "artifacts/scoring_mora",
            "reports_dir":   "reports/scoring_mora",
        }}
        return cfg_global, cfg

    def test_get_event_metadata_keys(self):
        meta = deploy_get_event_metadata(self._cfg())
        assert meta["event_col"] == "mora"
        assert meta["score_col"] == "score_mora"
        assert meta["pred_col"] == "pred_mora"
        assert meta["actual_col"] == "mora_real"

    def test_resolve_runtime_paths_returns_dict(self):
        cfg_global, cfg = self._paths_cfg()
        paths = deploy_resolve_runtime_paths(cfg_global, cfg)
        assert isinstance(paths, dict)
        assert "model_file" in paths
        assert "metrics_file" in paths

    def test_resolve_runtime_paths_uses_artifacts_dir(self):
        cfg_global, cfg = self._paths_cfg()
        paths = deploy_resolve_runtime_paths(cfg_global, cfg)
        assert "scoring_mora" in paths["model_file"]

    def test_make_risk_classifier_bajo(self):
        clf = _make_risk_classifier(0.3, {"bajo": 0.5, "medio": 1.0})
        assert clf(0.1) == "Bajo"

    def test_make_risk_classifier_medio(self):
        clf = _make_risk_classifier(0.3, {"bajo": 0.5, "medio": 1.0})
        assert clf(0.2) == "Medio"

    def test_make_risk_classifier_alto(self):
        clf = _make_risk_classifier(0.3, {"bajo": 0.5, "medio": 1.0})
        assert clf(0.35) == "Alto"

    def test_load_batch_input_from_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = _load_batch_input(df)
        pd.testing.assert_frame_equal(out, df)

    def test_load_batch_input_from_csv(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3]})
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        out = _load_batch_input(csv_path)
        assert list(out.columns) == ["x"]
        assert len(out) == 3