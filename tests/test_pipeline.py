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
  pytest tests/ --cov=src --cov-report=xml:../coverage.xml -v
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

from ft_engineering import (
    CrearFeaturesDerivadas,
    EliminarColumnas,
    ImputacionSegmentada,
    LimpiarTendenciaIngresos,
    Winsorizar,
)
from heuristic_model import HeuristicMoraModel
from model_training import build_model, find_optimal_threshold, summarize_classification

# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures compartidos

CFG_MINIMAL = {
    "feature_engineering": {
        "derived_features": {
            "antiguedad_ref_date": "2025-10-01",
            "edad_bucket_bins": [18, 26, 36, 46, 56, 66, 120],
            "edad_bucket_labels": ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
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
            "ratio_cuota_salario",
            "ratio_capital_salario",
            "ratio_otros_prestamos_salario",
            "dti_aprox",
        ]:
            assert col in out.columns, f"Falta columna: {col}"

    def test_sector_features_creados(self):
        out = self.transformer.transform(self.df)
        for col in [
            "creditos_sector_total",
            "pct_creditos_sectorFinanciero",
            "pct_creditos_sectorCooperativo",
            "pct_creditos_sectorReal",
        ]:
            assert col in out.columns, f"Falta columna: {col}"

    def test_temporales_creados(self):
        out = self.transformer.transform(self.df)
        for col in ["anio_prestamo", "mes_prestamo", "dia_semana_prestamo",
                    "antiguedad_prestamo_dias"]:
            assert col in out.columns, f"Falta columna: {col}"

    def test_edad_bucket_creado(self):
        out = self.transformer.transform(self.df)
        assert "edad_bucket" in out.columns

    def test_idempotente(self):
        out1 = self.transformer.transform(self.df)
        out2 = self.transformer.transform(out1)
        assert list(out1.columns) == list(out2.columns)

    def test_ratio_cuota_salario_valor(self):
        out = self.transformer.transform(self.df)
        expected = self.df["cuota_pactada"] / self.df["salario_cliente"]
        pd.testing.assert_series_equal(
            out["ratio_cuota_salario"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_dti_aprox_valor(self):
        out = self.transformer.transform(self.df)
        sal = self.df["salario_cliente"].replace(0, np.nan)
        expected = (self.df["cuota_pactada"] + self.df["total_otros_prestamos"]) / sal
        pd.testing.assert_series_equal(
            out["dti_aprox"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_no_modifica_dataframe_original(self):
        cols_before = list(self.df.columns)
        self.transformer.transform(self.df)
        assert list(self.df.columns) == cols_before

    def test_sector_total_correcto(self):
        out = self.transformer.transform(self.df)
        expected = (
            self.df["creditos_sectorFinanciero"]
            + self.df["creditos_sectorCooperativo"]
            + self.df["creditos_sectorReal"]
        )
        pd.testing.assert_series_equal(
            out["creditos_sector_total"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_antiguedad_no_negativa(self):
        out = self.transformer.transform(self.df)
        assert (out["antiguedad_prestamo_dias"] >= 0).all()

# ─────────────────────────────────────────────────────────────────────────────
#  LimpiarTendenciaIngresos

class TestLimpiarTendenciaIngresos:
    VALID = ["ESTABLE", "CRECIENTE", "DECRECIENTE"]

    def setup_method(self):
        self.transformer = LimpiarTendenciaIngresos(valid_values=self.VALID)

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
        assert out["promedio_ingresos_datacredito"].iloc[0] == 9_999_999.0

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
        for key in ["roc_auc", "pr_auc", "recall_mora", "precision_mora"]:
            assert key in result, f"Falta métrica: {key}"

    def test_roc_auc_en_rango(self):
        y, yp, p = self._data()
        result = summarize_classification(y, yp, p, model_name="test", verbose=False)
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_recall_mora_en_rango(self):
        y, yp, p = self._data()
        result = summarize_classification(y, yp, p, model_name="test", verbose=False)
        assert 0.0 <= result["recall_mora"] <= 1.0

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
        X_tr, y_tr, X_te, y_te = self._data()
        result = build_model("lr_test", LogisticRegression(max_iter=200),
                             X_tr, y_tr, X_te, y_te)
        assert isinstance(result, tuple)

    def test_estimador_puede_predecir(self):
        X_tr, y_tr, X_te, y_te = self._data()
        estimator, *_ = build_model("lr_test", LogisticRegression(max_iter=200),
                                    X_tr, y_tr, X_te, y_te)
        preds = estimator.predict(X_te)
        assert len(preds) == len(y_te)

    def test_preds_son_binarias(self):
        X_tr, y_tr, X_te, y_te = self._data()
        estimator, *_ = build_model("lr_test", LogisticRegression(max_iter=200),
                                    X_tr, y_tr, X_te, y_te)
        preds = estimator.predict(X_te)
        assert set(preds).issubset({0, 1})

# ─────────────────────────────────────────────────────────────────────────────
#  HeuristicMoraModel

class TestHeuristicMoraModel:
    def _df(self):
        return pd.DataFrame(
            {
                "puntaje_datacredito": [800, 700, 800, 700],
                "huella_consulta": [2, 8, 8, 8],
                "plazo_meses": [6, 6, 18, 18],
            }
        )

    def test_fit_returns_self(self):
        model = HeuristicMoraModel()
        df = self._df()
        y = np.array([0, 1, 0, 1])
        assert model.fit(df, y) is model

    def test_predict_retorna_array_binario(self):
        model = HeuristicMoraModel()
        df = self._df()
        preds = model.predict(df)
        assert set(preds).issubset({0, 1})

    def test_predict_longitud_correcta(self):
        model = HeuristicMoraModel()
        df = self._df()
        assert len(model.predict(df)) == len(df)

    def test_predict_proba_suma_uno(self):
        model = HeuristicMoraModel()
        df = self._df()
        probas = model.predict_proba(df)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0)

    def test_predict_proba_shape(self):
        model = HeuristicMoraModel()
        df = self._df()
        probas = model.predict_proba(df)
        assert probas.shape == (len(df), 2)

    def test_todas_señales_activas_predice_mora(self):
        """puntaje<760 + huella>5 + plazo>12 → mora=1 con min_signals=2."""
        model = HeuristicMoraModel(
            threshold_puntaje=760,
            threshold_huella=5,
            threshold_plazo=12,
            min_signals=2,
        )
        df = pd.DataFrame(
            {
                "puntaje_datacredito": [700],
                "huella_consulta": [8],
                "plazo_meses": [18],
            }
        )
        assert model.predict(df)[0] == 1

    def test_ninguna_señal_predice_no_mora(self):
        model = HeuristicMoraModel(
            threshold_puntaje=760,
            threshold_huella=5,
            threshold_plazo=12,
            min_signals=2,
        )
        df = pd.DataFrame(
            {
                "puntaje_datacredito": [900],
                "huella_consulta": [1],
                "plazo_meses": [6],
            }
        )
        assert model.predict(df)[0] == 0

    def test_get_params(self):
        model = HeuristicMoraModel(threshold_puntaje=750)
        params = model.get_params()
        assert "threshold_puntaje" in params
        assert params["threshold_puntaje"] == 750