"""
hueristic_model.py

Baseline heurístico (sin entrenamiento) para predecir mora (=1).
Útil para comparar contra modelos ML en model_training.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class HeuristicModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        score_threshold: float = 760,
        huella_threshold: int = 8,
        dti_threshold: float = 2.5,
        ratio_cuota_threshold: float = 0.35,
    ):
        self.score_threshold = score_threshold
        self.huella_threshold = huella_threshold
        self.dti_threshold = dti_threshold
        self.ratio_cuota_threshold = ratio_cuota_threshold

        self._med_score = None
        self._med_huella = None
        self._med_dti = None
        self._med_ratio = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        def _med(col: str) -> float:
            return float(pd.to_numeric(X.get(col, pd.Series(dtype=float)), errors="coerce").median())

        self._med_score = _med("puntaje_datacredito")
        self._med_huella = _med("huella_consulta")
        self._med_dti = _med("dti_aprox")
        self._med_ratio = _med("ratio_cuota_salario")

        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def _risk(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X).copy()

        score = pd.to_numeric(X.get("puntaje_datacredito"), errors="coerce").fillna(self._med_score)
        huella = pd.to_numeric(X.get("huella_consulta"), errors="coerce").fillna(self._med_huella)
        dti = pd.to_numeric(X.get("dti_aprox"), errors="coerce").fillna(self._med_dti)
        ratio = pd.to_numeric(X.get("ratio_cuota_salario"), errors="coerce").fillna(self._med_ratio)

        # escalas suaves a [0,1]
        score_r = np.clip((self.score_threshold - score) / 200.0, 0, 1)
        huella_r = np.clip((huella - self.huella_threshold) / 10.0, 0, 1)
        dti_r = np.clip((dti - self.dti_threshold) / 3.0, 0, 1)
        ratio_r = np.clip((ratio - self.ratio_cuota_threshold) / 0.5, 0, 1)

        return (0.45 * score_r + 0.2 * huella_r + 0.2 * dti_r + 0.15 * ratio_r).to_numpy(dtype=float)

    def predict_proba(self, X):
        p1 = np.clip(self._risk(X), 0, 1)
        p0 = 1 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
