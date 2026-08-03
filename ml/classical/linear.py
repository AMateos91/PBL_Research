from __future__ import annotations

import joblib
import numpy as np

from sklearn.linear_model import LinearRegression

from ..base import Model


class LinearRegressionModel(Model):

    def __init__(
        self,
        **kwargs,
    ):

        self.model = LinearRegression(
            **kwargs
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "LinearRegressionModel":

        self.model.fit(
            x,
            y,
        )

        return self

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )

    def save(
        self,
        path: str,
    ) -> None:

        joblib.dump(
            self.model,
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "LinearRegressionModel":

        instance = cls()

        instance.model = joblib.load(
            path
        )

        return instance
