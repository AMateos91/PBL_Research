from __future__ import annotations

from sklearn.linear_model import LinearRegression

from .base import ClassicalModel


class LinearRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            LinearRegression(
                **kwargs,
            )
        )
