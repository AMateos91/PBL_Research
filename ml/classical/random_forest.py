from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from .base import ClassicalModel


class RandomForestRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            RandomForestRegressor(
                **kwargs,
            )
        )
