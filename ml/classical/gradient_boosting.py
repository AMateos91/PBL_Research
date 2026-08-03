from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor

from .base import ClassicalModel


class GradientBoostingRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            GradientBoostingRegressor(
                **kwargs,
            )
        )
