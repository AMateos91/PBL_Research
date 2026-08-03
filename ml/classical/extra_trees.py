from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor

from .base import ClassicalModel


class ExtraTreesRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            ExtraTreesRegressor(
                **kwargs,
            )
        )
