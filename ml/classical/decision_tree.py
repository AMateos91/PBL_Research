from __future__ import annotations

from sklearn.tree import DecisionTreeRegressor

from .base import ClassicalModel


class DecisionTreeRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            DecisionTreeRegressor(
                **kwargs,
            )
        )
