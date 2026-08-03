from __future__ import annotations

from sklearn.linear_model import Ridge

from .base import ClassicalModel


class RidgeRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            Ridge(
                **kwargs,
            )
        )
