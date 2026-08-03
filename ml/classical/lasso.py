from __future__ import annotations

from sklearn.linear_model import Lasso

from .base import ClassicalModel


class LassoRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            Lasso(
                **kwargs,
            )
        )
