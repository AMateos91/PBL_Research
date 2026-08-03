from __future__ import annotations

from sklearn.linear_model import ElasticNet

from .base import ClassicalModel


class ElasticNetRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            ElasticNet(
                **kwargs,
            )
        )
