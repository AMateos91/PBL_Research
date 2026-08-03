from __future__ import annotations

from sklearn.svm import SVR

from .base import ClassicalModel


class SupportVectorRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            SVR(
                **kwargs,
            )
        )
