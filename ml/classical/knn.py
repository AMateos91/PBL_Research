from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor

from .base import ClassicalModel


class KNearestNeighborsRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            KNeighborsRegressor(
                **kwargs,
            )
        )
