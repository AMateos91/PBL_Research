from __future__ import annotations

from catboost import CatBoostRegressor

from .base import ClassicalModel


class CatBoostRegressionModel(ClassicalModel):

    def __init__(
        self,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            CatBoostRegressor(
                verbose=verbose,
                **kwargs,
            )
        )
