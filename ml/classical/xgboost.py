from __future__ import annotations

from xgboost import XGBRegressor

from .base import ClassicalModel


class XGBoostRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            XGBRegressor(
                **kwargs,
            )
        )
