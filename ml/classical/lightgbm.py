from __future__ import annotations

from lightgbm import LGBMRegressor

from .base import ClassicalModel


class LightGBMRegressionModel(ClassicalModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            LGBMRegressor(
                **kwargs,
            )
        )
