from __future__ import annotations

import numpy as np

from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from ..utils.constants import ScalingMethod


class Scaling:

    _SCALERS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    def __init__(
    self,
    method: ScalingMethod = ScalingMethod.STANDARD,
):

    self.scaler = self._SCALERS[
        method.value
    ]()

    def fit_transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        return self.scaler.fit_transform(
            x
        )

    def transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        return self.scaler.transform(
            x
        )
