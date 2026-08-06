from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ..utils.constants import Metric


class Metrics:

    _METRICS = {

        Metric.ACCURACY: accuracy_score,

        Metric.F1: f1_score,

        Metric.MAE: mean_absolute_error,

        Metric.MSE: mean_squared_error,

        Metric.R2: r2_score,

    }

    @classmethod
    def compute(
        cls,
        metric: Metric,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs,
    ):

        if metric not in cls._METRICS:

            raise ValueError(
                f"Unknown metric: {metric}"
            )

        return cls._METRICS[
            metric
        ](
            y_true,
            y_pred,
            **kwargs,
        )
        
