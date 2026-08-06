from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ..utils.constants import Metric


class Metrics:
    """
    Factory class for computing evaluation metrics.
    """

    _METRICS = {
        Metric.ACCURACY: accuracy_score,
        Metric.F1: f1_score,
        Metric.MAE: mean_absolute_error,
        Metric.MSE: mean_squared_error,
        Metric.R2: r2_score,
        Metric.RMSE: lambda y_true, y_pred, **kwargs: np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
                **kwargs,
            )
        ),
        Metric.BIAS: lambda y_true, y_pred, **kwargs: float(
            np.mean(
                y_pred - y_true
            )
        ),
    }

    @classmethod
    def compute(
        cls,
        metric: Metric,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
        **kwargs,
    ) -> float:
        """
        Compute the selected evaluation metric.
        """

        if y_true.shape != y_pred.shape:
            raise ValueError(
                "y_true and y_pred must have the same shape."
            )

        if y_true.size == 0:
            raise ValueError(
                "Input arrays cannot be empty."
            )

        try:
            metric_fn = cls._METRICS[metric]
        except KeyError as exc:
            raise ValueError(
                f"Unknown metric: {metric}"
            ) from exc

        return float(
            metric_fn(
                y_true,
                y_pred,
                **kwargs,
            )
        )
