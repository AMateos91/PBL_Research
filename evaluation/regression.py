from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ...utils.constants import Metric

from .evaluator import Evaluator


class RegressionEvaluator(Evaluator):
    """
    Evaluator for regression models.
    """

    def evaluate(
        self,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> dict[Metric, float]:

        if y_true.shape != y_pred.shape:
            raise ValueError(
                "y_true and y_pred must have the same shape."
            )

        if y_true.size == 0:
            raise ValueError(
                "Input arrays cannot be empty."
            )

        mse = mean_squared_error(
            y_true,
            y_pred,
        )

        return {

            Metric.MAE: mean_absolute_error(
                y_true,
                y_pred,
            ),

            Metric.MSE: mse,

            Metric.RMSE: np.sqrt(mse),

            Metric.R2: r2_score(
                y_true,
                y_pred,
            ),

            Metric.BIAS: float(
                np.mean(
                    y_pred - y_true
                )
            ),
        }
