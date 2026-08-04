from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ...utils.constants import Metric

from .evaluator import Evaluator


class RegressionEvaluator(Evaluator):

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[Metric, float]:

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

            Metric.RMSE: mse**0.5,

            Metric.R2: r2_score(
                y_true,
                y_pred,
            ),

        }
