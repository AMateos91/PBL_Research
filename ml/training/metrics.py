from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class Metric:

    _METRICS = {

        "accuracy": accuracy_score,

        "f1": f1_score,

        "mae": mean_absolute_error,

        "mse": mean_squared_error,

        "r2": r2_score,

    }

    @classmethod
    def compute(
        cls,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs,
    ):

        if name not in cls._METRICS:

            raise ValueError(
                f"Unknown metric: {name}"
            )

        return cls._METRICS[
            name
        ](
            y_true,
            y_pred,
            **kwargs,
        )
