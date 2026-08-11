from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ..utils.constants import Metric

from .evaluator import Evaluator


class ClassificationEvaluator(Evaluator):

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[Metric, float]:

        return {

            Metric.ACCURACY: accuracy_score(
                y_true,
                y_pred,
            ),

            Metric.PRECISION: precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            Metric.RECALL: recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            Metric.F1: f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

        }
