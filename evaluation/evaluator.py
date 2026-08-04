from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Evaluator(ABC):

    @abstractmethod
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        ...
