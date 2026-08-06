from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


class Evaluator(ABC):
    """
    Abstract base class for model evaluation.
    """

    @abstractmethod
    def evaluate(
        self,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> dict[str, float]:
        """
        Evaluate model predictions.

        Parameters
        ----------
        y_true : NDArray[np.floating]
            Ground-truth values.

        y_pred : NDArray[np.floating]
            Predicted values.

        Returns
        -------
        dict[str, float]
            Dictionary containing evaluation metrics.
        """
        raise NotImplementedError
