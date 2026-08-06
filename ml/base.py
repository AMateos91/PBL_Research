from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np
import torch


class Model(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> "Model":
        """
        Train the model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """
        Generate predictions.
        """
        raise NotImplementedError

    def eval(
        self,
    ) -> None:
        """
        Put the model in evaluation mode.

        Classical models do not require any action.
        Neural network models should override this method.
        """
        return

    @abstractmethod
    def save(
        self,
        path: str,
    ) -> None:
        """
        Save the trained model.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
    ) -> "Model":
        """
        Load a trained model.
        """
        raise NotImplementedError
