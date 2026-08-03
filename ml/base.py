from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np


class Model(ABC):

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "Model":
        """
        Train the model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Generate predictions.
        """
        raise NotImplementedError

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
