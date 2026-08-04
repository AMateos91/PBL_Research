from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..base import Model


class Explainer(ABC):

    def __init__(
        self,
        model: Model,
    ) -> None:

        self.model = model

    @abstractmethod
    def explain(
        self,
        x: np.ndarray,
    ):
        ...
