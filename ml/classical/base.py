from __future__ import annotations

from abc import ABC

import numpy as np

from ..base import Model
from ...utils.serialization import (
    load_model,
    save_model,
)


class ClassicalModel(Model, ABC):

    def __init__(
        self,
        model,
    ) -> None:

        self.model = model

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "ClassicalModel":

        self.model.fit(
            x,
            y,
        )

        return self

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        return self.model.predict(
            x,
        )

    def save(
        self,
        path: str,
    ) -> None:

        save_model(
            self.model,
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
    ):

        instance = cls.__new__(cls)

        instance.model = load_model(
            path,
        )

        return instance
