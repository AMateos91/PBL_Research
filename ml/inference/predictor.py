from __future__ import annotations

import numpy as np
import torch

from ..base import Model


class Predictor:
    """
    Generic predictor for classical and neural models.
    """

    def __init__(
        self,
        model: Model,
    ) -> None:

        self.model = model

    def predict(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """
        Generate predictions.
        """

        if isinstance(x, np.ndarray):
            return self.model.predict(x)

        if isinstance(x, torch.Tensor):

            self.model.eval()

            with torch.no_grad():
                return self.model.predict(x)

        raise TypeError(
            f"Unsupported input type: {type(x).__name__}"
        )
