from __future__ import annotations

import numpy as np
import torch

from ..base import Model


class Predictor:

    def __init__(
        self,
        model: Model,
    ) -> None:

        self.model = model

    def predict(
        self,
        x,
    ):

        if isinstance(
            x,
            np.ndarray,
        ):

            return self.model.predict(
                x,
            )

        if isinstance(
            x,
            torch.Tensor,
        ):

            self.model.model.eval()

            with torch.no_grad():

                return self.model.predict(
                    x,
                )

        raise TypeError(
            f"Unsupported input type: {type(x)}"
        )
