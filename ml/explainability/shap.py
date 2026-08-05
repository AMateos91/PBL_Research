from __future__ import annotations

import numpy as np
import shap
import torch

from ..neural.base import NeuralModel
from .explainer import Explainer


class SHAPExplainer(Explainer):

    def __init__(
        self,
        model,
        background: np.ndarray,
    ) -> None:

        super().__init__(
            model,
        )

        def predict_fn(
            x: np.ndarray,
        ) -> np.ndarray:

            if isinstance(
                self.model,
                NeuralModel,
            ):

                tensor = torch.as_tensor(
                    x,
                    dtype=torch.float32,
                )

                prediction = self.model.predict(
                    tensor,
                )

                if isinstance(
                    prediction,
                    torch.Tensor,
                ):

                    prediction = (
                        prediction.detach()
                        .cpu()
                        .numpy()
                    )

                return np.asarray(
                    prediction,
                )

            return np.asarray(
                self.model.predict(
                    x,
                )
            )

        self.explainer = shap.Explainer(
            predict_fn,
            background,
        )

    def explain(
        self,
        x: np.ndarray,
    ):

        return self.explainer(
            x,
        )
