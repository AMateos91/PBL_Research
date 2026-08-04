from __future__ import annotations

import numpy as np
import shap

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

        self.explainer = shap.Explainer(
            model.predict,
            background,
        )

    def explain(
        self,
        x: np.ndarray,
    ):

        return self.explainer(
            x,
        )
