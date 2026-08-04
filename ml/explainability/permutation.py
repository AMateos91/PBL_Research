from __future__ import annotations

import numpy as np

from sklearn.inspection import permutation_importance

from .explainer import Explainer


class PermutationImportance(Explainer):

    def explain(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ):

        return permutation_importance(

            estimator=self.model.model,

            X=x,

            y=y,

            **kwargs,

        )
