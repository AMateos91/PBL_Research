from __future__ import annotations

import numpy as np
import torch

from ..neural.base import NeuralModel
from .explainer import Explainer


class PermutationImportance(Explainer):

    def explain(
        self,
        x: np.ndarray,
        y: np.ndarray,
        scoring=None,
        n_repeats: int = 5,
        random_state: int | None = None,
    ) -> dict[str, np.ndarray]:

        rng = np.random.default_rng(random_state)

        x = np.asarray(x)
        y = np.asarray(y)

        baseline_prediction = self._predict(x)
        baseline_score = self._score(
            y,
            baseline_prediction,
            scoring,
        )

        importances = np.zeros(
            x.shape[1],
            dtype=float,
        )

        for feature in range(x.shape[1]):

            scores = []

            for _ in range(n_repeats):

                x_permuted = x.copy()

                rng.shuffle(
                    x_permuted[:, feature],
                )

                prediction = self._predict(
                    x_permuted,
                )

                score = self._score(
                    y,
                    prediction,
                    scoring,
                )

                scores.append(
                    score - baseline_score,
                )

            importances[feature] = np.mean(
                scores,
            )

        return {
            "importances": importances,
            "importances_mean": importances,
            "importances_std": np.zeros_like(importances),
        }

    def _predict(
        self,
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
            ).squeeze()

        prediction = self.model.predict(
            x,
        )

        return np.asarray(
            prediction,
        ).squeeze()

    def _score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scoring,
    ) -> float:

        if scoring is not None:

            return scoring(
                y_true,
                y_pred,
            )

        return np.mean(
            (
                y_true - y_pred
            ) ** 2
        )
