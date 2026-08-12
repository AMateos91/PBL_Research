from __future__ import annotations

import numpy as np

from ..base import Model
from ..utils.constants import EnsembleMethod


class Ensemble:

    def __init__(
        self,
        models: list[Model],
        method: EnsembleMethod = EnsembleMethod.MEAN,
        weights: list[float] | None = None,
    ) -> None:

        if not models:

            raise ValueError(
                "At least one model is required."
            )

        if (
            method == EnsembleMethod.WEIGHTED
            and weights is not None
            and len(weights) != len(models)
        ):

            raise ValueError(
                "The number of weights must match the number of models."
            )

        self.models = models
        self.method = method
        self.weights = weights

    def predict(
        self,
        x,
    ) -> np.ndarray:

        predictions = np.asarray(
            [
                model.predict(
                    x,
                )
                for model in self.models
            ]
        )

        match self.method:

            case EnsembleMethod.MEAN:

                return np.mean(
                    predictions,
                    axis=0,
                )

            case EnsembleMethod.MEDIAN:

                return np.median(
                    predictions,
                    axis=0,
                )

            case EnsembleMethod.WEIGHTED:

                if self.weights is None:

                    raise ValueError(
                        "Weights are required for a weighted ensemble."
                    )

                return np.average(
                    predictions,
                    axis=0,
                    weights=self.weights,
                )

            case EnsembleMethod.VOTING:

                predictions = np.rint(
                    predictions,
                ).astype(
                    int,
                )

                return np.apply_along_axis(
                    lambda values: np.bincount(
                        values,
                    ).argmax(),
                    axis=0,
                    arr=predictions,
                )

            case _:

                raise ValueError(
                    f"Unknown ensemble method: {self.method}"
                )
