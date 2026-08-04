from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .visualizer import Visualizer


class FeatureImportance(Visualizer):

    def plot(
        self,
        importance: np.ndarray,
        feature_names: list[str],
        top_k: int | None = None,
    ) -> None:

        importance = np.asarray(
            importance,
        )

        order = np.argsort(
            importance,
        )[::-1]

        if top_k is not None:

            order = order[:top_k]

        plt.figure()

        plt.barh(

            np.array(
                feature_names,
            )[order][::-1],

            importance[
                order
            ][::-1],

        )

        plt.xlabel(
            "Importance",
        )

        plt.tight_layout()

        plt.show()
