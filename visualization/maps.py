from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .visualizer import Visualizer


class MapVisualizer(Visualizer):

    def plot(
        self,
        image: np.ndarray,
        cmap: str = "viridis",
        colorbar: bool = True,
        **kwargs,
    ) -> None:

        plt.figure()

        im = plt.imshow(
            image,
            cmap=cmap,
            **kwargs,
        )

        if colorbar:

            plt.colorbar(
                im,
            )

        plt.tight_layout()

        plt.show()
