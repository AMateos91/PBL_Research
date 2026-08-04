from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .visualizer import Visualizer


class TimeSeriesVisualizer(Visualizer):

    def plot(
        self,
        dates,
        values: np.ndarray,
        label: str | None = None,
        **kwargs,
    ) -> None:

        plt.figure()

        plt.plot(
            dates,
            values,
            label=label,
            **kwargs,
        )

        if label is not None:

            plt.legend()

        plt.tight_layout()

        plt.show()
