from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .visualizer import Visualizer


class Plotter(Visualizer):

    def line(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> None:

        plt.figure()

        plt.plot(
            x,
            y,
            **kwargs,
        )

        plt.tight_layout()

        plt.show()

    def scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> None:

        plt.figure()

        plt.scatter(
            x,
            y,
            **kwargs,
        )

        plt.tight_layout()

        plt.show()

    def histogram(
        self,
        x: np.ndarray,
        bins: int = 30,
        **kwargs,
    ) -> None:

        plt.figure()

        plt.hist(
            x,
            bins=bins,
            **kwargs,
        )

        plt.tight_layout()

        plt.show()

    def bar(
        self,
        labels,
        values,
        **kwargs,
    ) -> None:

        plt.figure()

        plt.bar(
            labels,
            values,
            **kwargs,
        )

        plt.tight_layout()

        plt.show()

    def plot(
        self,
        *args,
        **kwargs,
    ) -> None:

        self.line(
            *args,
            **kwargs,
        )
