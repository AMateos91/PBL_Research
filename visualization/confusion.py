from __future__ import annotations

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from .visualizer import Visualizer


class ConfusionMatrix(Visualizer):

    def plot(
        self,
        y_true,
        y_pred,
        labels=None,
        cmap: str = "Blues",
    ) -> None:

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=labels,
        ).plot(
            cmap=cmap,
        )

        plt.tight_layout()

        plt.show()
