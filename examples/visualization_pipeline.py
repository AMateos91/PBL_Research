from __future__ import annotations

import numpy as np

from PBL_Research.visualization import (
    Plotter,
    ConfusionMatrix,
    FeatureImportance,
    TimeSeriesVisualizer,
    MapVisualizer,
)


def main() -> None:

    np.random.seed(42)

    plotter = Plotter()

    x = np.arange(100)
    y = np.random.randn(100)

    plotter.line(x, y)
    plotter.scatter(x, y)
    plotter.histogram(y)
    plotter.bar(
        ["A", "B", "C"],
        [1.5, 2.8, 1.2],
    )

    y_true = np.random.randint(
        0,
        2,
        200,
    )

    y_pred = np.random.randint(
        0,
        2,
        200,
    )

    ConfusionMatrix().plot(
        y_true,
        y_pred,
    )

    importance = np.random.rand(10)

    feature_names = [
        f"Feature {i}"
        for i in range(10)
    ]

    FeatureImportance().plot(
        importance,
        feature_names,
        top_k=10,
    )

    dates = np.arange(100)

    values = np.cumsum(
        np.random.randn(100)
    )

    TimeSeriesVisualizer().plot(
        dates,
        values,
        label="Synthetic signal",
    )

    image = np.random.rand(
        128,
        128,
    )

    MapVisualizer().plot(
        image,
        cmap="viridis",
        colorbar=True,
    )

    print(
        "Visualization pipeline completed successfully."
    )


if __name__ == "__main__":

    main()
