import numpy as np

from pbl_research.visualization.plots import Plotter


def test_plotter():

    plotter = Plotter()

    x = np.arange(10)

    y = x

    plotter.line(
        x,
        y,
    )

    assert True
