import numpy as np

from pbl_research.preprocessing.scaling import Scaling


def test_scaling():

    x = np.random.rand(20, 4)

    scaler = Scaling()

    y = scaler.fit_transform(x)

    assert y.shape == x.shape
