import numpy as np

from pbl_research.features.spectral import NDVI


def test_ndvi():

    nir = np.ones((10, 10))

    red = np.ones((10, 10))

    ndvi = NDVI()

    result = ndvi.compute(
        nir,
        red,
    )

    assert result.shape == nir.shape
