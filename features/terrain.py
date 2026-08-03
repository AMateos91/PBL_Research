from __future__ import annotations

import numpy as np
import xarray as xr

from .base import Feature
from ..utils.constants import Variable


class Terrain(Feature):

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        output = dataset.copy()

        if Variable.ELEVATION.value not in dataset:

            return output

        elevation = dataset[
            Variable.ELEVATION.value
        ]

        dy, dx = np.gradient(
            elevation.values
        )

        slope = np.arctan(
            np.sqrt(
                dx ** 2 + dy ** 2
            )
        )

        aspect = np.arctan2(
            -dx,
            dy,
        )

        output[
            Variable.SLOPE.value
        ] = xr.DataArray(

            slope,

            coords=elevation.coords,

            dims=elevation.dims,

        )

        output[
            Variable.ASPECT.value
        ] = xr.DataArray(

            aspect,

            coords=elevation.coords,

            dims=elevation.dims,

        )

        output.attrs.update(
            dataset.attrs
        )

        output.attrs[
            "terrain_features"
        ] = True

        return output
