from __future__ import annotations

from typing import Any

import xarray as xr

from .base import Preprocessing

from ...utils.constants import ResamplingMethod


class Resampling(Preprocessing):

    def __init__(
        self,
        resolution: float,
        resampling: str = ResamplingMethod.NEAREST.value,
    ):

        super().__init__()

        self.resolution = resolution

        self.resampling = resampling

    def process(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        dataset = dataset.rio.reproject(

            dataset.rio.crs,

            resolution=self.resolution,

            resampling=self.resampling,

        )

        dataset.attrs["resolution"] = self.resolution

        return dataset
