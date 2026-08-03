from __future__ import annotations

from typing import Any

import rioxarray
import xarray as xr

from .base import Preprocessing


class Reprojection(Preprocessing):

    def __init__(
        self,
        crs: str = "EPSG:4326",
        resolution: float | None = None,
        resampling: str = "nearest",
    ):

        super().__init__()

        self.crs = crs

        self.resolution = resolution

        self.resampling = resampling

    def process(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        kwargs: dict[str, Any] = {}

        if self.resolution is not None:

            kwargs["resolution"] = self.resolution

        dataset = dataset.rio.reproject(

            self.crs,

            resampling=self.resampling,

            **kwargs,

        )

        dataset.attrs["crs"] = self.crs

        return dataset
