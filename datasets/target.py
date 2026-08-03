from __future__ import annotations

import xarray as xr

from .base import DatasetBuilder


class Target(DatasetBuilder):

    def __init__(
        self,
        variable: str,
    ):

        self.variable = variable

    def build(
        self,
        dataset: xr.Dataset,
    ) -> xr.DataArray:

        return dataset[self.variable]
