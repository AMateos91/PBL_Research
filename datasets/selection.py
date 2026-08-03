from __future__ import annotations

import xarray as xr

from .base import DatasetBuilder


class Selection(DatasetBuilder):

    def __init__(
        self,
        variables: list[str],
    ):

        self.variables = variables

    def build(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        return dataset[self.variables]
