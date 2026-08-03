from __future__ import annotations

import xarray as xr

from .base import Preprocessing


class Stacking(Preprocessing):

    def __init__(
        self,
        compat: str = "override",
    ):

        super().__init__()

        self.compat = compat

    def process(
        self,
        *datasets: xr.Dataset,
    ) -> xr.Dataset:

        if len(datasets) == 0:

            raise ValueError(
                "At least one dataset is required."
            )

        dataset = xr.merge(

            datasets,

            compat=self.compat,

        )

        dataset.attrs["stacked"] = True

        return dataset

    def __call__(
        self,
        *datasets: xr.Dataset,
    ) -> xr.Dataset:

        return self.process(
            *datasets
        )
