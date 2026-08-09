from __future__ import annotations

import xarray as xr

from .base import Preprocessing

from ...utils.constants import Compatibility


class Stacking(Preprocessing):

    def __init__(
        self,
        compat: str = Compatibility.OVERRIDE.value
    ):

        super().__init__()

        self.compat = compat

    def process(
        self,
        *datasets: xr.Dataset
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
