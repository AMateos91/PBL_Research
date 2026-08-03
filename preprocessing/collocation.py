from __future__ import annotations

import xarray as xr

from .base import Preprocessing


class Collocation(Preprocessing):

    def __init__(
        self,
        join: str = "inner",
        compat: str = "override",
    ):

        super().__init__()

        self.join = join

        self.compat = compat

    def process(
        self,
        *datasets: xr.Dataset,
    ) -> xr.Dataset:

        if len(datasets) == 0:

            raise ValueError(
                "At least one dataset is required."
            )

        dataset = xr.align(

            *datasets,

            join=self.join,

        )

        dataset = xr.merge(

            dataset,

            compat=self.compat,

        )

        dataset.attrs["collocated"] = True

        return dataset

    def __call__(
        self,
        *datasets: xr.Dataset,
    ) -> xr.Dataset:

        return self.process(
            *datasets
        )
