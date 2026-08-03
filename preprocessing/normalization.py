from __future__ import annotations

import xarray as xr

from .base import Preprocessing


class Normalization(Preprocessing):

    def __init__(
        self,
        method: str = "minmax",
    ):

        super().__init__()

        self.method = method

    def process(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        normalized = xr.Dataset()

        for name, variable in dataset.data_vars.items():

            if self.method == "minmax":

                minimum = variable.min()

                maximum = variable.max()

                normalized[name] = (
                    variable - minimum
                ) / (maximum - minimum)

            elif self.method == "zscore":

                mean = variable.mean()

                std = variable.std()

                normalized[name] = (
                    variable - mean
                ) / std

            else:

                normalized[name] = variable

        normalized.attrs.update(
            dataset.attrs
        )

        normalized.attrs["normalized"] = True

        return normalized
