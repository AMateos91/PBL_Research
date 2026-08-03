from __future__ import annotations

import xarray as xr

from .base import Feature


class Albedo(Feature):

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        output = dataset.copy()

        variables = dataset.data_vars

        if {
            "blue",
            "green",
            "red",
            "nir",
            "swir",
            "swir2",
        }.issubset(variables):

            output["albedo"] = (
                0.356 * dataset["blue"]
                + 0.130 * dataset["green"]
                + 0.373 * dataset["red"]
                + 0.085 * dataset["nir"]
                + 0.072 * dataset["swir"]
                - 0.0018
            )

        output.attrs.update(
            dataset.attrs
        )

        output.attrs["albedo_features"] = True

        return output
