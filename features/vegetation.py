from __future__ import annotations

import xarray as xr

from .base import Feature


class Vegetation(Feature):

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        output = dataset.copy()

        variables = dataset.data_vars

        if {
            "red",
            "nir",
        }.issubset(variables):

            output["ndvi"] = (
                dataset["nir"] - dataset["red"]
            ) / (
                dataset["nir"] + dataset["red"] + 1e-10
            )

            output["savi"] = (
                1.5
                * (
                    dataset["nir"] - dataset["red"]
                )
                / (
                    dataset["nir"]
                    + dataset["red"]
                    + 0.5
                )
            )

        if {
            "blue",
            "red",
            "nir",
        }.issubset(variables):

            output["evi"] = (
                2.5
                * (
                    dataset["nir"] - dataset["red"]
                )
                / (
                    dataset["nir"]
                    + 6.0 * dataset["red"]
                    - 7.5 * dataset["blue"]
                    + 1.0
                )
            )

        output.attrs.update(
            dataset.attrs
        )

        output.attrs["vegetation_features"] = True

        return output
