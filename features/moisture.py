from __future__ import annotations

import xarray as xr

from .base import Feature


class Moisture(Feature):

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        output = dataset.copy()

        variables = dataset.data_vars

        if {
            "green",
            "nir",
        }.issubset(variables):

            output["ndwi"] = (
                dataset["green"] - dataset["nir"]
            ) / (
                dataset["green"] + dataset["nir"] + 1e-10
            )

        if {
            "nir",
            "swir",
        }.issubset(variables):

            output["ndmi"] = (
                dataset["nir"] - dataset["swir"]
            ) / (
                dataset["nir"] + dataset["swir"] + 1e-10
            )

            output["msi"] = (
                dataset["swir"]
                / (
                    dataset["nir"] + 1e-10
                )
            )

        output.attrs.update(
            dataset.attrs
        )

        output.attrs["moisture_features"] = True

        return output
