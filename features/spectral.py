from __future__ import annotations

import xarray as xr

from .base import Feature


class Spectral(Feature):

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
            "swir",
            "nir",
        }.issubset(variables):

            output["ndmi"] = (
                dataset["nir"] - dataset["swir"]
            ) / (
                dataset["nir"] + dataset["swir"] + 1e-10
            )

        if {
            "swir",
            "nir",
        }.issubset(variables):

            output["ndbi"] = (
                dataset["swir"] - dataset["nir"]
            ) / (
                dataset["swir"] + dataset["nir"] + 1e-10
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
                    + 6 * dataset["red"]
                    - 7.5 * dataset["blue"]
                    + 1
                )
            )

        output.attrs.update(
            dataset.attrs
        )

        output.attrs["spectral_features"] = True

        return output
