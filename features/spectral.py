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
            "nir",
            "swir",
        }.issubset(variables):

            output["ndbi"] = (
                dataset["swir"] - dataset["nir"]
            ) / (
                dataset["swir"] + dataset["nir"] + 1e-10
            )

        output.attrs.update(
            dataset.attrs
        )

        output.attrs["spectral_features"] = True

        return output
