from __future__ import annotations

import xarray as xr

from .base import Feature


class Thermal(Feature):

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        output = dataset.copy()

        variables = dataset.data_vars

        if "thermal" in variables:

            output["surface_temperature"] = (
                dataset["thermal"]
            )

            output["temperature_anomaly"] = (
                dataset["thermal"]
                - dataset["thermal"].mean()
            )

        output.attrs.update(
            dataset.attrs
        )

        output.attrs["thermal_features"] = True

        return output
