from __future__ import annotations

import xarray as xr

from .base import Feature


class Heterogeneity(Feature):

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        output = dataset.copy()

        for variable in dataset.data_vars.values():

            name = variable.name

            output[
                f"{name}_mean"
            ] = variable.mean()

            output[
                f"{name}_std"
            ] = variable.std()

            output[
                f"{name}_variance"
            ] = variable.var()

            output[
                f"{name}_range"
            ] = (
                variable.max()
                - variable.min()
            )

        output.attrs.update(
            dataset.attrs
        )

        output.attrs[
            "heterogeneity_features"
        ] = True

        return output
