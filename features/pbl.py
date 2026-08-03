from __future__ import annotations

import xarray as xr

from .base import Feature
from ..utils.constants import Variable


class PBL(Feature):

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        output = dataset.copy()

        variables = dataset.data_vars

        if {
            Variable.SURFACE_TEMPERATURE.value,
            Variable.AIR_TEMPERATURE.value,
        }.issubset(variables):

            output[
                Variable.TEMPERATURE_GRADIENT.value
            ] = (
                dataset[
                    Variable.SURFACE_TEMPERATURE.value
                ]
                -
                dataset[
                    Variable.AIR_TEMPERATURE.value
                ]
            )

        if {
            Variable.SENSIBLE_HEAT_FLUX.value,
            Variable.LATENT_HEAT_FLUX.value,
        }.issubset(variables):

            output[
                Variable.BOWEN_RATIO.value
            ] = (
                dataset[
                    Variable.SENSIBLE_HEAT_FLUX.value
                ]
                /
                (
                    dataset[
                        Variable.LATENT_HEAT_FLUX.value
                    ]
                    + 1e-10
                )
            )

        if {
            Variable.WIND_SPEED.value,
            Variable.SURFACE_TEMPERATURE.value,
        }.issubset(variables):

            output[
                Variable.THERMAL_ADVECTION_PROXY.value
            ] = (
                dataset[
                    Variable.WIND_SPEED.value
                ]
                *
                dataset[
                    Variable.SURFACE_TEMPERATURE.value
                ]
            )

        output.attrs.update(
            dataset.attrs
        )

        output.attrs[
            "pbl_features"
        ] = True

        return output
