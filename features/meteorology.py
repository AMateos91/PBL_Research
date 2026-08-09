from __future__ import annotations

import numpy as np
import xarray as xr

from .base import Feature
from ...utils.constants import Variable


class Meteorology(Feature):

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        output = dataset.copy()

        variables = dataset.data_vars

        if {
            Variable.U_WIND.value,
            Variable.V_WIND.value,
        }.issubset(variables):

            output[
                Variable.WIND_SPEED.value
            ] = np.sqrt(
                dataset[
                    Variable.U_WIND.value
                ] ** 2
                +
                dataset[
                    Variable.V_WIND.value
                ] ** 2
            )

            output[
                Variable.WIND_DIRECTION.value
            ] = (
                np.degrees(
                    np.arctan2(
                        dataset[
                            Variable.U_WIND.value
                        ],
                        dataset[
                            Variable.V_WIND.value
                        ],
                    )
                )
                + 360
            ) % 360

        if {
            Variable.AIR_TEMPERATURE.value,
            Variable.DEW_POINT_TEMPERATURE.value,
        }.issubset(variables):

            air = (
                dataset[
                    Variable.AIR_TEMPERATURE.value
                ] - 273.15
            )

            dew = (
                dataset[
                    Variable.DEW_POINT_TEMPERATURE.value
                ] - 273.15
            )

            saturation = (
                0.6108
                * np.exp(
                    17.27 * air
                    /
                    (
                        air + 237.3
                    )
                )
            )

            actual = (
                0.6108
                * np.exp(
                    17.27 * dew
                    /
                    (
                        dew + 237.3
                    )
                )
            )

            output[
                Variable.VPD.value
            ] = saturation - actual

        output.attrs.update(
            dataset.attrs
        )

        output.attrs[
            "meteorology_features"
        ] = True

        return output
