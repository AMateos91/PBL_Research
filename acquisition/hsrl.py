from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .hdf import HDFReader


class HSRLReader(HDFReader):

    VARIABLE_ALIASES = {

        "latitude": [
            "latitude",
            "lat",
            "Latitude",
            "Lat",
            "gps_lat",
        ],

        "longitude": [
            "longitude",
            "lon",
            "Longitude",
            "Lon",
            "gps_lon",
        ],

        "time": [
            "time",
            "Time",
            "UTC_Time",
            "gps_time",
        ],

        "altitude": [
            "altitude",
            "Altitude",
            "GPSAltitude",
            "AircraftAltitude",
            "alt",
            "gpd_alt",
        ],

        "cloud_height": [
            "cloud_height",
            "CloudHeight",
            "Cloud_Height",
        ],

        "backscatter": [
            "Backscatter",
            "backscatter",
            "Aerosol_Backscatter",
            "532_bsc",
            "AB_prfl",
            "bscNorm",
        ],

        "cloud_extinction": [
            "cloud_ext_prfl",
            "cloud_ext_average",
            "CloudExtinction",
            "Cloud_Extinction",
        ],

        "optical_depth": [
            "OD",
            "optical_depth",
            "OpticalDepth",
        ],

        "reflectance": [
            "Reflectance",
            "ReflectanceAvg",
        ],

        "wind_speed": [
            "WindSpeedDerivedCM",
            "WindSpeedDerivedHU",
            "WindSpeed",
            "wind_speed",
        ],

    }

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        super().__init__(
            path,
        )

    def find_variable(
        self,
        aliases: list[str],
    ) -> str | None:

        datasets = self.datasets

        for alias in aliases:

            alias_lower = alias.lower()

            for dataset in datasets:

                dataset_lower = dataset.lower()

                if (
                    dataset_lower == alias_lower
                    or dataset_lower.endswith(
                        f"/{alias_lower}"
                    )
                    or alias_lower in dataset_lower
                ):

                    return dataset

        return None

    def variable_map(
        self,
    ) -> dict[str, str | None]:

        mapping = {}

        for name, aliases in self.VARIABLE_ALIASES.items():

            mapping[name] = self.find_variable(
                aliases,
            )

        return mapping

    def read_dataset(
        self,
    ) -> xr.Dataset:

        mapping = self.variable_map()

        coordinates = {}
        variables = {}

        for name, path in mapping.items():

            if path is None:
                continue

            try:

                values = np.asarray(
                    self.read(
                        path,
                    )
                )

            except Exception:

                continue

            if values.ndim == 0:

                variables[name] = values.item()

                continue

            if values.ndim == 1:

                if name in {
                    "latitude",
                    "longitude",
                    "time",
                    "altitude",
                }:

                    coordinates[name] = (
                        "observation",
                        values,
                    )

                else:

                    variables[name] = (
                        "observation",
                        values,
                    )

                continue

            if values.ndim == 2:

                variables[name] = (
                    (
                        "observation",
                        "level",
                    ),
                    values,
                )

                continue

            if values.ndim == 3:

                variables[name] = (
                    (
                        "observation",
                        "level",
                        "channel",
                    ),
                    values,
                )

                continue

        dataset = xr.Dataset(
            data_vars=variables,
            coords=coordinates,
            attrs={
                "source": str(
                    self.path,
                ),
                "instrument": "NASA HSRL-2",
            },
        )

        return dataset

    def available_variables(
        self,
    ) -> list[str]:

        return [
            name
            for name, path in self.variable_map().items()
            if path is not None
        ]

    def has_variable(
        self,
        name: str,
    ) -> bool:

        return (
            name
            in self.available_variables()
        )

    def get(
        self,
        name: str,
    ) -> np.ndarray:

        mapping = self.variable_map()

        if name not in mapping:

            raise KeyError(
                name,
            )

        if mapping[name] is None:

            raise KeyError(
                name,
            )

        return np.asarray(
            self.read(
                mapping[name],
            )
        )
