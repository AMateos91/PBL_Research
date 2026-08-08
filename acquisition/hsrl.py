from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .hdf import HDFReader


class HSRLReader(HDFReader):

    VARIABLE_ALIASES = {
        "latitude": [
            "gps_lat",
            "lat",
            "latitude",
            "Latitude",
            "Lat",
            "/Nav_Data/gps_lat",
        ],
        "longitude": [
            "gps_lon",
            "lon",
            "longitude",
            "Longitude",
            "Lon",
            "/Nav_Data/gps_lon",
        ],
        "time": [
            "gps_time",
            "time",
            "Time",
            "UTC_Time",
            "/Nav_Data/gps_time",
        ],
        "altitude": [
            "gps_alt",
            "gpd_alt",
            "alt",
            "altitude",
            "Altitude",
            "GPSAltitude",
            "AircraftAltitude",
            "/DataProducts/DEM_alt",
        ],
        "cloud_height": [
            "cloud_height",
            "CLOUD_HEIGHT",
            "/DataProducts/cloud_height",
        ],
        "backscatter": [
            "bsNorm",
            "Backscatter",
            "backscatter",
            "Aerosol_Backscatter",
            "532_bsc",
            "/DataProducts/AB_prfl",
        ],
    }

    PROFILE_VARIABLES = {
        "backscatter": "/DataProducts/AB_prfl",
        "backscatter_prfl": "/DataProducts/AB_prfl",
        "attenuated_backscatter": "/DataProducts/AB_prfl",
        "iab_prfl": "/DataProducts/IAB_prfl",
        "iab": "/DataProducts/IAB",
        "lid_prfl": "/DataProducts/LID_prfl",
        "od_prfl": "/DataProducts/OD_prfl",
        "transp_prfl": "/DataProducts/TransP_prfl",
        "lid": "/DataProducts/LID",
        "sc": "/DataProducts/Sc",
        "surface_depol": "/DataProducts/SurfaceDepol",
        "reflectance": "/DataProducts/Reflectance",
        "reflectance_avg": "/DataProducts/ReflectanceAvg",
        "mult_scat_frac": "/DataProducts/MultScatFrac",
        "mult_scat_frac_prfl": "/DataProducts/MultScatFrac_prfl",
        "cloud_ext_prfl": "/DataProducts/cloud_ext_prfl",
        "trans_surface": "/DataProducts/TransSurface",
        "wind_speed_derived_cm": "/DataProducts/WindSpeedDerivedCM",
        "wind_speed_derived_hu": "/DataProducts/WindSpeedDerivedHU",
        "bsc_norm": "/DataProducts/bscNorm",
    }

    ONE_DIMENSIONAL_VARIABLES = {
        "latitude": "/lat",
        "longitude": "/lon",
        "time": "/time",
        "altitude": "/alt",
        "cloud_height": "/DataProducts/cloud_height",
        "dem_alt": "/DataProducts/DEM_alt",
        "selection_index": "/DataProducts/selection_index",
        "wind_speed_derived_cm": "/DataProducts/WindSpeedDerivedCM",
        "wind_speed_derived_hu": "/DataProducts/WindSpeedDerivedHU",
        "bsc_norm": "/DataProducts/bscNorm",
        "iab": "/DataProducts/IAB",
        "lid": "/DataProducts/LID",
        "reflectance": "/DataProducts/Reflectance",
        "reflectance_avg": "/DataProducts/ReflectanceAvg",
        "sc": "/DataProducts/Sc",
        "surface_depol": "/DataProducts/SurfaceDepol",
        "trans_surface": "/DataProducts/TransSurface",
        "mult_scat_frac": "/DataProducts/MultScatFrac",
        "cloud_ext_average": "/DataProducts/cloud_ext_average",
    }

    TIME_EPOCH = "2022-01-11T00:00:00Z"

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        super().__init__(
            path,
        )

    def _find_dataset(
        self,
        aliases: list[str],
    ) -> str | None:

        datasets = self.datasets

        normalized = {
            dataset.lower().rstrip("/"): dataset
            for dataset in datasets
        }

        for alias in aliases:

            alias_normalized = (
                alias.lower().rstrip("/")
            )

            if alias_normalized in normalized:

                return normalized[
                    alias_normalized
                ]

        for alias in aliases:

            alias_name = (
                alias.lower()
                .rstrip("/")
                .split("/")
                [-1]
            )

            for dataset in datasets:

                dataset_name = (
                    dataset.lower()
                    .rstrip("/")
                    .split("/")
                    [-1]
                )

                if dataset_name == alias_name:

                    return dataset

        return None

    def variable_map(
        self,
    ) -> dict[str, str | None]:

        mapping = {}

        for name, aliases in self.VARIABLE_ALIASES.items():

            mapping[name] = self._find_dataset(
                aliases,
            )

        for name, path in self.PROFILE_VARIABLES.items():

            if name not in mapping:

                mapping[name] = self._find_dataset(
                    [path],
                )

        return mapping

    def available_variables(
        self,
    ) -> list[str]:

        mapping = self.variable_map()

        return [
            name
            for name, path in mapping.items()
            if path is not None
        ]

    def _read_observation_variable(
        self,
        name: str,
        path: str,
        observation_length: int,
    ) -> np.ndarray | None:

        try:

            values = np.asarray(
                self.read(
                    path,
                )
            )

        except Exception:

            return None

        if values.ndim != 1:

            return None

        if values.size != observation_length:

            return None

        if name == "time":

            values = (
                self._convert_time(
                    values,
                )
            )

        return values

    def _read_profile_variable(
        self,
        name: str,
        path: str,
        observation_length: int,
    ) -> np.ndarray | None:

        try:

            values = np.asarray(
                self.read(
                    path,
                )
            )

        except Exception:

            return None

        if values.ndim != 2:

            return None

        if values.shape[0] != observation_length:

            return None

        return values

    def _convert_time(
        self,
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        return (
            values.astype("datetime64[s]")
            + np.datetime64(
                self.TIME_EPOCH,
            )
        )

    def _read_level_coordinate(
        self,
    ) -> np.ndarray:

        try:

            values = np.asarray(
                self.read(
                    "/z",
                )
            )

        except Exception:

            try:

                values = np.asarray(
                    self.read(
                        "z",
                    )
                )

            except Exception:

                return np.arange(
                    501,
                    dtype=np.int32,
                )

        return values

    def read_dataset(
        self,
    ) -> xr.Dataset:

        mapping = self.variable_map()

        cloud_height_path = mapping.get(
            "cloud_height",
        )

        if cloud_height_path is None:

            raise RuntimeError(
                "HSRL cloud_height variable was not found."
            )

        cloud_height = np.asarray(
            self.read(
                cloud_height_path,
            )
        )

        if cloud_height.ndim != 1:

            raise RuntimeError(
                "HSRL cloud_height is not a one-dimensional observation variable."
            )

        observation_length = int(
            cloud_height.size
        )

        if observation_length == 0:

            raise RuntimeError(
                "HSRL cloud_height contains no observations."
            )

        coordinates = {}

        variables = {}

        for name in (
            "latitude",
            "longitude",
            "time",
            "altitude",
            "cloud_height",
        ):

            path = mapping.get(
                name,
            )

            if path is None:

                continue

            values = self._read_observation_variable(
                name,
                path,
                observation_length,
            )

            if values is None:

                continue

            coordinates[
                name
            ] = (
                ("observation",),
                values,
            )

        level = self._read_level_coordinate()

        coordinates[
            "level"
        ] = (
            ("level",),
            level,
        )

        for name, path in mapping.items():

            if path is None:

                continue

            if name in {
                "latitude",
                "longitude",
                "time",
                "altitude",
                "cloud_height",
            }:

                continue

            values = self._read_profile_variable(
                name,
                path,
                observation_length,
            )

            if values is not None:

                if values.shape[1] == len(level):

                    variables[
                        name
                    ] = (
                        (
                            "observation",
                            "level",
                        ),
                        values,
                    )

        for name, path in mapping.items():

            if path is None:

                continue

            if name in variables:

                continue

            if name in {
                "latitude",
                "longitude",
                "time",
                "altitude",
                "cloud_height",
            }:

                continue

            try:

                values = np.asarray(
                    self.read(
                        path,
                    )
                )

            except Exception:

                continue

            if (
                values.ndim == 1
                and values.size == observation_length
            ):

                if name == "time":

                    values = self._convert_time(
                        values,
                    )

                variables[
                    name
                ] = (
                    (
                        "observation",
                    ),
                    values,
                )

        dataset = xr.Dataset(
            data_vars=variables,
            coords=coordinates,
            attrs={
                "source": str(
                    self.path,
                ),
                "instrument": "NASA HSRL-2",
                "observation_length": observation_length,
            },
        )

        return dataset

    def read_observations(
        self,
    ) -> xr.Dataset:

        return self.read_dataset()

    def __repr__(
        self,
    ) -> str:

        return (
            f"HSRLReader("
            f"file='{self.path}', "
            f"variables={len(self.available_variables())}"
            f")"
        )
