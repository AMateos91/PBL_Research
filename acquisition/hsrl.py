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
        ],

        "longitude": [
            "gps_lon",
            "lon",
            "longitude",
            "Longitude",
            "Lon",
        ],

        "time": [
            "gps_time",
            "time",
            "Time",
            "UTC_Time",
        ],

        "altitude": [
            "gps_alt",
            "gpd_alt",
            "alt",
            "altitude",
            "Altitude",
            "GPSAltitude",
            "AircraftAltitude",
        ],

        "cloud_height": [
            "cloud_height",
            "CLOUD_HEIGHT",
            "/DataProducts/cloud_height",
        ],

        "backscatter": [
            "bscNorm",
            "Backscatter",
            "backscatter",
            "Aerosol_Backscatter",
            "532_bsc",
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

        normalized = {
            str(dataset).strip().lower(): str(dataset)
            for dataset in datasets
        }

        for alias in aliases:

            alias_normalized = (
                alias.strip().lower()
            )

            if alias_normalized in normalized:

                return normalized[
                    alias_normalized
                ]

        for alias in aliases:

            alias_normalized = (
                alias.strip().lower()
            )

            for dataset in datasets:

                dataset_normalized = (
                    str(dataset)
                    .strip()
                    .lower()
                )

                if dataset_normalized.endswith(
                    "/" + alias_normalized
                ):

                    return str(dataset)

        for alias in aliases:

            alias_normalized = (
                alias.strip().lower()
            )

            for dataset in datasets:

                dataset_normalized = (
                    str(dataset)
                    .strip()
                    .lower()
                )

                if alias_normalized in dataset_normalized:

                    return str(dataset)

        return None

    def variable_map(
        self,
    ) -> dict[str, str | None]:

        mapping = {}

        for name, aliases in (
            self.VARIABLE_ALIASES.items()
        ):

            mapping[name] = self.find_variable(
                aliases,
            )

        return mapping

    @staticmethod
    def _safe_array(
        values,
    ) -> np.ndarray:

        return np.asarray(
            values,
        )

    @staticmethod
    def _dimension_name(
        name: str,
        ndim: int,
    ) -> tuple[str, ...]:

        if ndim == 0:

            return ()

        if ndim == 1:

            return (
                "observation",
            )

        if ndim == 2:

            return (
                "observation",
                "level",
            )

        if ndim == 3:

            return (
                "observation",
                "level",
                "channel",
            )

        return tuple(
            f"dim_{index}"
            for index in range(ndim)
        )

    @staticmethod
    def _is_scalar(
        values: np.ndarray,
    ) -> bool:

        return values.ndim == 0

    @staticmethod
    def _is_singleton(
        values: np.ndarray,
    ) -> bool:

        return values.size == 1

    @staticmethod
    def _valid_observation_length(
        values: np.ndarray,
        observation_length: int | None,
    ) -> bool:

        if values.ndim == 0:

            return True

        if values.size == 1:

            return True

        if values.ndim == 1:

            if observation_length is None:

                return True

            return (
                len(values)
                == observation_length
            )

        if values.ndim >= 2:

            if observation_length is None:

                return True

            return (
                values.shape[0]
                == observation_length
            )

        return False

    @staticmethod
    def _convert_time(
        values: np.ndarray,
    ) -> np.ndarray:

        if values.size == 0:

            return values

        if np.issubdtype(
            values.dtype,
            np.datetime64,
        ):

            return values

        if not np.issubdtype(
            values.dtype,
            np.number,
        ):

            return values

        finite = np.isfinite(
            values.astype(
                np.float64,
                copy=False,
            )
        )

        if not finite.any():

            return values

        sample = np.abs(
            values.astype(
                np.float64,
                copy=False,
            )[finite]
        )

        magnitude = float(
            np.nanmedian(
                sample,
            )
        )

        if magnitude > 1e17:

            return values.astype(
                "datetime64[ns]"
            )

        if magnitude > 1e14:

            return (
                values.astype(
                    "datetime64[us]"
                )
                .astype(
                    "datetime64[ns]"
                )
            )

        if magnitude > 1e11:

            return (
                values.astype(
                    "datetime64[ms]"
                )
                .astype(
                    "datetime64[ns]"
                )
            )

        if magnitude > 1e9:

            return (
                (
                    values.astype(
                        np.float64,
                    )
                    * 1e9
                )
                .astype(
                    "datetime64[ns]"
                )
            )

        return values

    def _read_variable(
        self,
        path: str,
    ) -> np.ndarray | None:

        try:

            values = self.read(
                path,
            )

        except Exception as exc:

            print(
                f"Unable to read {path}: {exc}"
            )

            return None

        try:

            return self._safe_array(
                values,
            )

        except Exception as exc:

            print(
                f"Unable to convert {path}: {exc}"
            )

            return None

    def read_dataset(
        self,
    ) -> xr.Dataset:

        mapping = self.variable_map()

        print(
            "\nHSRL variable mapping:"
        )

        for name, path in mapping.items():

            print(
                f"  {name}: {path}"
            )

        coordinates = {}

        variables = {}

        observation_length = None

        for name in (
            "latitude",
            "longitude",
            "time",
        ):

            path = mapping.get(
                name,
            )

            if path is None:

                continue

            values = self._read_variable(
                path,
            )

            if values is None:

                continue

            if values.ndim == 1 and values.size > 1:

                observation_length = (
                    len(values)
                )

                break

        if observation_length is None:

            raise RuntimeError(
                "Unable to determine HSRL observation length."
            )

        for name, path in mapping.items():

            if path is None:

                continue

            values = self._read_variable(
                path,
            )

            if values is None:

                continue

            if self._is_scalar(
                values,
            ):

                variables[name] = values

                continue

            if self._is_singleton(
                values,
            ):

                if (
                    name in {
                        "latitude",
                        "longitude",
                        "time",
                        "altitude",
                    }
                ):

                    print(
                        f"Skipping singleton coordinate "
                        f"{name}: {path}"
                    )

                    continue

                variables[name] = values.squeeze()

                continue

            if not self._valid_observation_length(
                values,
                observation_length,
            ):

                print(
                    f"Skipping incompatible variable "
                    f"{name}: {path} "
                    f"shape={values.shape} "
                    f"observation_length="
                    f"{observation_length}"
                )

                continue

            if name == "time":

                values = self._convert_time(
                    values,
                )

            dimensions = self._dimension_name(
                name,
                values.ndim,
            )

            if (
                name in {
                    "latitude",
                    "longitude",
                    "time",
                    "altitude",
                }
                and values.ndim == 1
            ):

                if len(values) != observation_length:

                    print(
                        f"Skipping incompatible coordinate "
                        f"{name}: {path} "
                        f"shape={values.shape}"
                    )

                    continue

                coordinates[name] = (
                    dimensions,
                    values,
                )

            else:

                variables[name] = (
                    dimensions,
                    values,
                )

        if "latitude" not in coordinates:

            raise RuntimeError(
                "HSRL latitude could not be loaded."
            )

        if "longitude" not in coordinates:

            raise RuntimeError(
                "HSRL longitude could not be loaded."
            )

        if "time" not in coordinates:

            raise RuntimeError(
                "HSRL time could not be loaded."
            )

        if "cloud_height" not in variables:

            raise RuntimeError(
                "HSRL cloud_height could not be loaded."
            )

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

        mapping = self.variable_map()

        return [
            name
            for name, path in mapping.items()
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

        path = mapping[name]

        if path is None:

            raise KeyError(
                name,
            )

        values = self.read(
            path,
        )

        return np.asarray(
            values,
        )
