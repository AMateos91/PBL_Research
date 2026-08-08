from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

from .base import HDFReader


class HSRLReader(HDFReader):
    """
    Reader for NASA ACTIVATE HSRL-2 HDF5 observations.
    """

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
            "gps_altitude",
            "alt",
            "altitude",
            "Altitude",
            "AircraftAltitude",
            "GPSAltitude",
        ],
        "cloud_height": [
            "/DataProducts/cloud_height",
            "cloud_height",
            "CLOUD_HEIGHT",
        ],
        "backscatter": [
            "/DataProducts/bscNorm",
            "bscNorm",
            "Backscatter",
            "backscatter",
            "Aerosol_Backscatter",
            "532_bsc",
        ],
    }

    DIRECT_PATHS = {
        "latitude": "/Nav_Data/gps_lat",
        "longitude": "/Nav_Data/gps_lon",
        "time": "/Nav_Data/gps_time",
        "altitude": "/Nav_Data/gps_alt",
        "cloud_height": "/DataProducts/cloud_height",
        "backscatter": "/DataProducts/bscNorm",
    }

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        super().__init__(path)

    @staticmethod
    def _dataset_exists(
        file: h5py.File,
        path: str,
    ) -> bool:
        try:
            file[path]
            return True
        except KeyError:
            return False

    @staticmethod
    def _read_hdf5(
        file: h5py.File,
        path: str,
    ) -> np.ndarray:
        return np.asarray(file[path])

    def variable_map(self) -> dict[str, str | None]:
        mapping: dict[str, str | None] = {}

        with h5py.File(self.path, "r") as file:

            for name, direct_path in self.DIRECT_PATHS.items():

                if self._dataset_exists(
                    file,
                    direct_path,
                ):
                    mapping[name] = direct_path
                    continue

                mapping[name] = None

                aliases = self.VARIABLE_ALIASES.get(
                    name,
                    [],
                )

                for alias in aliases:

                    if alias.startswith("/"):
                        candidate = alias
                    else:
                        candidate = self._find_dataset(
                            file,
                            alias,
                        )

                    if candidate is not None:
                        mapping[name] = candidate
                        break

        return mapping

    @staticmethod
    def _find_dataset(
        file: h5py.File,
        name: str,
    ) -> str | None:

        result: list[str] = []

        def visitor(
            path: str,
            obj: Any,
        ) -> None:

            if not isinstance(
                obj,
                h5py.Dataset,
            ):
                return

            basename = path.split("/")[-1]

            if basename == name:
                result.append(
                    "/" + path
                    if not path.startswith("/")
                    else path
                )

        file.visititems(visitor)

        if result:
            return result[0]

        return None

    @staticmethod
    def _normalise_1d(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(values)

        if values.ndim == 1:
            return values

        if values.ndim == 2 and 1 in values.shape:
            return values.reshape(-1)

        return values

    @staticmethod
    def _observation_length(
        cloud_height: np.ndarray | None,
        latitude: np.ndarray | None,
        longitude: np.ndarray | None,
        time: np.ndarray | None,
    ) -> int | None:

        if cloud_height is not None:

            values = HSRLReader._normalise_1d(
                cloud_height,
            )

            if values.ndim == 1:
                return int(values.shape[0])

        candidates = (
            latitude,
            longitude,
            time,
        )

        for values in candidates:

            if values is None:
                continue

            values = HSRLReader._normalise_1d(
                values,
            )

            if values.ndim == 1:
                return int(values.shape[0])

        return None

    @staticmethod
    def _prepare_variable(
        name: str,
        values: np.ndarray,
        observation_length: int,
    ) -> tuple[str, tuple[str, ...], np.ndarray] | None:

        values = np.asarray(values)

        if name in {
            "latitude",
            "longitude",
            "time",
            "altitude",
            "cloud_height",
        }:

            values = HSRLReader._normalise_1d(
                values,
            )

            if (
                values.ndim == 1
                and len(values) == observation_length
            ):

                return (
                    name,
                    ("observation",),
                    values,
                )

        if values.ndim == 1:

            if len(values) == observation_length:

                return (
                    name,
                    ("observation",),
                    values,
                )

            return None

        if values.ndim == 2:

            if values.shape[0] == observation_length:

                return (
                    name,
                    (
                        "observation",
                        "level",
                    ),
                    values,
                )

            if values.shape[1] == observation_length:

                values = values.T

                return (
                    name,
                    (
                        "observation",
                        "level",
                    ),
                    values,
                )

            return None

        if values.ndim == 3:

            if values.shape[0] == observation_length:

                return (
                    name,
                    (
                        "observation",
                        "level",
                        "channel",
                    ),
                    values,
                )

            return None

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
                f"    {name}: {path}"
            )

        cloud_height = None
        latitude = None
        longitude = None
        time = None

        with h5py.File(
            self.path,
            "r",
        ) as file:

            if mapping.get(
                "cloud_height"
            ) is not None:

                cloud_height = self._read_hdf5(
                    file,
                    mapping["cloud_height"],
                )

            if mapping.get(
                "latitude"
            ) is not None:

                latitude = self._read_hdf5(
                    file,
                    mapping["latitude"],
                )

            if mapping.get(
                "longitude"
            ) is not None:

                longitude = self._read_hdf5(
                    file,
                    mapping["longitude"],
                )

            if mapping.get(
                "time"
            ) is not None:

                time = self._read_hdf5(
                    file,
                    mapping["time"],
                )

            observation_length = (
                self._observation_length(
                    cloud_height,
                    latitude,
                    longitude,
                    time,
                )
            )

            if observation_length is None:

                raise RuntimeError(
                    "Unable to determine HSRL observation length."
                )

            print(
                "\nHSRL observation length:",
                observation_length,
            )

            coordinates: dict[
                str,
                tuple[
                    tuple[str, ...],
                    np.ndarray,
                ],
            ] = {}

            variables: dict[
                str,
                tuple[
                    tuple[str, ...],
                    np.ndarray,
                ],
            ] = {}

            for name, path in mapping.items():

                if path is None:
                    continue

                try:

                    values = self._read_hdf5(
                        file,
                        path,
                    )

                except Exception as exc:

                    print(
                        f"Skipping {name}: {exc}"
                    )

                    continue

                prepared = self._prepare_variable(
                    name,
                    values,
                    observation_length,
                )

                if prepared is None:

                    print(
                        f"Skipping {name}: "
                        f"incompatible shape "
                        f"{np.asarray(values).shape}"
                    )

                    continue

                variable_name, dims, data = prepared

                if variable_name in {
                    "latitude",
                    "longitude",
                    "time",
                    "altitude",
                    "cloud_height",
                }:

                    coordinates[
                        variable_name
                    ] = (
                        dims,
                        data,
                    )

                else:

                    variables[
                        variable_name
                    ] = (
                        dims,
                        data,
                    )

        dataset = xr.Dataset(
            data_vars=variables,
            coords=coordinates,
            attrs={
                "source": str(self.path),
                "instrument": "NASA HSRL-2",
            },
        )

        return dataset

    def read(
        self,
        path: str,
    ) -> np.ndarray:

        with h5py.File(
            self.path,
            "r",
        ) as file:

            return self._read_hdf5(
                file,
                path,
            )
