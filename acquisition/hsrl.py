from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from .hdf import HDFReader


class HSRLReader(HDFReader):

    VARIABLE_PATHS = {
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
    def _read(
        file: h5py.File,
        path: str,
    ) -> np.ndarray:

        if path not in file:

            raise RuntimeError(
                f"HSRL variable not found: {path}"
            )

        return np.asarray(
            file[path][...]
        )

    @staticmethod
    def _normalise(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(values)

        if values.ndim == 0:
            return values.reshape(1)

        if values.ndim == 1:
            return values

        if values.ndim == 2:

            if values.shape[1] == 1:
                return values[:, 0]

            if values.shape[0] == 1:
                return values[0, :]

        raise RuntimeError(
            "Cannot normalise HSRL observation "
            f"with shape {values.shape}."
        )

    def variable_map(
        self,
    ) -> dict[str, str | None]:

        mapping = {}

        with h5py.File(
            self.path,
            "r",
        ) as file:

            for name, path in self.VARIABLE_PATHS.items():

                if path in file:
                    mapping[name] = path
                else:
                    mapping[name] = None

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

    def read_dataset(
        self,
    ) -> xr.Dataset:

        print(
            "\nHSRL variable mapping:"
        )

        mapping = self.variable_map()

        for name, path in mapping.items():

            print(
                f"    {name}: {path}"
            )

        required = [
            "latitude",
            "longitude",
            "time",
            "altitude",
            "cloud_height",
        ]

        missing = [
            name
            for name in required
            if mapping.get(name) is None
        ]

        if missing:

            raise RuntimeError(
                "Required HSRL variables could not be loaded: "
                + ", ".join(missing)
            )

        with h5py.File(
            self.path,
            "r",
        ) as file:

            raw_latitude = self._read(
                file,
                self.VARIABLE_PATHS["latitude"],
            )

            raw_longitude = self._read(
                file,
                self.VARIABLE_PATHS["longitude"],
            )

            raw_time = self._read(
                file,
                self.VARIABLE_PATHS["time"],
            )

            raw_altitude = self._read(
                file,
                self.VARIABLE_PATHS["altitude"],
            )

            raw_cloud_height = self._read(
                file,
                self.VARIABLE_PATHS["cloud_height"],
            )

            raw_backscatter = self._read(
                file,
                self.VARIABLE_PATHS["backscatter"],
            )

        latitude = self._normalise(
            raw_latitude
        )

        longitude = self._normalise(
            raw_longitude
        )

        time = self._normalise(
            raw_time
        )

        altitude = self._normalise(
            raw_altitude
        )

        cloud_height = self._normalise(
            raw_cloud_height
        )

        backscatter = self._normalise(
            raw_backscatter
        )

        lengths = {
            "latitude": latitude.size,
            "longitude": longitude.size,
            "time": time.size,
            "altitude": altitude.size,
            "cloud_height": cloud_height.size,
            "backscatter": backscatter.size,
        }

        print(
            "\nHSRL variable shapes:"
        )

        for name, size in lengths.items():

            print(
                f"    {name}: ({size},)"
            )

        observation_length = latitude.size

        for name, size in lengths.items():

            if size != observation_length:

                raise RuntimeError(
                    "Conflicting HSRL observation lengths: "
                    f"latitude={observation_length}, "
                    f"{name}={size}"
                )

        print(
            "\nHSRL observation length:",
            observation_length,
        )

        dataset = xr.Dataset(

            data_vars={
                "latitude": (
                    ("observation",),
                    latitude,
                ),
                "longitude": (
                    ("observation",),
                    longitude,
                ),
                "time": (
                    ("observation",),
                    time,
                ),
                "altitude": (
                    ("observation",),
                    altitude,
                ),
                "cloud_height": (
                    ("observation",),
                    cloud_height,
                ),
                "backscatter": (
                    ("observation",),
                    backscatter,
                ),
            },

            coords={
                "observation": np.arange(
                    observation_length
                ),
            },

            attrs={
                "source": str(
                    self.path
                ),
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

            return self._read(
                file,
                path,
            )
