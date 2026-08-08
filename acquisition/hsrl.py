from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from .hdf import HDFReader


class HSRLReader(HDFReader):

    VARIABLE_ALIASES = {
        "latitude": [
            "latitude",
            "gps_lat",
            "lat",
            "Latitude",
            "Lat",
            "/Nav_Data/gps_lat",
        ],
        "longitude": [
            "longitude",
            "gps_lon",
            "lon",
            "Longitude",
            "Lon",
            "/Nav_Data/gps_lon",
        ],
        "time": [
            "time",
            "gps_time",
            "Time",
            "UTC_Time",
            "/Nav_Data/gps_time",
        ],
        "altitude": [
            "altitude",
            "gps_alt",
            "gps_altitude",
            "alt",
            "Altitude",
            "GPSAltitude",
            "AircraftAltitude",
            "/Nav_Data/gps_alt",
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
            "/DataProducts/bscNorm",
        ],
    }

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        super().__init__(path)

    def variable_map(self) -> dict[str, str | None]:

        mapping = {}

        datasets = set(self.datasets)

        for name, aliases in self.VARIABLE_ALIASES.items():

            selected = None

            for alias in aliases:

                if alias.startswith("/"):

                    candidate = alias

                else:

                    candidate = f"/{alias}"

                if candidate in datasets:
                    selected = candidate
                    break

                if alias in datasets:
                    selected = alias
                    break

            mapping[name] = selected

        return mapping

    def available_variables(self) -> list[str]:

        mapping = self.variable_map()

        return [
            name
            for name, path in mapping.items()
            if path is not None
        ]

    @staticmethod
    def _normalise_observation(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(values)

        if values.ndim == 0:
            return values.reshape(1)

        if values.ndim == 1:
            return values

        if values.ndim == 2:

            if 1 in values.shape:
                return values.reshape(-1)

        raise ValueError(
            f"Cannot normalise HSRL observation with shape "
            f"{values.shape}."
        )

    def _read_hsrl_variable(
        self,
        file: h5py.File,
        path: str | None,
    ) -> np.ndarray | None:

        if path is None:
            return None

        if path not in file:
            return None

        values = np.asarray(
            file[path][...]
        )

        return self._normalise_observation(
            values
        )

    def read_dataset(
        self,
    ) -> xr.Dataset:

        mapping = self.variable_map()

        print("\nHSRL variable mapping:")

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
                "Required HSRL variables not found: "
                + ", ".join(missing)
            )

        with h5py.File(
            self.path,
            "r",
        ) as file:

            latitude = self._read_hsrl_variable(
                file,
                mapping["latitude"],
            )

            longitude = self._read_hsrl_variable(
                file,
                mapping["longitude"],
            )

            time = self._read_hsrl_variable(
                file,
                mapping["time"],
            )

            altitude = self._read_hsrl_variable(
                file,
                mapping["altitude"],
            )

            cloud_height = self._read_hsrl_variable(
                file,
                mapping["cloud_height"],
            )

            backscatter = self._read_hsrl_variable(
                file,
                mapping["backscatter"],
            )

        values = {
            "latitude": latitude,
            "longitude": longitude,
            "time": time,
            "altitude": altitude,
            "cloud_height": cloud_height,
            "backscatter": backscatter,
        }

        for name, value in values.items():

            if value is None:
                continue

            print(
                f"{name}: shape={value.shape}"
            )

        observation_length = None

        for name in [
            "latitude",
            "longitude",
            "time",
            "altitude",
            "cloud_height",
            "backscatter",
        ]:

            value = values.get(name)

            if value is not None:

                if value.ndim != 1:
                    raise RuntimeError(
                        f"HSRL variable {name} is not 1-D "
                        f"after normalisation: {value.shape}"
                    )

                if value.size > 1:

                    if observation_length is None:
                        observation_length = value.size

                    elif value.size != observation_length:
                        raise RuntimeError(
                            f"Conflicting HSRL observation sizes: "
                            f"{name}={value.size}, "
                            f"expected={observation_length}."
                        )

        if observation_length is None:
            raise RuntimeError(
                "Unable to determine HSRL observation length."
            )

        print(
            "HSRL observation length:",
            observation_length,
        )

        coordinates = {
            "observation": np.arange(
                observation_length
            )
        }

        data_vars = {}

        if latitude is not None:
            data_vars["latitude"] = (
                ("observation",),
                latitude,
            )

        if longitude is not None:
            data_vars["longitude"] = (
                ("observation",),
                longitude,
            )

        if time is not None:
            data_vars["time"] = (
                ("observation",),
                time,
            )

        if altitude is not None:
            data_vars["altitude"] = (
                ("observation",),
                altitude,
            )

        if cloud_height is not None:
            data_vars["cloud_height"] = (
                ("observation",),
                cloud_height,
            )

        if backscatter is not None:
            data_vars["backscatter"] = (
                ("observation",),
                backscatter,
            )

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coordinates,
            attrs={
                "source": str(
                    self.path
                ),
                "instrument": "NASA HSRL-2",
            },
        )

        return dataset
