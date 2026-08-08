from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr


class HSRLReader:

    VARIABLE_MAP = {
        "latitude": "lat",
        "longitude": "lon",
        "altitude": "alt",
        "time": "time",
        "cloud_height": "DataProducts/cloud_height",
        "backscatter": "DataProducts/AB_prfl",
        "z": "z",
        "aerosol_backscatter": "DataProducts/IAB",
        "lid": "DataProducts/LID",
        "multiple_scattering_fraction": "DataProducts/MultScatFrac",
        "optical_depth": "DataProducts/OD_prfl",
        "reflectance": "DataProducts/Reflectance",
        "reflectance_average": "DataProducts/ReflectanceAvg",
        "cloud_extent_average": "DataProducts/cloud_ext_average",
        "cloud_extent_profile": "DataProducts/cloud_ext_prfl",
        "transmittance": "DataProducts/TransP",
        "surface_transmittance": "DataProducts/TransSurface",
        "wind_speed_cm": "DataProducts/WindSpeedDerivedCM",
        "wind_speed_hu": "DataProducts/WindSpeedDerivedHU",
        "normalized_backscatter": "DataProducts/bscNorm",
        "selection_index": "DataProducts/selection_index",
        "temperature": "State/temperature",
        "temperature_profile": "State/temperature_prfl",
    }

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"HSRL file not found: {self.path}"
            )

    def variable_map(self) -> dict[str, str]:

        return self.VARIABLE_MAP.copy()

    def available_variables(self) -> list[str]:

        available = []

        with h5py.File(self.path, "r") as file:

            for name, path in self.VARIABLE_MAP.items():

                if path in file:
                    available.append(name)

        return available

    def read(
        self,
        path: str,
    ) -> np.ndarray:

        with h5py.File(self.path, "r") as file:

            if path not in file:
                raise KeyError(
                    f"Dataset not found: {path}"
                )

            return np.asarray(
                file[path]
            )

    def _read_attributes(
        self,
        path: str,
    ) -> dict:

        with h5py.File(self.path, "r") as file:

            if path not in file:
                return {}

            return {
                key: value
                for key, value in file[path].attrs.items()
            }

    @staticmethod
    def _decode_time(
        values: np.ndarray,
    ) -> np.ndarray:

        origin = pd.Timestamp(
            "2022-01-11T00:00:00Z"
        )

        return (
            origin
            + pd.to_timedelta(
                values,
                unit="s",
            )
        ).to_numpy()

    @staticmethod
    def _clean_1d(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        return values.reshape(-1)

    @staticmethod
    def _clean_profile(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                f"Expected 2-D profile, got {values.shape}"
            )

        return values

    def read_dataset(
        self,
    ) -> xr.Dataset:

        latitude = self._clean_1d(
            self.read("lat")
        )

        longitude = self._clean_1d(
            self.read("lon")
        )

        altitude = self._clean_1d(
            self.read("alt")
        )

        raw_time = self._clean_1d(
            self.read("time")
        )

        time = self._decode_time(
            raw_time
        )

        cloud_height = self._clean_1d(
            self.read(
                "DataProducts/cloud_height"
            )
        )

        z = self._clean_1d(
            self.read("z")
        )

        backscatter = self._clean_profile(
            self.read(
                "DataProducts/AB_prfl"
            )
        )

        n_observations = len(
            cloud_height
        )

        if len(latitude) != n_observations:
            raise ValueError(
                "Latitude length does not match "
                "cloud_height."
            )

        if len(longitude) != n_observations:
            raise ValueError(
                "Longitude length does not match "
                "cloud_height."
            )

        if len(altitude) != n_observations:
            raise ValueError(
                "Altitude length does not match "
                "cloud_height."
            )

        if len(time) != n_observations:
            raise ValueError(
                "Time length does not match "
                "cloud_height."
            )

        if backscatter.shape[0] != n_observations:
            raise ValueError(
                "Backscatter observation dimension "
                "does not match cloud_height."
            )

        if backscatter.shape[1] != len(z):
            raise ValueError(
                "Backscatter level dimension "
                "does not match z."
            )

        data_vars = {
            "cloud_height": (
                ("observation",),
                cloud_height,
            ),
            "backscatter": (
                (
                    "observation",
                    "level",
                ),
                backscatter,
            ),
        }

        optional_profiles = {
            "aerosol_backscatter":
                "DataProducts/IAB",
            "lid":
                "DataProducts/LID",
            "multiple_scattering_fraction":
                "DataProducts/MultScatFrac",
            "optical_depth":
                "DataProducts/OD_prfl",
            "cloud_extent_profile":
                "DataProducts/cloud_ext_prfl",
            "temperature_profile":
                "State/temperature_prfl",
        }

        optional_1d = {
            "reflectance":
                "DataProducts/Reflectance",
            "reflectance_average":
                "DataProducts/ReflectanceAvg",
            "cloud_extent_average":
                "DataProducts/cloud_ext_average",
            "transmittance":
                "DataProducts/TransP",
            "surface_transmittance":
                "DataProducts/TransSurface",
            "wind_speed_cm":
                "DataProducts/WindSpeedDerivedCM",
            "wind_speed_hu":
                "DataProducts/WindSpeedDerivedHU",
            "normalized_backscatter":
                "DataProducts/bscNorm",
            "selection_index":
                "DataProducts/selection_index",
            "temperature":
                "State/temperature",
        }

        with h5py.File(
            self.path,
            "r",
        ) as file:

            for name, path in optional_profiles.items():

                if path not in file:
                    continue

                values = np.asarray(
                    file[path],
                    dtype=np.float64,
                )

                if (
                    values.ndim == 2
                    and values.shape
                    == (
                        n_observations,
                        len(z),
                    )
                ):

                    data_vars[name] = (
                        (
                            "observation",
                            "level",
                        ),
                        values,
                    )

            for name, path in optional_1d.items():

                if path not in file:
                    continue

                values = np.asarray(
                    file[path],
                    dtype=np.float64,
                ).reshape(-1)

                if len(values) == n_observations:

                    data_vars[name] = (
                        (
                            "observation",
                        ),
                        values,
                    )

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "observation": np.arange(
                    n_observations
                ),
                "level": np.arange(
                    len(z)
                ),
                "latitude": (
                    "observation",
                    latitude,
                ),
                "longitude": (
                    "observation",
                    longitude,
                ),
                "altitude": (
                    "observation",
                    altitude,
                ),
                "time": (
                    "observation",
                    time,
                ),
                "z": (
                    "level",
                    z,
                ),
            },
            attrs={
                "source": str(
                    self.path
                ),
                "instrument": "NASA HSRL-2",
                "time_reference":
                    "2022-01-11T00:00:00Z",
            },
        )

        return dataset
