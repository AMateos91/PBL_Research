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

    PROFILE_VARIABLES = {
        "backscatter": "DataProducts/AB_prfl",
        "aerosol_backscatter": "DataProducts/IAB",
        "lid": "DataProducts/LID",
        "multiple_scattering_fraction": "DataProducts/MultScatFrac",
        "optical_depth": "DataProducts/OD_prfl",
        "cloud_extent_profile": "DataProducts/cloud_ext_prfl",
        "temperature_profile": "State/temperature_prfl",
    }

    OBSERVATION_VARIABLES = {
        "reflectance": "DataProducts/Reflectance",
        "reflectance_average": "DataProducts/ReflectanceAvg",
        "cloud_extent_average": "DataProducts/cloud_ext_average",
        "transmittance": "DataProducts/TransP",
        "surface_transmittance": "DataProducts/TransSurface",
        "wind_speed_cm": "DataProducts/WindSpeedDerivedCM",
        "wind_speed_hu": "DataProducts/WindSpeedDerivedHU",
        "normalized_backscatter": "DataProducts/bscNorm",
        "selection_index": "DataProducts/selection_index",
        "temperature": "State/temperature",
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

        with h5py.File(
            self.path,
            "r",
        ) as file:

            for name, path in self.VARIABLE_MAP.items():

                if path in file:
                    available.append(name)

        return available

    def read(
        self,
        path: str,
    ) -> np.ndarray:

        with h5py.File(
            self.path,
            "r",
        ) as file:

            if path not in file:
                raise KeyError(
                    f"Dataset not found: {path}"
                )

            return np.asarray(
                file[path]
            )

    @staticmethod
    def _as_1d(
        values: np.ndarray,
    ) -> np.ndarray:

        return np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

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

    def read_dataset(
        self,
    ) -> xr.Dataset:

        cloud_height = self._as_1d(
            self.read(
                "DataProducts/cloud_height"
            )
        )

        n_observations = len(
            cloud_height
        )

        latitude = self._as_1d(
            self.read("lat")
        )

        longitude = self._as_1d(
            self.read("lon")
        )

        altitude = self._as_1d(
            self.read("alt")
        )

        raw_time = self._as_1d(
            self.read("time")
        )

        time = self._decode_time(
            raw_time
        )

        z = self._as_1d(
            self.read("z")
        )

        if len(latitude) != n_observations:
            raise ValueError(
                "lat does not match cloud_height: "
                f"{len(latitude)} != {n_observations}"
            )

        if len(longitude) != n_observations:
            raise ValueError(
                "lon does not match cloud_height: "
                f"{len(longitude)} != {n_observations}"
            )

        if len(altitude) != n_observations:
            raise ValueError(
                "alt does not match cloud_height: "
                f"{len(altitude)} != {n_observations}"
            )

        if len(time) != n_observations:
            raise ValueError(
                "time does not match cloud_height: "
                f"{len(time)} != {n_observations}"
            )

        backscatter = np.asarray(
            self.read(
                "DataProducts/AB_prfl"
            ),
            dtype=np.float64,
        )

        if backscatter.shape != (
            n_observations,
            len(z),
        ):

            raise ValueError(
                "AB_prfl has unexpected shape: "
                f"{backscatter.shape}; expected "
                f"({n_observations}, {len(z)})"
            )

        data_vars = {
            "cloud_height": (
                "observation",
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

        with h5py.File(
            self.path,
            "r",
        ) as file:

            for name, path in self.PROFILE_VARIABLES.items():

                if name == "backscatter":
                    continue

                if path not in file:
                    continue

                values = np.asarray(
                    file[path],
                    dtype=np.float64,
                )

                if values.ndim != 2:
                    continue

                if values.shape != (
                    n_observations,
                    len(z),
                ):
                    continue

                data_vars[name] = (
                    (
                        "observation",
                        "level",
                    ),
                    values,
                )

            for name, path in self.OBSERVATION_VARIABLES.items():

                if path not in file:
                    continue

                values = np.asarray(
                    file[path],
                    dtype=np.float64,
                ).reshape(-1)

                if len(values) != n_observations:
                    continue

                data_vars[name] = (
                    "observation",
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
            },
        )

        return dataset
