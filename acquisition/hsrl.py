from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .hdf import HDFReader


class HSRLReader(HDFReader):

    PROFILE_VARIABLES = {
        "AB_prfl": "/DataProducts/AB_prfl",
        "IAB_prfl": "/DataProducts/IAB_prfl",
        "LID_prfl": "/DataProducts/LID_prfl",
        "MultScatFrac_prfl": "/DataProducts/MultScatFrac_prfl",
        "OD_prfl": "/DataProducts/OD_prfl",
        "TransP_prfl": "/DataProducts/TransP_prfl",
        "cloud_ext_prfl": "/DataProducts/cloud_ext_prfl",
        "temperature_prfl": "/DataProducts/temperature_prfl",
    }

    NAVIGATION_VARIABLES = {
        "latitude": "/Nav_Data/gps_lat",
        "longitude": "/Nav_Data/gps_lon",
        "time": "/Nav_Data/gps_time",
        "altitude": "/Nav_Data/gps_alt",
    }

    PRODUCT_VARIABLES = {
        "cloud_height": "/DataProducts/cloud_height",
    }

    VERTICAL_COORDINATE = "/DataProducts/Altitude"

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        super().__init__(path)

        self._mapping = None
        self.observation_length = None
        self.level_length = None

    def variable_map(self) -> dict[str, str]:

        mapping = {}

        mapping.update(
            self.NAVIGATION_VARIABLES
        )

        mapping.update(
            self.PRODUCT_VARIABLES
        )

        mapping.update(
            self.PROFILE_VARIABLES
        )

        mapping["z"] = self.VERTICAL_COORDINATE

        self._mapping = mapping

        return dict(mapping)

    def available_variables(self) -> list[str]:

        datasets = set(self.datasets)

        return [
            name
            for name, path in self.variable_map().items()
            if path in datasets
        ]

    def _read_array(
        self,
        path: str,
    ) -> np.ndarray:

        return np.asarray(
            self.read(path)
        )

    @staticmethod
    def _squeeze_navigation(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(values)

        if values.ndim == 1:
            return values

        if values.ndim == 2:

            if values.shape[1] == 1:
                return values[:, 0]

            if values.shape[0] == 1:
                return values[0]

        return np.squeeze(values)

    def _read_navigation(
        self,
        name: str,
        path: str,
    ) -> np.ndarray:

        values = self._read_array(
            path
        )

        values = self._squeeze_navigation(
            values
        )

        if values.ndim != 1:
            raise RuntimeError(
                f"HSRL {name} is not one-dimensional: "
                f"shape={values.shape}"
            )

        return values

    def _read_profile(
        self,
        name: str,
        path: str,
    ) -> np.ndarray:

        values = self._read_array(
            path
        )

        values = np.asarray(
            values
        )

        if values.ndim != 2:

            raise RuntimeError(
                f"HSRL profile {name} is not two-dimensional: "
                f"shape={values.shape}"
            )

        if self.observation_length is not None:

            if values.shape[0] != self.observation_length:

                if values.shape[1] == self.observation_length:

                    values = values.T

                else:

                    raise RuntimeError(
                        f"HSRL profile {name} has incompatible "
                        f"shape={values.shape}"
                    )

        if self.level_length is not None:

            if values.shape[1] != self.level_length:

                if values.shape[0] == self.level_length:

                    values = values.T

                else:

                    raise RuntimeError(
                        f"HSRL profile {name} has incompatible "
                        f"level dimension: "
                        f"shape={values.shape}"
                    )

        return values

    def _read_altitude(
        self,
    ) -> np.ndarray:

        values = self._read_array(
            self.VERTICAL_COORDINATE
        )

        values = np.asarray(
            values
        )

        values = np.squeeze(
            values
        )

        if values.ndim != 1:

            raise RuntimeError(
                "HSRL Altitude could not be converted "
                "to a one-dimensional vertical coordinate."
            )

        return values

    def read_dataset(
        self,
    ) -> xr.Dataset:

        mapping = self.variable_map()

        print("\nHSRL variable mapping:")

        for name, path in mapping.items():

            print(
                f"{name}: {path}"
            )

        altitude = self._read_altitude()

        self.level_length = int(
            altitude.size
        )

        latitude = self._read_navigation(
            "latitude",
            self.NAVIGATION_VARIABLES[
                "latitude"
            ],
        )

        longitude = self._read_navigation(
            "longitude",
            self.NAVIGATION_VARIABLES[
                "longitude"
            ],
        )

        time = self._read_navigation(
            "time",
            self.NAVIGATION_VARIABLES[
                "time"
            ],
        )

        aircraft_altitude = self._read_navigation(
            "altitude",
            self.NAVIGATION_VARIABLES[
                "altitude"
            ],
        )

        self.observation_length = int(
            latitude.size
        )

        if longitude.size != self.observation_length:

            raise RuntimeError(
                "HSRL longitude has a different "
                "observation length."
            )

        if time.size != self.observation_length:

            raise RuntimeError(
                "HSRL time has a different "
                "observation length."
            )

        if (
            aircraft_altitude.size
            != self.observation_length
        ):

            raise RuntimeError(
                "HSRL altitude has a different "
                "observation length."
            )

        cloud_height = self._read_navigation(
            "cloud_height",
            self.PRODUCT_VARIABLES[
                "cloud_height"
            ],
        )

        if cloud_height.size != self.observation_length:

            raise RuntimeError(
                "HSRL cloud_height has a different "
                "observation length."
            )

        variables = {}

        for name, path in self.PROFILE_VARIABLES.items():

            try:

                values = self._read_profile(
                    name,
                    path,
                )

            except Exception as exc:

                print(
                    f"Skipping {name}: {exc}"
                )

                continue

            variables[name] = (
                (
                    "observation",
                    "level",
                ),
                values,
            )

        coordinates = {

            "latitude": (
                "observation",
                latitude,
            ),

            "longitude": (
                "observation",
                longitude,
            ),

            "time": (
                "observation",
                time,
            ),

            "altitude": (
                "observation",
                aircraft_altitude,
            ),

            "cloud_height": (
                "observation",
                cloud_height,
            ),

            "z": (
                "level",
                altitude,
            ),

        }

        dataset = xr.Dataset(
            data_vars=variables,
            coords=coordinates,
        )

        dataset.attrs.update(
            {
                "source": str(
                    self.path
                ),
                "instrument": "NASA HSRL-2",
                "observation_length": (
                    self.observation_length
                ),
                "level_length": (
                    self.level_length
                ),
            }
        )

        print(
            "\nHSRL dataset successfully built."
        )

        print(
            "Observations:",
            self.observation_length,
        )

        print(
            "Vertical levels:",
            self.level_length,
        )

        print(
            "Profiles:",
            list(
                variables.keys()
            ),
        )

        return dataset

    def __repr__(
        self,
    ) -> str:

        return (
            "HSRLReader("
            f"path={self.path!r}"
            ")"
        )
