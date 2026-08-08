from __future__ import annotations

from pathlib import Path

import h5py
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
            "gps_altitude",
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
        super().__init__(path)

        self._mapping: dict[str, str | None] | None = None
        self.coordinates: dict[str, object] = {}
        self.variables: dict[str, object] = {}
        self.observation_length: int | None = None
        self.name = self.path.stem

    def variable_map(self) -> dict[str, str | None]:

        if self._mapping is not None:
            return dict(self._mapping)

        file = self.open()

        datasets = set(self.datasets)

        mapping: dict[str, str | None] = {}

        for name, aliases in self.VARIABLE_ALIASES.items():

            found = None

            for alias in aliases:

                if alias.startswith("/"):
                    candidates = [alias]
                else:
                    candidates = [
                        f"/Nav_Data/{alias}",
                        f"/DataProducts/{alias}",
                        f"/{alias}",
                    ]

                for candidate in candidates:

                    if candidate in datasets:
                        found = candidate
                        break

                if found is not None:
                    break

            mapping[name] = found

        self._mapping = mapping

        return dict(mapping)

    def available_variables(self) -> list[str]:

        return [
            name
            for name, path in self.variable_map().items()
            if path is not None
        ]

    def _find_dataset(self, path: str) -> h5py.Dataset:

        file = self.open()

        if path not in file:
            raise KeyError(
                f"HSRL dataset not found: {path}"
            )

        obj = file[path]

        if not isinstance(obj, h5py.Dataset):
            raise TypeError(
                f"HSRL path is not a dataset: {path}"
            )

        return obj

    def _read_variable(
        self,
        path: str,
    ) -> tuple[tuple[str, ...], np.ndarray]:

        dataset = self._find_dataset(path)

        values = np.asarray(
            dataset[...]
        )

        values = np.squeeze(values)

        if values.ndim == 0:
            dims = ()

        elif values.ndim == 1:

            dims = (
                "observation",
            )

            if values.size == 501 and not values.size == self.observation_length:
                dims = (
                    "level",
                )

        elif values.ndim == 2:

            if values.shape[0] == 1:
                values = values.reshape(-1)
                dims = (
                    "level",
                )

            elif (
                self.observation_length is not None
                and values.shape[0] == self.observation_length
            ):
                dims = (
                    "observation",
                    "level",
                )

            elif (
                self.observation_length is not None
                and values.shape[1] == self.observation_length
            ):
                values = values.T
                dims = (
                    "observation",
                    "level",
                )

            else:
                dims = (
                    "observation",
                    "level",
                )

        else:
            dims = tuple(
                f"dim_{i}"
                for i in range(values.ndim)
            )

        return dims, values

    def _read_first_existing(
        self,
        candidates: list[str],
    ) -> tuple[tuple[str, ...], np.ndarray] | None:

        datasets = set(self.datasets)

        for path in candidates:

            if path in datasets:
                return self._read_variable(path)

        return None

    def _read_vertical_coordinate(
        self,
    ) -> tuple[tuple[str, ...], np.ndarray] | None:

        candidates = [
            "/DataProducts/Altitude",
            "/DataProducts/altitude",
            "/DataProducts/z",
            "/z",
            "/Altitude",
        ]

        result = self._read_first_existing(
            candidates
        )

        if result is None:
            return None

        dims, values = result

        values = np.asarray(values).reshape(-1)

        return (
            ("level",),
            values,
        )

    def _valid_observation_length(
        self,
        observation_length: int | None,
    ) -> bool:

        if observation_length is None:
            return False

        if observation_length <= 1:
            return False

        if self.observation_length is None:
            self.observation_length = int(
                observation_length
            )
            return True

        return (
            int(observation_length)
            == int(self.observation_length)
        )

    def _normalise_observation(
        self,
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(values)

        if values.ndim == 0:
            return values

        if values.ndim == 1:
            return values

        if values.ndim == 2:

            if values.shape[1] == 1:
                return values[:, 0]

            if values.shape[0] == 1:
                return values[0]

        return values

    def read_dataset(
        self,
    ) -> xr.Dataset:

        mapping = self.variable_map()

        print("\nHSRL variable mapping:")

        for name, path in mapping.items():
            print(
                f"    {name}: {path}"
            )

        coordinates: dict[str, object] = {}
        variables: dict[str, object] = {}

        vertical = self._read_vertical_coordinate()

        if vertical is not None:

            z_dims, z_values = vertical

            coordinates["z"] = (
                z_dims,
                z_values,
            )

            coordinates["level"] = (
                "level",
                np.arange(
                    z_values.size
                ),
            )

        observation_length = None

        for name in (
            "latitude",
            "longitude",
            "time",
            "altitude",
        ):

            path = mapping.get(name)

            if path is None:
                continue

            result = self._read_variable(path)

            dims, values = result

            values = self._normalise_observation(
                values
            )

            if values.ndim != 1:
                continue

            if observation_length is None:
                observation_length = values.size

            elif values.size != observation_length:
                continue

        if observation_length is None:
            raise RuntimeError(
                "Unable to determine HSRL observation length."
            )

        self.observation_length = int(
            observation_length
        )

        print(
            "HSRL observation length:",
            self.observation_length,
        )

        for name, path in mapping.items():

            if path is None:
                continue

            try:

                dims, values = self._read_variable(
                    path
                )

            except Exception:
                continue

            values = np.asarray(values)

            if name in (
                "latitude",
                "longitude",
                "time",
                "altitude",
                "cloud_height",
            ):

                values = self._normalise_observation(
                    values
                )

                if values.ndim != 1:
                    continue

                if values.size != self.observation_length:
                    continue

                coordinates[name] = (
                    "observation",
                    values,
                )

                continue

            if name == "backscatter":

                if values.ndim == 1:

                    if values.size == self.observation_length:

                        variables[name] = (
                            (
                                "observation",
                            ),
                            values,
                        )

                elif values.ndim == 2:

                    if (
                        values.shape[0]
                        == self.observation_length
                    ):

                        if "level" in coordinates:

                            level_size = coordinates[
                                "level"
                            ][1].size

                            if values.shape[1] == level_size:

                                variables[name] = (
                                    (
                                        "observation",
                                        "level",
                                    ),
                                    values,
                                )

                            else:

                                variables[name] = (
                                    (
                                        "observation",
                                        "level",
                                    ),
                                    values,
                                )

                        else:

                            variables[name] = (
                                (
                                    "observation",
                                    "level",
                                ),
                                values,
                            )

                continue

            if values.ndim == 0:

                variables[name] = (
                    (),
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

        if "altitude" not in coordinates:
            raise RuntimeError(
                "HSRL altitude could not be loaded."
            )

        if "backscatter" in variables:

            backscatter = variables[
                "backscatter"
            ][1]

            if (
                backscatter.ndim == 2
                and "level" not in coordinates
            ):

                coordinates["level"] = (
                    "level",
                    np.arange(
                        backscatter.shape[1]
                    ),
                )

        if "z" in coordinates:

            z_values = coordinates["z"][1]

            if (
                "level" in coordinates
                and z_values.size
                != coordinates["level"][1].size
            ):

                coordinates["level"] = (
                    "level",
                    np.arange(
                        z_values.size
                    ),
                )

        dataset = xr.Dataset(
            data_vars=variables,
            coords=coordinates,
        )

        dataset.attrs.update(
            {
                "source": "NASA ACTIVATE HSRL-2",
                "instrument": "HSRL-2",
                "file": self.path.name,
            }
        )

        if "time" in dataset:

            time_values = dataset[
                "time"
            ].values

            try:

                if np.issubdtype(
                    time_values.dtype,
                    np.number,
                ):

                    origin = np.datetime64(
                        "2022-01-11T00:00:00"
                    )

                    datetime_values = (
                        origin
                        + time_values.astype(
                            "timedelta64[s]"
                        )
                    )

                    dataset = dataset.assign_coords(
                        time=(
                            "observation",
                            datetime_values,
                        )
                    )

            except Exception:
                pass

        self.coordinates = dict(
            dataset.coords
        )

        self.variables = dict(
            dataset.data_vars
        )

        return dataset
