from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from .base import HDFReader


class HSRLReader(HDFReader):

    VARIABLE_ALIASES = {
        "latitude": [
            "gps_lat",
            "lat",
            "latitude",
            "Latitude",
            "Lat",
            "/Nav_Data/gps_lat",
            "/lat",
        ],
        "longitude": [
            "gps_lon",
            "lon",
            "longitude",
            "Longitude",
            "Lon",
            "/Nav_Data/gps_lon",
            "/lon",
        ],
        "time": [
            "gps_time",
            "time",
            "Time",
            "UTC_Time",
            "/Nav_Data/gps_time",
            "/time",
        ],
        "altitude": [
            "gps_alt",
            "gpd_alt",
            "alt",
            "altitude",
            "Altitude",
            "GPSAltitude",
            "AircraftAltitude",
            "/Nav_Data/gps_alt",
            "/alt",
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

    DIRECT_PATHS = {
        "latitude": "/Nav_Data/gps_lat",
        "longitude": "/Nav_Data/gps_lon",
        "time": "/Nav_Data/gps_time",
        "altitude": "/Nav_Data/gps_alt",
        "cloud_height": "/DataProducts/cloud_height",
        "backscatter": "/DataProducts/bscNorm",
        "ab_prfl": "/DataProducts/AB_prfl",
        "z": "/z",
    }

    def __init__(self, source):
        super().__init__(source)
        self.source = Path(source)

    def variable_map(self):
        return dict(self.DIRECT_PATHS)

    def _read_variable(self, path):
        try:
            with h5py.File(self.source, "r") as f:
                if path not in f:
                    return None

                value = np.asarray(f[path])

            if value.ndim == 2 and value.shape[1] == 1:
                value = value[:, 0]

            elif value.ndim == 2 and value.shape[0] == 1:
                value = value[0, :]

            return value

        except Exception:
            return None

    def _valid_observation_length(self, observation_length):
        if observation_length is None:
            return False

        try:
            observation_length = int(observation_length)
        except (TypeError, ValueError):
            return False

        return observation_length > 1

    def _read_direct(self, path):
        value = self._read_variable(path)

        if value is None:
            return None

        value = np.asarray(value)

        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]

        elif value.ndim == 2 and value.shape[0] == 1:
            value = value[0, :]

        return value

    def _determine_observation_length(self):
        candidates = [
            self.DIRECT_PATHS["latitude"],
            self.DIRECT_PATHS["longitude"],
            self.DIRECT_PATHS["time"],
            self.DIRECT_PATHS["altitude"],
            self.DIRECT_PATHS["cloud_height"],
            self.DIRECT_PATHS["backscatter"],
        ]

        for path in candidates:
            values = self._read_direct(path)

            if values is None:
                continue

            values = np.asarray(values)

            if values.ndim == 1 and values.size > 1:
                return int(values.size)

        return None

    def read_dataset(self):
        mapping = self.variable_map()

        print("\nHSRL variable mapping:")

        for name, path in mapping.items():
            print(f"    {name}: {path}")

        observation_length = self._determine_observation_length()

        if not self._valid_observation_length(observation_length):
            raise RuntimeError(
                "Unable to determine HSRL observation length."
            )

        print(
            f"HSRL observation length: {observation_length}"
        )

        latitude = self._read_direct(
            mapping["latitude"]
        )

        longitude = self._read_direct(
            mapping["longitude"]
        )

        time = self._read_direct(
            mapping["time"]
        )

        altitude = self._read_direct(
            mapping["altitude"]
        )

        cloud_height = self._read_direct(
            mapping["cloud_height"]
        )

        backscatter = self._read_direct(
            mapping["backscatter"]
        )

        if latitude is None:
            raise RuntimeError(
                "HSRL latitude could not be loaded."
            )

        if longitude is None:
            raise RuntimeError(
                "HSRL longitude could not be loaded."
            )

        if time is None:
            raise RuntimeError(
                "HSRL time could not be loaded."
            )

        if altitude is None:
            raise RuntimeError(
                "HSRL altitude could not be loaded."
            )

        if cloud_height is None:
            raise RuntimeError(
                "HSRL cloud height could not be loaded."
            )

        if backscatter is None:
            raise RuntimeError(
                "HSRL backscatter could not be loaded."
            )

        latitude = np.asarray(latitude).reshape(-1)
        longitude = np.asarray(longitude).reshape(-1)
        time = np.asarray(time).reshape(-1)
        altitude = np.asarray(altitude).reshape(-1)
        cloud_height = np.asarray(cloud_height).reshape(-1)
        backscatter = np.asarray(backscatter).reshape(-1)

        if len(latitude) != observation_length:
            raise RuntimeError(
                "HSRL latitude length does not match "
                "the observation length."
            )

        if len(longitude) != observation_length:
            raise RuntimeError(
                "HSRL longitude length does not match "
                "the observation length."
            )

        if len(time) != observation_length:
            raise RuntimeError(
                "HSRL time length does not match "
                "the observation length."
            )

        if len(altitude) != observation_length:
            raise RuntimeError(
                "HSRL altitude length does not match "
                "the observation length."
            )

        if len(cloud_height) != observation_length:
            raise RuntimeError(
                "HSRL cloud height length does not match "
                "the observation length."
            )

        if len(backscatter) != observation_length:
            raise RuntimeError(
                "HSRL backscatter length does not match "
                "the observation length."
            )

        data_vars = {
            "altitude": (
                "observation",
                altitude,
            ),
            "cloud_height": (
                "observation",
                cloud_height,
            ),
            "backscatter": (
                "observation",
                backscatter,
            ),
        }

        ab_prfl = self._read_direct(
            mapping["ab_prfl"]
        )

        z = self._read_direct(
            mapping["z"]
        )

        if ab_prfl is not None and z is not None:

            ab_prfl = np.asarray(ab_prfl)

            z = np.asarray(z).reshape(-1)

            if (
                ab_prfl.ndim == 2
                and ab_prfl.shape[0] == observation_length
                and ab_prfl.shape[1] == len(z)
            ):
                data_vars["ab_prfl"] = (
                    ("observation", "level"),
                    ab_prfl,
                )

        time = np.asarray(time, dtype=np.float64)

        valid_time = np.isfinite(time)

        if np.any(valid_time):

            base_time = np.datetime64(
                "2022-01-11T00:00:00",
                "ns",
            )

            time_datetime = (
                base_time
                + (
                    time * 1_000_000_000
                ).astype("timedelta64[ns]")
            )

        else:

            time_datetime = np.full(
                observation_length,
                np.datetime64("NaT"),
                dtype="datetime64[ns]",
            )

        coords = {
            "observation": np.arange(
                observation_length,
                dtype=np.int64,
            ),
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
                time_datetime,
            ),
        }

        if z is not None:

            z = np.asarray(z).reshape(-1)

            if len(z) > 1:

                coords["level"] = np.arange(
                    len(z),
                    dtype=np.int64,
                )

                coords["z"] = (
                    "level",
                    z,
                )

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
        )

        dataset.attrs.update(
            {
                "source": str(self.source),
                "instrument": "ACTIVATE HSRL2",
                "observation_length": observation_length,
                "time_reference": (
                    "seconds since "
                    "2022-01-11T00:00:00Z"
                ),
            }
        )

        return dataset
