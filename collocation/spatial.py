from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree


class SpatialCollocator:

    EARTH_RADIUS = 6371008.8

    def __init__(
        self,
        satellite_lat: str = "latitude",
        satellite_lon: str = "longitude",
        airborne_lat: str = "latitude",
        airborne_lon: str = "longitude",
        max_distance: float | None = None,
    ) -> None:

        self.satellite_lat = satellite_lat
        self.satellite_lon = satellite_lon
        self.airborne_lat = airborne_lat
        self.airborne_lon = airborne_lon
        self.max_distance = max_distance

    @staticmethod
    def _coordinates(
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> np.ndarray:

        return np.deg2rad(
            np.column_stack(
                (
                    latitude,
                    longitude,
                )
            )
        )

    def collocate(
        self,
        satellite: xr.Dataset,
        airborne: xr.Dataset,
    ) -> xr.Dataset:

        satellite_lat = np.asarray(
            satellite[self.satellite_lat].values,
            dtype=float,
        )

        satellite_lon = np.asarray(
            satellite[self.satellite_lon].values,
            dtype=float,
        )

        airborne_lat = np.asarray(
            airborne[self.airborne_lat].values,
            dtype=float,
        )

        airborne_lon = np.asarray(
            airborne[self.airborne_lon].values,
            dtype=float,
        )

        satellite_valid = (
            np.isfinite(satellite_lat)
            & np.isfinite(satellite_lon)
        )

        airborne_valid = (
            np.isfinite(airborne_lat)
            & np.isfinite(airborne_lon)
        )

        satellite_indices = np.flatnonzero(
            satellite_valid
        )

        airborne_indices = np.flatnonzero(
            airborne_valid
        )

        if satellite_indices.size == 0:
            raise ValueError(
                "No valid satellite coordinates."
            )

        if airborne_indices.size == 0:
            raise ValueError(
                "No valid airborne coordinates."
            )

        tree = BallTree(
            self._coordinates(
                satellite_lat[satellite_valid],
                satellite_lon[satellite_valid],
            ),
            metric="haversine",
        )

        distances, neighbours = tree.query(
            self._coordinates(
                airborne_lat[airborne_valid],
                airborne_lon[airborne_valid],
            ),
            k=1,
        )

        distances = (
            distances[:, 0]
            * self.EARTH_RADIUS
        )

        neighbours = neighbours[:, 0]

        matched_satellite = (
            satellite_indices[neighbours]
        )

        if self.max_distance is not None:

            keep = (
                distances
                <= self.max_distance
            )

            airborne_indices = (
                airborne_indices[keep]
            )

            matched_satellite = (
                matched_satellite[keep]
            )

            distances = distances[keep]

        result = airborne.isel(
            observation=airborne_indices
        ).copy()

        result["spatial_distance_m"] = (
            "observation",
            distances,
        )

        for name, variable in satellite.data_vars.items():

            values = np.asarray(
                variable.values
            )

            if values.ndim != 1:
                continue

            if values.shape[0] != satellite_lat.shape[0]:
                continue

            selected = values[
                matched_satellite
            ]

            result[
                f"satellite_{name}"
            ] = (
                "observation",
                selected,
            )

        result.attrs[
            "spatial_collocation"
        ] = True

        return result

    def __call__(
        self,
        satellite: xr.Dataset,
        airborne: xr.Dataset,
    ) -> xr.Dataset:

        return self.collocate(
            satellite,
            airborne,
        )
