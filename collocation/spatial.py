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
    def _radians(
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

        sat_lat = np.asarray(
            satellite[self.satellite_lat].values,
            dtype=float,
        )

        sat_lon = np.asarray(
            satellite[self.satellite_lon].values,
            dtype=float,
        )

        air_lat = np.asarray(
            airborne[self.airborne_lat].values,
            dtype=float,
        )

        air_lon = np.asarray(
            airborne[self.airborne_lon].values,
            dtype=float,
        )

        sat_valid = (
            np.isfinite(sat_lat)
            & np.isfinite(sat_lon)
        )

        air_valid = (
            np.isfinite(air_lat)
            & np.isfinite(air_lon)
        )

        if not np.any(sat_valid):

            raise ValueError(
                "Satellite dataset contains no valid "
                "coordinates."
            )

        if not np.any(air_valid):

            raise ValueError(
                "Airborne dataset contains no valid "
                "coordinates."
            )

        sat_indices = np.flatnonzero(
            sat_valid
        )

        tree = BallTree(
            self._radians(
                sat_lat[sat_valid],
                sat_lon[sat_valid],
            ),
            metric="haversine",
        )

        air_indices = np.flatnonzero(
            air_valid
        )

        distances, indices = tree.query(
            self._radians(
                air_lat[air_valid],
                air_lon[air_valid],
            ),
            k=1,
        )

        distances = (
            distances[:, 0]
            * self.EARTH_RADIUS
        )

        indices = indices[:, 0]

        matched_satellite_indices = (
            sat_indices[indices]
        )

        keep = np.ones(
            len(air_indices),
            dtype=bool,
        )

        if self.max_distance is not None:

            keep = (
                distances
                <= self.max_distance
            )

        air_indices = air_indices[keep]

        matched_satellite_indices = (
            matched_satellite_indices[keep]
        )

        distances = distances[keep]

        result = airborne.isel(
            observation=air_indices
        ).copy()

        satellite_match = satellite.isel(
            observation=matched_satellite_indices
        ).copy()

        satellite_match = (
            satellite_match.rename(
                {
                    name: f"satellite_{name}"
                    for name in satellite_match.data_vars
                }
            )
        )

        satellite_match = (
            satellite_match.drop_vars(
                "observation",
                errors="ignore",
            )
        )

        result = xr.merge(
            [
                result,
                satellite_match,
            ],
            compat="override",
        )

        result[
            "spatial_distance_m"
        ] = (
            "observation",
            distances,
        )

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
