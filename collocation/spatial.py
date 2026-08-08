from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree


class SpatialCollocator:
    """Spatial collocation of satellite and airborne observations."""

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
                [
                    latitude,
                    longitude,
                ]
            )
        )

    @staticmethod
    def _valid(
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> np.ndarray:

        return (
            np.isfinite(latitude)
            & np.isfinite(longitude)
        )

    def collocate(
        self,
        satellite: xr.Dataset,
        airborne: xr.Dataset,
    ) -> xr.Dataset:

        satellite_lat = np.asarray(
            satellite[
                self.satellite_lat
            ].values,
            dtype=float,
        )

        satellite_lon = np.asarray(
            satellite[
                self.satellite_lon
            ].values,
            dtype=float,
        )

        airborne_lat = np.asarray(
            airborne[
                self.airborne_lat
            ].values,
            dtype=float,
        )

        airborne_lon = np.asarray(
            airborne[
                self.airborne_lon
            ].values,
            dtype=float,
        )

        satellite_valid = self._valid(
            satellite_lat,
            satellite_lon,
        )

        airborne_valid = self._valid(
            airborne_lat,
            airborne_lon,
        )

        if not np.any(satellite_valid):

            raise ValueError(
                "Satellite dataset contains no valid "
                "coordinates."
            )

        if not np.any(airborne_valid):

            raise ValueError(
                "Airborne dataset contains no valid "
                "coordinates."
            )

        satellite_indices = np.flatnonzero(
            satellite_valid
        )

        airborne_indices = np.flatnonzero(
            airborne_valid
        )

        tree = BallTree(
            self._radians(
                satellite_lat[
                    satellite_valid
                ],
                satellite_lon[
                    satellite_valid
                ],
            ),
            metric="haversine",
        )

        distances, neighbours = tree.query(
            self._radians(
                airborne_lat[
                    airborne_valid
                ],
                airborne_lon[
                    airborne_valid
                ],
            ),
            k=1,
        )

        distances = (
            distances[:, 0]
            * self.EARTH_RADIUS
        )

        neighbours = neighbours[:, 0]

        matched_satellite_indices = (
            satellite_indices[
                neighbours
            ]
        )

        keep = np.ones(
            distances.shape,
            dtype=bool,
        )

        if self.max_distance is not None:

            keep &= (
                distances
                <= self.max_distance
            )

        airborne_indices = (
            airborne_indices[keep]
        )

        matched_satellite_indices = (
            matched_satellite_indices[keep]
        )

        distances = distances[keep]

        airborne_result = airborne.isel(
            observation=airborne_indices
        ).copy()

        satellite_variables = {}

        for name, variable in satellite.data_vars.items():

            values = np.asarray(
                variable.values
            )

            if (
                values.ndim == 1
                and values.shape[0]
                == satellite_lat.shape[0]
            ):

                satellite_variables[
                    f"satellite_{name}"
                ] = (
                    "observation",
                    values[
                        matched_satellite_indices
                    ],
                )

        satellite_result = xr.Dataset(
            data_vars=satellite_variables,
            coords={
                "observation": (
                    airborne_result[
                        "observation"
                    ].values
                )
            },
        )

        result = xr.merge(
            [
                airborne_result,
                satellite_result,
            ],
            compat="override",
        )

        result[
            "spatial_distance_m"
        ] = (
            "observation",
            distances,
        )

        result.attrs[
            "spatial_collocation"
        ] = True

        result.attrs[
            "spatial_max_distance_m"
        ] = self.max_distance

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
