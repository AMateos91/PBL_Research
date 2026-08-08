from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree


AggregationMethod = Literal[
    "nearest",
    "mean",
    "distance_weighted",
]


class SpatialCollocator:
    """
    Spatial collocation between airborne and satellite
    observations using a BallTree and the Haversine metric.

    The airborne dataset may contain vertical profiles
    with dimensions (observation, level). The level
    dimension is preserved.
    """

    EARTH_RADIUS = 6371008.8

    def __init__(
        self,
        satellite_lat: str = "latitude",
        satellite_lon: str = "longitude",
        airborne_lat: str = "latitude",
        airborne_lon: str = "longitude",
        satellite_alt: str | None = None,
        airborne_alt: str | None = None,
        max_distance: float | None = None,
        max_altitude_difference: float | None = None,
        k: int = 1,
        aggregation: AggregationMethod = "nearest",
    ) -> None:

        self.satellite_lat = satellite_lat
        self.satellite_lon = satellite_lon
        self.airborne_lat = airborne_lat
        self.airborne_lon = airborne_lon

        self.satellite_alt = satellite_alt
        self.airborne_alt = airborne_alt

        self.max_distance = max_distance
        self.max_altitude_difference = (
            max_altitude_difference
        )

        self.k = max(1, int(k))
        self.aggregation = aggregation

        if self.aggregation not in {
            "nearest",
            "mean",
            "distance_weighted",
        }:
            raise ValueError(
                f"Unknown aggregation method: "
                f"{self.aggregation}"
            )

    @staticmethod
    def _validate_coordinates(
        dataset: xr.Dataset,
        latitude: str,
        longitude: str,
    ) -> str:

        if latitude not in dataset:
            raise ValueError(
                f"Missing latitude variable: {latitude}"
            )

        if longitude not in dataset:
            raise ValueError(
                f"Missing longitude variable: {longitude}"
            )

        lat = dataset[latitude]

        lon = dataset[longitude]

        if lat.ndim != 1:
            raise ValueError(
                f"{latitude} must be one-dimensional."
            )

        if lon.ndim != 1:
            raise ValueError(
                f"{longitude} must be one-dimensional."
            )

        if lat.dims != lon.dims:
            raise ValueError(
                "Latitude and longitude must use "
                "the same dimension."
            )

        return lat.dims[0]

    @staticmethod
    def _coordinates_to_radians(
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

    @staticmethod
    def _valid_coordinates(
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> np.ndarray:

        return (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & (latitude >= -90.0)
            & (latitude <= 90.0)
            & (longitude >= -180.0)
            & (longitude <= 180.0)
        )

    def _prepare_satellite(
        self,
        satellite: xr.Dataset,
    ) -> tuple[
        xr.Dataset,
        str,
        np.ndarray,
    ]:

        dimension = self._validate_coordinates(
            satellite,
            self.satellite_lat,
            self.satellite_lon,
        )

        latitude = np.asarray(
            satellite[
                self.satellite_lat
            ].values,
            dtype=float,
        )

        longitude = np.asarray(
            satellite[
                self.satellite_lon
            ].values,
            dtype=float,
        )

        valid = self._valid_coordinates(
            latitude,
            longitude,
        )

        if not np.any(valid):
            raise ValueError(
                "Satellite dataset contains no valid "
                "coordinates."
            )

        valid_indices = np.flatnonzero(valid)

        satellite_valid = satellite.isel(
            {
                dimension: valid_indices,
            }
        )

        return (
            satellite_valid,
            dimension,
            valid_indices,
        )

    def _query(
        self,
        tree: BallTree,
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        valid = self._valid_coordinates(
            latitude,
            longitude,
        )

        distances = np.full(
            (
                latitude.size,
                self.k,
            ),
            np.inf,
            dtype=float,
        )

        indices = np.full(
            (
                latitude.size,
                self.k,
            ),
            -1,
            dtype=int,
        )

        if np.any(valid):

            coordinates = (
                self._coordinates_to_radians(
                    latitude[valid],
                    longitude[valid],
                )
            )

            queried_distances, queried_indices = (
                tree.query(
                    coordinates,
                    k=self.k,
                )
            )

            queried_distances *= (
                self.EARTH_RADIUS
            )

            distances[valid] = (
                queried_distances
            )

            indices[valid] = (
                queried_indices
            )

        return distances, indices

    def _aggregate(
        self,
        satellite: xr.Dataset,
        indices: np.ndarray,
        distances: np.ndarray,
        dimension: str,
    ) -> xr.Dataset:

        if self.aggregation == "nearest":

            return satellite.isel(
                {
                    dimension: indices[:, 0],
                }
            )

        selected = satellite.isel(
            {
                dimension: indices.reshape(-1),
            }
        )

        selected = selected.assign_coords(
            {
                "_neighbour": (
                    (
                        dimension,
                    ),
                    np.repeat(
                        np.arange(
                            indices.shape[0]
                        ),
                        indices.shape[1],
                    ),
                )
            }
        )

        selected = selected.swap_dims(
            {
                dimension: "_neighbour"
            }
        )

        selected = selected.drop_vars(
            dimension,
            errors="ignore",
        )

        if self.aggregation == "mean":

            return selected.groupby(
                "_neighbour"
            ).mean()

        weights = 1.0 / (
            distances + 1e-12
        )

        weights /= weights.sum(
            axis=1,
            keepdims=True,
        )

        weight_array = xr.DataArray(
            weights.reshape(-1),
            dims=("_neighbour",),
        )

        return selected.weighted(
            weight_array
        ).mean(
            dim="_neighbour"
        )

    def collocate(
        self,
        satellite: xr.Dataset,
        airborne: xr.Dataset,
    ) -> xr.Dataset:

        if not isinstance(
            satellite,
            xr.Dataset,
        ):
            raise TypeError(
                "satellite must be an xarray.Dataset."
            )

        if not isinstance(
            airborne,
            xr.Dataset,
        ):
            raise TypeError(
                "airborne must be an xarray.Dataset."
            )

        airborne_dimension = (
            self._validate_coordinates(
                airborne,
                self.airborne_lat,
                self.airborne_lon,
            )
        )

        (
            satellite_valid,
            satellite_dimension,
            _,
        ) = self._prepare_satellite(
            satellite
        )

        satellite_latitude = np.asarray(
            satellite_valid[
                self.satellite_lat
            ].values,
            dtype=float,
        )

        satellite_longitude = np.asarray(
            satellite_valid[
                self.satellite_lon
            ].values,
            dtype=float,
        )

        tree_coordinates = (
            self._coordinates_to_radians(
                satellite_latitude,
                satellite_longitude,
            )
        )

        tree = BallTree(
            tree_coordinates,
            metric="haversine",
        )

        airborne_latitude = np.asarray(
            airborne[
                self.airborne_lat
            ].values,
            dtype=float,
        )

        airborne_longitude = np.asarray(
            airborne[
                self.airborne_lon
            ].values,
            dtype=float,
        )

        distances, indices = self._query(
            tree,
            airborne_latitude,
            airborne_longitude,
        )

        nearest_distance = distances[:, 0]

        valid_match = np.isfinite(
            nearest_distance
        )

        if self.max_distance is not None:

            valid_match &= (
                nearest_distance
                <= self.max_distance
            )

        if not np.any(valid_match):

            return airborne.isel(
                {
                    airborne_dimension: slice(
                        0,
                        0,
                    )
                }
            )

        airborne_indices = np.flatnonzero(
            valid_match
        )

        result = airborne.isel(
            {
                airborne_dimension: airborne_indices
            }
        ).copy()

        matched_indices = indices[
            valid_match
        ]

        matched_distances = distances[
            valid_match
        ]

        if self.aggregation == "nearest":

            satellite_match = (
                satellite_valid.isel(
                    {
                        satellite_dimension:
                            matched_indices[:, 0]
                    }
                )
            )

        else:

            satellite_match = self._aggregate(
                satellite_valid,
                matched_indices,
                matched_distances,
                satellite_dimension,
            )

        rename_satellite = {}

        for variable in satellite_match.data_vars:

            rename_satellite[
                variable
            ] = f"satellite_{variable}"

        satellite_match = satellite_match.rename(
            rename_satellite
        )

        satellite_match = (
            satellite_match.drop_vars(
                satellite_dimension,
                errors="ignore",
            )
        )

        if satellite_match.sizes:

            satellite_match = (
                satellite_match.assign_coords(
                    {
                        airborne_dimension: (
                            result[
                                airborne_dimension
                            ].values
                        )
                    }
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
        ] = xr.DataArray(
            nearest_distance[
                valid_match
            ],
            dims=(
                airborne_dimension,
            ),
            coords={
                airborne_dimension:
                    result[
                        airborne_dimension
                    ]
                }
            },
        )

        if (
            self.airborne_alt is not None
            and self.satellite_alt is not None
            and self.max_altitude_difference
            is not None
            and self.airborne_alt in result
            and f"satellite_{self.satellite_alt}"
            in result
        ):

            altitude_difference = np.abs(
                result[
                    self.airborne_alt
                ]
                - result[
                    f"satellite_{self.satellite_alt}"
                ]
            )

            result[
                "altitude_difference_m"
            ] = altitude_difference

            result = result.where(
                altitude_difference
                <= self.max_altitude_difference,
                drop=True,
            )

        result.attrs.update(
            {
                "spatial_collocation": True,
                "spatial_aggregation": (
                    self.aggregation
                ),
                "spatial_max_distance_m": (
                    self.max_distance
                ),
            }
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
