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
    Spatial collocation between satellite and airborne
    observations using a BallTree with the Haversine metric.

    Airborne vertical profiles with dimensions
    (observation, level) are preserved unchanged.
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

        if k < 1:
            raise ValueError(
                "k must be greater than or equal to 1."
            )

        if aggregation not in {
            "nearest",
            "mean",
            "distance_weighted",
        }:
            raise ValueError(
                f"Unknown aggregation method: {aggregation}"
            )

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

        self.k = int(k)
        self.aggregation = aggregation

    @staticmethod
    def _validate_coordinates(
        dataset: xr.Dataset,
        latitude: str,
        longitude: str,
    ) -> str:

        if latitude not in dataset:
            raise ValueError(
                f"Missing latitude coordinate: {latitude}"
            )

        if longitude not in dataset:
            raise ValueError(
                f"Missing longitude coordinate: {longitude}"
            )

        latitude_data = dataset[latitude]
        longitude_data = dataset[longitude]

        if latitude_data.ndim != 1:
            raise ValueError(
                f"{latitude} must be one-dimensional."
            )

        if longitude_data.ndim != 1:
            raise ValueError(
                f"{longitude} must be one-dimensional."
            )

        if latitude_data.dims != longitude_data.dims:
            raise ValueError(
                "Latitude and longitude must use "
                "the same dimension."
            )

        return latitude_data.dims[0]

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

    def _build_tree(
        self,
        satellite: xr.Dataset,
        satellite_dimension: str,
    ) -> tuple[BallTree, np.ndarray]:

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
                "latitude/longitude coordinates."
            )

        valid_indices = np.flatnonzero(valid)

        coordinates = self._radians(
            latitude[valid],
            longitude[valid],
        )

        tree = BallTree(
            coordinates,
            metric="haversine",
        )

        return tree, valid_indices

    def _query(
        self,
        tree: BallTree,
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        valid = self._valid_coordinates(
            latitude,
            longitude,
        )

        distances = np.full(
            latitude.shape,
            np.inf,
            dtype=float,
        )

        indices = np.full(
            latitude.shape,
            -1,
            dtype=int,
        )

        if not np.any(valid):
            return distances, indices

        coordinates = self._radians(
            latitude[valid],
            longitude[valid],
        )

        query_distances, query_indices = (
            tree.query(
                coordinates,
                k=1,
            )
        )

        distances[valid] = (
            query_distances[:, 0]
            * self.EARTH_RADIUS
        )

        indices[valid] = (
            query_indices[:, 0]
        )

        return distances, indices

    @staticmethod
    def _rename_satellite_variables(
        dataset: xr.Dataset,
        satellite_dimension: str,
    ) -> xr.Dataset:

        data_vars = {
            name: f"satellite_{name}"
            for name in dataset.data_vars
        }

        coordinates = {}

        for name in dataset.coords:

            if name == satellite_dimension:
                continue

            if name in {
                "latitude",
                "longitude",
                "altitude",
            }:

                coordinates[name] = (
                    f"satellite_{name}"
                )

        dataset = dataset.rename(
            {
                **data_vars,
                **coordinates,
            }
        )

        return dataset

    def _select_nearest(
        self,
        satellite: xr.Dataset,
        satellite_dimension: str,
        valid_indices: np.ndarray,
        airborne_dimension: str,
        match_indices: np.ndarray,
    ) -> xr.Dataset:

        satellite_indices = valid_indices[
            match_indices
        ]

        selected = satellite.isel(
            {
                satellite_dimension: xr.DataArray(
                    satellite_indices,
                    dims=airborne_dimension,
                )
            }
        )

        selected = selected.drop_vars(
            satellite_dimension,
            errors="ignore",
        )

        selected = self._rename_satellite_variables(
            selected,
            satellite_dimension,
        )

        return selected

    def _aggregate_neighbours(
        self,
        satellite: xr.Dataset,
        satellite_dimension: str,
        valid_indices: np.ndarray,
        airborne_dimension: str,
        query_latitude: np.ndarray,
        query_longitude: np.ndarray,
    ) -> tuple[xr.Dataset, np.ndarray]:

        coordinates = self._radians(
            query_latitude,
            query_longitude,
        )

        satellite_latitude = np.asarray(
            satellite[
                self.satellite_lat
            ].values,
            dtype=float,
        )

        satellite_longitude = np.asarray(
            satellite[
                self.satellite_lon
            ].values,
            dtype=float,
        )

        satellite_coordinates = self._radians(
            satellite_latitude[valid_indices],
            satellite_longitude[valid_indices],
        )

        tree = BallTree(
            satellite_coordinates,
            metric="haversine",
        )

        distances, indices = tree.query(
            coordinates,
            k=self.k,
        )

        distances *= self.EARTH_RADIUS

        selected_indices = valid_indices[
            indices
        ]

        selected = satellite.isel(
            {
                satellite_dimension: xr.DataArray(
                    selected_indices,
                    dims=(
                        airborne_dimension,
                        "neighbour",
                    ),
                )
            }
        )

        selected = selected.drop_vars(
            satellite_dimension,
            errors="ignore",
        )

        if self.aggregation == "mean":

            selected = selected.mean(
                dim="neighbour",
                skipna=True,
            )

        elif self.aggregation == "distance_weighted":

            weights = 1.0 / (
                distances + 1e-12
            )

            weights /= weights.sum(
                axis=1,
                keepdims=True,
            )

            weight_array = xr.DataArray(
                weights,
                dims=(
                    airborne_dimension,
                    "neighbour",
                ),
            )

            selected = selected.weighted(
                weight_array
            ).mean(
                dim="neighbour",
                skipna=True,
            )

        else:

            selected = selected.isel(
                neighbour=0
            )

        selected = self._rename_satellite_variables(
            selected,
            satellite_dimension,
        )

        return selected, distances[:, 0]

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

        satellite_dimension = (
            self._validate_coordinates(
                satellite,
                self.satellite_lat,
                self.satellite_lon,
            )
        )

        satellite_latitude = np.asarray(
            satellite[
                self.satellite_lat
            ].values,
            dtype=float,
        )

        satellite_longitude = np.asarray(
            satellite[
                self.satellite_lon
            ].values,
            dtype=float,
        )

        valid_satellite = self._valid_coordinates(
            satellite_latitude,
            satellite_longitude,
        )

        if not np.any(valid_satellite):
            raise ValueError(
                "Satellite dataset contains no valid "
                "coordinates."
            )

        valid_indices = np.flatnonzero(
            valid_satellite
        )

        tree_coordinates = self._radians(
            satellite_latitude[
                valid_satellite
            ],
            satellite_longitude[
                valid_satellite
            ],
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

        if self.k == 1:

            distances, match_indices = self._query(
                tree,
                airborne_latitude,
                airborne_longitude,
            )

            valid_match = np.isfinite(
                distances
            )

            if self.max_distance is not None:

                valid_match &= (
                    distances
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
                    airborne_dimension: (
                        airborne_indices
                    )
                }
            ).copy()

            satellite_match = (
                self._select_nearest(
                    satellite,
                    satellite_dimension,
                    valid_indices,
                    airborne_dimension,
                    match_indices[
                        valid_match
                    ],
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
                distances[
                    valid_match
                ],
                dims=(
                    airborne_dimension,
                ),
                coords={
                    airborne_dimension: result[
                        airborne_dimension
                    ]
                },
            )

        else:

            valid_airborne = (
                self._valid_coordinates(
                    airborne_latitude,
                    airborne_longitude,
                )
            )

            if not np.any(valid_airborne):

                return airborne.isel(
                    {
                        airborne_dimension: slice(
                            0,
                            0,
                        )
                    }
                )

            airborne_indices = np.flatnonzero(
                valid_airborne
            )

            result = airborne.isel(
                {
                    airborne_dimension: (
                        airborne_indices
                    )
                }
            ).copy()

            satellite_match, distances = (
                self._aggregate_neighbours(
                    satellite,
                    satellite_dimension,
                    valid_indices,
                    airborne_dimension,
                    airborne_latitude[
                        valid_airborne
                    ],
                    airborne_longitude[
                        valid_airborne
                    ],
                )
            )

            valid_distance = np.isfinite(
                distances
            )

            if self.max_distance is not None:

                valid_distance &= (
                    distances
                    <= self.max_distance
                )

            if not np.all(valid_distance):

                result = result.isel(
                    {
                        airborne_dimension: (
                            np.flatnonzero(
                                valid_distance
                            )
                        )
                    }
                )

                satellite_match = satellite_match.isel(
                    {
                        airborne_dimension: (
                            np.flatnonzero(
                                valid_distance
                            )
                        )
                    }
                )

                distances = distances[
                    valid_distance
                ]

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
                distances,
                dims=(
                    airborne_dimension,
                ),
                coords={
                    airborne_dimension: result[
                        airborne_dimension
                    ]
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
                "spatial_k": self.k,
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
