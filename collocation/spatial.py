from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
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

    The airborne dataset may contain vertical profiles with
    dimensions (observation, level). The level dimension is
    preserved during spatial collocation.
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

        self.k = max(
            1,
            k,
        )

        if aggregation not in {
            "nearest",
            "mean",
            "distance_weighted",
        }:

            raise ValueError(
                f"Unknown aggregation method: {aggregation}"
            )

        self.aggregation = aggregation

    @staticmethod
    def _validate_dataset(
        dataset: xr.Dataset,
        latitude: str,
        longitude: str,
    ) -> None:

        if not isinstance(
            dataset,
            xr.Dataset,
        ):

            raise TypeError(
                "SpatialCollocator requires "
                "xarray.Dataset inputs."
            )

        if latitude not in dataset:

            raise ValueError(
                f"Missing latitude variable: {latitude}"
            )

        if longitude not in dataset:

            raise ValueError(
                f"Missing longitude variable: {longitude}"
            )

    @staticmethod
    def _observation_dimension(
        dataset: xr.Dataset,
        latitude: str,
    ) -> str:

        dims = dataset[latitude].dims

        if len(dims) != 1:

            raise ValueError(
                "Latitude must have exactly one dimension."
            )

        return dims[0]

    @staticmethod
    def _to_radians(
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> np.ndarray:

        latitude = np.asarray(
            latitude,
            dtype=float,
        )

        longitude = np.asarray(
            longitude,
            dtype=float,
        )

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

    def _build_tree(
        self,
        satellite: xr.Dataset,
        satellite_observation_dimension: str,
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

        coordinates = self._to_radians(
            latitude[valid],
            longitude[valid],
        )

        tree = BallTree(
            coordinates,
            metric="haversine",
        )

        satellite_indices = np.flatnonzero(
            valid
        )

        return tree, satellite_indices

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

            coordinates = self._to_radians(
                latitude[valid],
                longitude[valid],
            )

            query_distances, query_indices = (
                tree.query(
                    coordinates,
                    k=self.k,
                )
            )

            query_distances *= (
                self.EARTH_RADIUS
            )

            distances[valid] = query_distances

            indices[valid] = query_indices

        return distances, indices

    def _aggregate_satellite(
        self,
        satellite: xr.Dataset,
        indices: np.ndarray,
        distances: np.ndarray,
    ) -> xr.Dataset:

        valid = indices >= 0

        if not np.any(valid):

            return xr.Dataset()

        indices = indices[valid]
        distances = distances[valid]

        selected = satellite.isel(
            {
                self._satellite_observation_dimension: (
                    indices
                )
            }
        )

        if self.aggregation == "nearest":

            return selected.isel(
                {
                    self._satellite_observation_dimension: 0
                }
            )

        if self.aggregation == "mean":

            return selected.mean(
                dim=self._satellite_observation_dimension,
                skipna=True,
            )

        weights = 1.0 / (
            distances + 1e-12
        )

        weights /= weights.sum()

        return selected.weighted(
            xr.DataArray(
                weights,
                dims=(
                    self._satellite_observation_dimension,
                ),
            )
        ).mean(
            dim=self._satellite_observation_dimension,
            skipna=True,
        )

    def _satellite_matches(
        self,
        satellite: xr.Dataset,
        indices: np.ndarray,
        distances: np.ndarray,
        satellite_observation_dimension: str,
    ) -> xr.Dataset:

        valid = indices >= 0

        if not np.any(valid):

            return xr.Dataset()

        selected_indices = indices[valid]

        selected_distances = distances[valid]

        selected = satellite.isel(
            {
                satellite_observation_dimension: (
                    selected_indices
                )
            }
        )

        if self.aggregation == "nearest":

            selected = selected.isel(
                {
                    satellite_observation_dimension: 0
                }
            )

            return selected

        if self.aggregation == "mean":

            return selected.mean(
                dim=satellite_observation_dimension,
                skipna=True,
            )

        weights = 1.0 / (
            selected_distances + 1e-12
        )

        weights /= weights.sum()

        weight_array = xr.DataArray(
            weights,
            dims=(
                satellite_observation_dimension,
            ),
        )

        return selected.weighted(
            weight_array
        ).mean(
            dim=satellite_observation_dimension,
            skipna=True,
        )

    def collocate(
        self,
        satellite: xr.Dataset,
        airborne: xr.Dataset,
    ) -> xr.Dataset:

        self._validate_dataset(
            satellite,
            self.satellite_lat,
            self.satellite_lon,
        )

        self._validate_dataset(
            airborne,
            self.airborne_lat,
            self.airborne_lon,
        )

        airborne_observation_dimension = (
            self._observation_dimension(
                airborne,
                self.airborne_lat,
            )
        )

        satellite_observation_dimension = (
            self._observation_dimension(
                satellite,
                self.satellite_lat,
            )
        )

        self._satellite_observation_dimension = (
            satellite_observation_dimension
        )

        satellite, _ = (
            satellite,
            satellite_observation_dimension,
        )

        tree, satellite_indices = (
            self._build_tree(
                satellite,
                satellite_observation_dimension,
            )
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

        distances, tree_indices = self._query(
            tree,
            airborne_latitude,
            airborne_longitude,
        )

        satellite_indices = np.where(
            tree_indices >= 0,
            satellite_indices[
                np.maximum(
                    tree_indices,
                    0,
                )
            ],
            -1,
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

            raise ValueError(
                "No spatial matches satisfy the "
                "configured distance constraints."
            )

        airborne_indices = np.flatnonzero(
            valid_match
        )

        result = airborne.isel(
            {
                airborne_observation_dimension: (
                    airborne_indices
                )
            }
        ).copy()

        match_indices = satellite_indices[
            valid_match
        ]

        match_distances = distances[
            valid_match
        ]

        nearest_satellite_indices = (
            match_indices[:, 0]
        )

        nearest_satellite_distances = (
            match_distances[:, 0]
        )

        satellite_match = satellite.isel(
            {
                satellite_observation_dimension: (
                    nearest_satellite_indices
                )
            }
        )

        satellite_match = (
            satellite_match.drop_indexes(
                satellite_observation_dimension,
                errors="ignore",
            )
        )

        satellite_match = (
            satellite_match.rename(
                {
                    variable: (
                        f"satellite_{variable}"
                    )
                    for variable in satellite_match.data_vars
                }
            )
        )

        satellite_match = (
            satellite_match.rename(
                {
                    coordinate: (
                        f"satellite_{coordinate}"
                    )
                    for coordinate in satellite_match.coords
                    if coordinate
                    not in {
                        satellite_observation_dimension,
                    }
                }
            )
        )

        satellite_match = (
            satellite_match.drop_vars(
                satellite_observation_dimension,
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
            (
                airborne[
                    self.airborne_lat
                ]
                .isel(
                    {
                        airborne_observation_dimension: (
                            airborne_indices
                        )
                    }
                )
                * 0
            )
            + nearest_satellite_distances
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
