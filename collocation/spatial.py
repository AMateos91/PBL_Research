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
        self.max_altitude_difference = max_altitude_difference

        self.k = max(1, k)

        self.aggregation = aggregation

    @staticmethod
    def _to_dataframe(
        data: pd.DataFrame | xr.Dataset,
    ) -> pd.DataFrame:

        if isinstance(data, pd.DataFrame):

            return data.copy()

        if isinstance(data, xr.Dataset):

            return data.to_dataframe().reset_index()

        raise TypeError(
            f"Unsupported type: {type(data)}"
        )

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:

        missing = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def _to_radians(
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

    def _prepare_inputs(
        self,
        satellite: pd.DataFrame | xr.Dataset,
        airborne: pd.DataFrame | xr.Dataset,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        satellite = self._to_dataframe(
            satellite,
        )

        airborne = self._to_dataframe(
            airborne,
        )

        self._validate_columns(
            satellite,
            [
                self.satellite_lat,
                self.satellite_lon,
            ],
        )

        self._validate_columns(
            airborne,
            [
                self.airborne_lat,
                self.airborne_lon,
            ],
        )

        satellite = satellite.reset_index(
            drop=True,
        )

        airborne = airborne.reset_index(
            drop=True,
        )

        return satellite, airborne

    def _build_tree(
        self,
        satellite: pd.DataFrame,
    ) -> BallTree:

        coordinates = self._to_radians(
            satellite[
                self.satellite_lat
            ].to_numpy(),
            satellite[
                self.satellite_lon
            ].to_numpy(),
        )

        return BallTree(
            coordinates,
            metric="haversine",
        )

    def _query(
        self,
        tree: BallTree,
        airborne: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:

        coordinates = self._to_radians(
            airborne[
                self.airborne_lat
            ].to_numpy(),
            airborne[
                self.airborne_lon
            ].to_numpy(),
        )

        distances, indices = tree.query(
            coordinates,
            k=self.k,
        )

        distances *= self.EARTH_RADIUS

        return distances, indices

    def _aggregate(
        self,
        neighbours: pd.DataFrame,
        distances: np.ndarray,
    ) -> pd.Series:

        if self.aggregation == "nearest":

            return neighbours.iloc[0].copy()

        numeric = neighbours.select_dtypes(
            include=np.number,
        )

        result = neighbours.iloc[0].copy()

        if self.aggregation == "mean":

            result.loc[numeric.columns] = (
                numeric.mean()
            )

            return result

        if self.aggregation == "distance_weighted":

            weights = 1.0 / (
                distances + 1e-12
            )

            weights /= weights.sum()

            result.loc[numeric.columns] = np.average(
                numeric.to_numpy(),
                axis=0,
                weights=weights,
            )

            return result

        raise ValueError(
            f"Unknown aggregation method: {self.aggregation}"
        )

    def collocate(
        self,
        satellite: pd.DataFrame | xr.Dataset,
        airborne: pd.DataFrame | xr.Dataset,
    ) -> pd.DataFrame:

        satellite, airborne = self._prepare_inputs(
            satellite,
            airborne,
        )

        tree = self._build_tree(
            satellite,
        )

        distances, indices = self._query(
            tree,
            airborne,
        )

        rows = []

        for i in range(
            len(airborne)
        ):

            neighbour_indices = np.atleast_1d(
                indices[i]
            )

            neighbour_distances = np.atleast_1d(
                distances[i]
            )

            neighbours = satellite.iloc[
                neighbour_indices
            ].reset_index(
                drop=True,
            )

            satellite_row = self._aggregate(
                neighbours,
                neighbour_distances,
            )

            row = airborne.iloc[
                i
            ].to_dict()

            for key, value in satellite_row.items():

                row[
                    f"satellite_{key}"
                ] = value

            row[
                "spatial_distance_m"
            ] = float(
                neighbour_distances.min()
            )

            if (
                self.airborne_alt is not None
                and self.satellite_alt is not None
                and self.max_altitude_difference
                is not None
            ):

                if (
                    self.airborne_alt in airborne.columns
                    and self.satellite_alt in satellite.columns
                ):

                    altitude_difference = abs(
                        row[
                            self.airborne_alt
                        ]
                        - satellite_row[
                            self.satellite_alt
                        ]
                    )

                    row[
                        "altitude_difference_m"
                    ] = altitude_difference

                    if (
                        altitude_difference
                        > self.max_altitude_difference
                    ):

                        continue

            if (
                self.max_distance
                is not None
                and row[
                    "spatial_distance_m"
                ]
                > self.max_distance
            ):

                continue

            rows.append(
                row,
            )

        return pd.DataFrame(
            rows,
        )

    def __call__(
        self,
        satellite: pd.DataFrame | xr.Dataset,
        airborne: pd.DataFrame | xr.Dataset,
    ) -> pd.DataFrame:

        return self.collocate(
            satellite,
            airborne,
        )
