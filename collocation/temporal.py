from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr


class TemporalCollocator:
    """
    Temporal collocation between airborne and satellite
    observations.
    """

    def __init__(
        self,
        satellite_time: str = "time",
        airborne_time: str = "time",
        tolerance: timedelta = timedelta(minutes=30),
    ) -> None:

        self.satellite_time = satellite_time
        self.airborne_time = airborne_time
        self.tolerance = tolerance

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
            [self.satellite_time],
        )

        self._validate_columns(
            airborne,
            [self.airborne_time],
        )

        satellite[self.satellite_time] = pd.to_datetime(
            satellite[self.satellite_time],
            utc=True,
        )

        airborne[self.airborne_time] = pd.to_datetime(
            airborne[self.airborne_time],
            utc=True,
        )

        satellite = satellite.sort_values(
            self.satellite_time,
        ).reset_index(drop=True)

        airborne = airborne.sort_values(
            self.airborne_time,
        ).reset_index(drop=True)

        return satellite, airborne

    def collocate(
        self,
        satellite: pd.DataFrame | xr.Dataset,
        airborne: pd.DataFrame | xr.Dataset,
    ) -> pd.DataFrame:
        """
        Perform temporal collocation between airborne and
        satellite observations.

        Parameters
        ----------
        satellite
            Satellite observations.

        airborne
            Airborne observations.

        Returns
        -------
        pandas.DataFrame
            Temporally collocated observations.
        """

        satellite, airborne = self._prepare_inputs(
            satellite,
            airborne,
        )

        satellite = satellite.rename(
            columns={
                self.satellite_time: "satellite_time",
            }
        )

        airborne = airborne.rename(
            columns={
                self.airborne_time: "airborne_time",
            }
        )
        
        airborne = airborne.dropna(subset=["time"]).sort_values("time")
        satellite = satellite.dropna(subset=["time"]).sort_values("time")
       
        result = pd.merge_asof(
            airborne,
            satellite,
            left_on="airborne_time",
            right_on="satellite_time",
            direction="nearest",
            tolerance=self.tolerance,
            suffixes=(
                "",
                "_satellite",
            ),
        )

        result = result.dropna(
            subset=[
                "satellite_time",
            ]
        ).reset_index(
            drop=True,
        )

        result[
            "temporal_difference"
        ] = (
            result[
                "airborne_time"
            ]
            - result[
                "satellite_time"
            ]
        ).abs()

        result[
            "temporal_difference_seconds"
        ] = (
            result[
                "temporal_difference"
            ]
            .dt.total_seconds()
        )

        return result

    def __call__(
        self,
        satellite: pd.DataFrame | xr.Dataset,
        airborne: pd.DataFrame | xr.Dataset,
    ) -> pd.DataFrame:

        return self.collocate(
            satellite,
            airborne,
        )
