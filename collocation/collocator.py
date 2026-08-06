from __future__ import annotations

import pandas as pd

from .spatial import SpatialCollocator
from .temporal import TemporalCollocator


class Collocator:

    def __init__(
        self,
        spatial: SpatialCollocator | None = None,
        temporal: TemporalCollocator | None = None,
    ) -> None:

        self.spatial = (
            spatial
            if spatial is not None
            else SpatialCollocator()
        )

        self.temporal = (
            temporal
            if temporal is not None
            else TemporalCollocator()
        )

    def collocate(
        self,
        satellite: pd.DataFrame,
        airborne: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Perform spatio-temporal collocation between
        satellite observations and airborne measurements.
        """

        spatial_matches = self.spatial.collocate(
            satellite=satellite,
            airborne=airborne,
        )

        temporal_matches = self.temporal.collocate(
            satellite=spatial_matches,
            airborne=airborne,
        )

        return temporal_matches

    def __call__(
        self,
        satellite: pd.DataFrame,
        airborne: pd.DataFrame,
    ) -> pd.DataFrame:

        return self.collocate(
            satellite,
            airborne,
        )
