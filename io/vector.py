from __future__ import annotations

import geopandas as gpd

from .reader import Reader


class VectorReader(Reader):

    def read(
        self,
        path: str,
    ) -> gpd.GeoDataFrame:

        return gpd.read_file(
            path,
        )
