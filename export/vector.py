from __future__ import annotations

import geopandas as gpd

from .exporter import Exporter


class VectorExporter(Exporter):

    def export(
        self,
        gdf: gpd.GeoDataFrame,
        path: str,
        driver: str = "GPKG",
    ) -> None:

        gdf.to_file(

            path,

            driver=driver,

        )
