from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio

from pyproj import CRS
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.geometry import box

from .exceptions import BoundingBoxError


class Geometry:

    @staticmethod
    def bounding_box(
        west: float,
        south: float,
        east: float,
        north: float,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:

        if west >= east:
            raise BoundingBoxError(
                "Invalid longitude limits."
            )

        if south >= north:
            raise BoundingBoxError(
                "Invalid latitude limits."
            )

        return (
            west,
            south,
            east,
            north,
        )

    @staticmethod
    def polygon(
        bbox: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> Polygon:

        return box(*bbox)

    @staticmethod
    def area_of_interest(
        bbox: tuple[
            float,
            float,
            float,
            float,
        ],
        crs: str = "EPSG:4326",
    ) -> gpd.GeoDataFrame:

        return gpd.GeoDataFrame(
            geometry=[
                Geometry.polygon(bbox)
            ],
            crs=crs,
        )

    @staticmethod
    def transform_point(
        x: float,
        y: float,
        source_crs: str,
        target_crs: str,
    ) -> tuple[
        float,
        float,
    ]:

        transformer = Transformer.from_crs(
            CRS.from_user_input(source_crs),
            CRS.from_user_input(target_crs),
            always_xy=True,
        )

        return transformer.transform(
            x,
            y,
        )

    @staticmethod
    def transform_bbox(
        bbox: tuple[
            float,
            float,
            float,
            float,
        ],
        source_crs: str,
        target_crs: str,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:

        west, south = Geometry.transform_point(
            bbox[0],
            bbox[1],
            source_crs,
            target_crs,
        )

        east, north = Geometry.transform_point(
            bbox[2],
            bbox[3],
            source_crs,
            target_crs,
        )

        return (
            west,
            south,
            east,
            north,
        )

    @staticmethod
    def raster_bounds(
        raster: Path,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:

        with rasterio.open(raster) as src:

            bounds = src.bounds

        return (
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )

    @staticmethod
    def raster_crs(
        raster: Path,
    ) -> str:

        with rasterio.open(raster) as src:

            return str(src.crs)

    @staticmethod
    def intersects(
        first: Polygon,
        second: Polygon,
    ) -> bool:

        return first.intersects(second)

    @staticmethod
    def intersection(
        first: Polygon,
        second: Polygon,
    ) -> Polygon:

        return first.intersection(second)
