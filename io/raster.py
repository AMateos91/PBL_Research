from __future__ import annotations

import rasterio

from .reader import Reader


class RasterReader(Reader):

    def read(
        self,
        path: str,
    ):

        with rasterio.open(
            path,
        ) as src:

            image = src.read()

            metadata = src.meta.copy()

        return image, metadata
