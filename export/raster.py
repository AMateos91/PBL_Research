from __future__ import annotations

import numpy as np
import rasterio

from rasterio.transform import Affine

from .exporter import Exporter


class RasterExporter(Exporter):

    def export(
        self,
        image: np.ndarray,
        path: str,
        transform: Affine,
        crs: str,
        dtype: str | None = None,
    ) -> None:

        if image.ndim == 2:

            image = image[np.newaxis, ...]

        if dtype is None:

            dtype = image.dtype

        with rasterio.open(

            path,

            "w",

            driver="GTiff",

            height=image.shape[1],

            width=image.shape[2],

            count=image.shape[0],

            dtype=dtype,

            crs=crs,

            transform=transform,

        ) as dst:

            dst.write(
                image,
            )
