from __future__ import annotations

from pathlib import Path

import rioxarray as rxr
import xarray as xr

from .base import Preprocessing


class Loader(Preprocessing):

    def __init__(
        self,
        masked: bool = True,
    ):

        super().__init__()

        self.masked = masked

    def process(
        self,
        files: list[Path],
    ) -> xr.Dataset:

        if not files:

            raise ValueError(
                "No input files were provided."
            )

        variables = {}

        for file in files:

            raster = rxr.open_rasterio(
                file,
                masked=self.masked,
            )

            if "band" in raster.dims:

                raster = raster.squeeze(
                    "band",
                    drop=True,
                )

            variables[
                file.stem.split(".")[-1]
            ] = raster

        dataset = xr.Dataset(
            variables
        )

        dataset.attrs["source"] = "LP DAAC"

        return dataset
