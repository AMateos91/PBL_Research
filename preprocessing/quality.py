from __future__ import annotations

import xarray as xr

from .base import Preprocessing


class Quality(Preprocessing):

    def __init__(
        self,
        drop_invalid: bool = True,
    ):

        super().__init__()

        self.drop_invalid = drop_invalid

    def process(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        if "quality_flag" not in dataset:

            return dataset

        quality = dataset["quality_flag"]

        if self.drop_invalid:

            dataset = dataset.where(
                quality == 1,
                drop=False,
            )

        dataset.attrs["quality_checked"] = True

        return dataset
