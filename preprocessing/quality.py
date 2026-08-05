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

        if "Fmask" not in dataset:

            return dataset

        fmask = dataset["Fmask"].astype("uint8")

        cloud = (fmask & (1 << 1)) != 0
        shadow = (fmask & (1 << 3)) != 0
        snow = (fmask & (1 << 4)) != 0
        water = (fmask & (1 << 5)) != 0

        dataset["cloud"] = cloud
        dataset["shadow"] = shadow
        dataset["snow"] = snow
        dataset["water"] = water

        dataset["valid"] = ~(
            cloud |
            shadow |
            snow
        )

        if self.drop_invalid:

            dataset = dataset.where(
                dataset["valid"],
                drop=False,
            )

        dataset.attrs["quality_checked"] = True

        return dataset
