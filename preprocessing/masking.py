from __future__ import annotations

import xarray as xr

from .base import Preprocessing


class Masking(Preprocessing):

    def __init__(
        self,
        mask_variable: str = "mask",
        mask_value: int | float | bool = 1,
    ):

        super().__init__()

        self.mask_variable = mask_variable

        self.mask_value = mask_value

    def process(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        if self.mask_variable not in dataset:

            return dataset

        mask = dataset[self.mask_variable]

        dataset = dataset.where(
            mask == self.mask_value
        )

        dataset.attrs["masked"] = True

        return dataset
