from __future__ import annotations

import xarray as xr

from .base import Preprocessing


class Masking(Preprocessing):

    def __init__(
        self,
        mask_variable: str = "valid",
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

        dataset = dataset.where(
               dataset[self.mask_variable]
    )

        dataset.attrs["masked"] = True

        return dataset
