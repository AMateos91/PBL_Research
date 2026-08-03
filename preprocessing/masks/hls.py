from __future__ import annotations

import xarray as xr

from .base import Mask


class HLSMask(Mask):

    def apply(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        return dataset
