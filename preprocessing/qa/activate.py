from __future__ import annotations

import xarray as xr

from .base import QualityMask


class ACTIVATEQuality(QualityMask):

    def apply(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        return dataset
