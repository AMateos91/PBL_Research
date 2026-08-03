from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import xarray as xr


class QualityMask(ABC):

    @abstractmethod
    def apply(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        ...

    def __call__(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        return self.apply(
            dataset
        )
