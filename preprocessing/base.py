from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import xarray as xr

from ..utils.logger import Logger


class Preprocessing(ABC):

    def __init__(self):

        self.logger = Logger.get(
            self.__class__.__name__
        )

    @abstractmethod
    def process(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        ...

    def __call__(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        return self.process(
            dataset
        )
