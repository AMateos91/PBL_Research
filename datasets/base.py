from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import xarray as xr


class DatasetBuilder(ABC):

    @abstractmethod
    def build(
        self,
        dataset: xr.Dataset,
    ):
        """
        Build a machine learning dataset.
        """

        raise NotImplementedError
