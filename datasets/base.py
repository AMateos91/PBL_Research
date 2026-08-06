from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

import xarray as xr


class DatasetBuilder(ABC):
    """
    Abstract base class for machine learning dataset builders.
    """

    @abstractmethod
    def build(
        self,
        dataset: xr.Dataset,
    ) -> Any:
        """
        Build a machine learning dataset from an ``xarray.Dataset``.

        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset containing the variables required for
            machine learning.

        Returns
        -------
        Any
            Dataset representation produced by the concrete builder.
        """
        raise NotImplementedError
