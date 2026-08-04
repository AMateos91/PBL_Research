from __future__ import annotations

import numpy as np
import torch
import xarray as xr

from .base import DatasetBuilder
from .tensor import TensorDataset


class DefaultDatasetBuilder(DatasetBuilder):

    def __init__(
        self,
        feature_variables: list[str],
        target_variable: str,
    ) -> None:

        self.feature_variables = feature_variables
        self.target_variable = target_variable

    def build(
        self,
        dataset: xr.Dataset,
    ) -> TensorDataset:

        x = np.stack(
            [
                dataset[var].values.ravel()
                for var in self.feature_variables
            ],
            axis=1,
        )

        y = dataset[
            self.target_variable
        ].values.ravel()

        x = torch.tensor(
            x,
            dtype=torch.float32,
        )

        y = torch.tensor(
            y,
            dtype=torch.float32,
        )

        return TensorDataset(
            x=x,
            y=y,
        )
