from __future__ import annotations

import numpy as np
import torch
import xarray as xr

from .base import DatasetBuilder
from .tensor import TensorDataset


class DefaultDatasetBuilder(DatasetBuilder):
    """
    Default implementation of a machine learning dataset builder.
    """

    def __init__(
        self,
        feature_variables: list[str],
        target_variable: str,
    ) -> None:
        if not feature_variables:
            raise ValueError(
                "At least one feature variable must be provided."
            )

        self.feature_variables = feature_variables
        self.target_variable = target_variable

    def build(
        self,
        dataset: xr.Dataset,
    ) -> TensorDataset:
        """
        Build a TensorDataset from an xarray.Dataset.
        """

        missing_features = [
            variable
            for variable in self.feature_variables
            if variable not in dataset
        ]

        if missing_features:
            raise ValueError(
                f"Feature variables not found: {missing_features}"
            )

        if self.target_variable not in dataset:
            raise ValueError(
                f"Target variable '{self.target_variable}' not found."
            )

        x = np.stack(
            [
                dataset[variable].values.ravel()
                for variable in self.feature_variables
            ],
            axis=1,
        )

        y = dataset[
            self.target_variable
        ].values.ravel()

        valid_mask = (
            np.isfinite(x).all(axis=1)
            & np.isfinite(y)
        )

        x = x[valid_mask]
        y = y[valid_mask]

        x = torch.as_tensor(
            x,
            dtype=torch.float32,
        )

        y = torch.as_tensor(
            y,
            dtype=torch.float32,
        )

        return TensorDataset(
            x=x,
            y=y,
        )
