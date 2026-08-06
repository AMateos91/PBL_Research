from __future__ import annotations

from collections.abc import Iterable

import xarray as xr

from .base import Feature


class FeaturePipeline:
    """
    Sequential feature engineering pipeline.
    """

    def __init__(
        self,
        features: Iterable[Feature],
    ) -> None:

        self.features = list(features)

    def add(
        self,
        feature: Feature,
    ) -> None:

        self.features.append(feature)

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """
        Apply every feature extractor sequentially.
        """

        for feature in self.features:

            dataset = feature.compute(dataset)

        return dataset

    def __call__(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        return self.compute(dataset)

    def __len__(
        self,
    ) -> int:

        return len(self.features)

    def __iter__(
        self,
    ):

        return iter(self.features)
