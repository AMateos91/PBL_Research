from __future__ import annotations

import pandas as pd
import xarray as xr

from .base import Preprocessing
from ..utils.constants import TemporalMethod

class Temporal(Preprocessing):

    def __init__(
        self,
        frequency: str | None = None,
        method: str = TemporalMethod.NEAREST.value,
        tolerance: str | None = None,
):

           super().__init__()

           self.frequency = frequency

           self.method = method

           self.tolerance = tolerance

    def process(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        if "time" not in dataset.coords:

            return dataset

        if self.frequency is not None:

            time = pd.date_range(

                start=dataset.time.min().item(),

                end=dataset.time.max().item(),

                freq=self.frequency,

            )

            dataset = dataset.reindex(

                time=time,

                method=self.method,

                tolerance=self.tolerance,

            )

        dataset.attrs["temporal_alignment"] = True

        return dataset
