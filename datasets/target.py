from __future__ import annotations

import xarray as xr


class Target:
    """
    Extract the target variable from an xarray.Dataset.
    """

    def __init__(
        self,
        variable: str,
    ) -> None:
        self.variable = variable

    def build(
        self,
        dataset: xr.Dataset,
    ) -> xr.DataArray:
        """
        Return the target variable as an xarray.DataArray.
        """

        if self.variable not in dataset:
            raise ValueError(
                f"Target variable '{self.variable}' not found."
            )

        return dataset[self.variable]
