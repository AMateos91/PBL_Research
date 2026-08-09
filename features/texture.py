from __future__ import annotations

import numpy as np
import xarray as xr

from skimage.feature import graycomatrix
from skimage.feature import graycoprops

from .base import Feature
from ...utils.constants import TextureMetric


class Texture(Feature):

    def __init__(
        self,
        distances: tuple[int, ...] = (1,),
        angles: tuple[float, ...] = (0.0,),
        levels: int = 32,
    ):

        super().__init__()

        self.distances = distances

        self.angles = angles

        self.levels = levels

    def compute(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:

        output = dataset.copy()

        for variable in dataset.data_vars.values():

            values = variable.values.astype(
                np.float32
            )

            minimum = np.nanmin(values)

            maximum = np.nanmax(values)

            if maximum == minimum:

                continue

            scaled = (
                (
                    values - minimum
                )
                / (
                    maximum - minimum
                )
                * (
                    self.levels - 1
                )
            ).astype(np.uint8)

            glcm = graycomatrix(

                scaled,

                distances=self.distances,

                angles=self.angles,

                levels=self.levels,

                symmetric=True,

                normed=True,

            )

            for metric in TextureMetric:

                output[
                    f"{variable.name}_{metric.value.lower()}"
                ] = graycoprops(
                    glcm,
                    metric.value,
                ).mean()

        output.attrs.update(
            dataset.attrs
        )

        output.attrs[
            "texture_features"
        ] = True

        return output
