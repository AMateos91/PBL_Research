from __future__ import annotations

from pathlib import Path

from .earthdata import EarthData

from ..utils.constants import Dataset
from ..utils.types import BoundingBox
from ..utils.types import TimeRange


class HLS(EarthData):

    def __init__(self):

        super().__init__()

    def search(
        self,
        temporal: TimeRange,
        bounding_box: BoundingBox,
        cloud_cover: float | None = None,
    ):

        kwargs = {

            "short_name": Dataset.HLSL30.value,

            "temporal": temporal,

            "bounding_box": bounding_box,

        }

        if cloud_cover is not None:

            kwargs["cloud_cover"] = (
                0,
                cloud_cover,
            )

        return super().search(
            **kwargs
        )

    def download(
        self,
        temporal: TimeRange,
        bounding_box: BoundingBox,
        directory: Path,
        cloud_cover: float | None = None,
    ):

        results = self.search(
            temporal=temporal,
            bounding_box=bounding_box,
            cloud_cover=cloud_cover,
        )

        return super().download(
            results,
            directory,
        )

    @staticmethod
    def bands():

        return {

            "blue": "B02",

            "green": "B03",

            "red": "B04",

            "nir": "B08",

            "swir1": "B11",

            "swir2": "B12",

        }

    @staticmethod
    def qa_band():

        return "Fmask"
