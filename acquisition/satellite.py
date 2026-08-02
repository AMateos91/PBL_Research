from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from .earthdata import EarthData

from ..utils.types import BoundingBox
from ..utils.types import TimeRange


class Satellite(EarthData):

    SHORT_NAME = ""

    PLATFORM = ""

    SENSOR = ""

    def __init__(self):

        super().__init__()

    @property
    def metadata(self) -> dict[str, str]:

        return {

            "short_name": self.SHORT_NAME,

            "platform": self.PLATFORM,

            "sensor": self.SENSOR,

        }

    def search(
        self,
        temporal: TimeRange,
        bounding_box: BoundingBox,
        cloud_cover: float | None = None,
        **kwargs,
    ):

        parameters = {

            "short_name": self.SHORT_NAME,

            "temporal": temporal,

            "bounding_box": bounding_box,

        }

        if cloud_cover is not None:

            parameters["cloud_cover"] = (
                0,
                cloud_cover,
            )

        parameters.update(kwargs)

        return super().search(
            **parameters
        )

    def download(
        self,
        temporal: TimeRange,
        bounding_box: BoundingBox,
        directory: Path,
        cloud_cover: float | None = None,
        **kwargs,
    ):

        results = self.search(
            temporal=temporal,
            bounding_box=bounding_box,
            cloud_cover=cloud_cover,
            **kwargs,
        )

        return super().download(
            results,
            directory,
        )

    @staticmethod
    @abstractmethod
    def bands() -> dict[str, str]:
        ...

    @staticmethod
    @abstractmethod
    def qa_band() -> str:
        ...
