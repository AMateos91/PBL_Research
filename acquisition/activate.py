from __future__ import annotations

from enum import Enum
from pathlib import Path

from .earthdata import EarthData

from ..utils.types import BoundingBox
from ..utils.types import TimeRange


class ACTIVATEProduct(Enum):

    HSRL2 = "ACTIVATE-HSRL2"

    RSP = "ACTIVATE-RSP"

    CPL = "ACTIVATE-CPL"

    NAVIGATION = "ACTIVATE-NAV"

    METEOROLOGY = "ACTIVATE-MET"


class ACTIVATE(EarthData):

    PLATFORM = "NASA Airborne Campaign"

    SENSOR = "Multi-Instrument Payload"

    def __init__(
        self,
        product: ACTIVATEProduct,
    ):

        super().__init__()

        self.product = product

    @property
    def short_name(self) -> str:

        return self.product.value

    @property
    def metadata(self) -> dict[str, str]:

        return {

            "short_name": self.short_name,

            "platform": self.PLATFORM,

            "sensor": self.SENSOR,

        }

    def search(
        self,
        temporal: TimeRange,
        bounding_box: BoundingBox | None = None,
        **kwargs,
    ):

        parameters = {

            "short_name": self.short_name,

            "temporal": temporal,

        }

        if bounding_box is not None:

            parameters["bounding_box"] = bounding_box

        parameters.update(kwargs)

        return super().search(
            **parameters
        )

    def download(
        self,
        temporal: TimeRange,
        directory: Path,
        bounding_box: BoundingBox | None = None,
        **kwargs,
    ):

        results = self.search(
            temporal=temporal,
            bounding_box=bounding_box,
            **kwargs,
        )

        return super().download(
            results,
            directory,
        )

    @staticmethod
    def variables() -> dict[str, str]:

        return {

            "temperature": "temperature",

            "humidity": "humidity",

            "pressure": "pressure",

            "wind_speed": "wind_speed",

            "wind_direction": "wind_direction",

            "aerosol_backscatter": "backscatter",

            "cloud_properties": "cloud",

            "flight_altitude": "altitude",

        }
