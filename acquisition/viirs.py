from __future__ import annotations

from .satellite import Satellite


class VIIRS(Satellite):

    SHORT_NAME = "VNP09GA"
    PLATFORM = "Suomi NPP"
    SENSOR = "VIIRS"

    @staticmethod
    def bands():

        return {

            "blue": "M3",

            "green": "M4",

            "red": "M5",

            "nir": "M7",

            "swir1": "M10",

            "swir2": "M11",

        }

    @staticmethod
    def qa_band():

        return "QF2"
