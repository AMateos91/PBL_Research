from __future__ import annotations

from .satellite import Satellite


class Landsat(Satellite):

    SHORT_NAME = "LANDSAT_OT_C2_L2"
    PLATFORM = "Landsat-8/9"
    SENSOR = "OLI/TIRS"

    @staticmethod
    def bands():

        return {

            "coastal": "SR_B1",

            "blue": "SR_B2",

            "green": "SR_B3",

            "red": "SR_B4",

            "nir": "SR_B5",

            "swir1": "SR_B6",

            "swir2": "SR_B7",

        }

    @staticmethod
    def qa_band():

        return "QA_PIXEL"
