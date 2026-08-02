from __future__ import annotations

from .satellite import Satellite


class HLS(Satellite):

    SHORT_NAME = "HLSL30"
    PLATFORM = "Landsat-8/9 + Sentinel-2"
    SENSOR = "OLI/TIRS + MSI"

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
