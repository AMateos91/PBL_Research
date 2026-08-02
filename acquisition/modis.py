from __future__ import annotations

from .satellite import Satellite


class MODIS(Satellite):

    SHORT_NAME = "MOD09GA"
    PLATFORM = "Terra"
    SENSOR = "MODIS"

    @staticmethod
    def bands():

        return {

            "red": "sur_refl_b01",

            "nir": "sur_refl_b02",

            "blue": "sur_refl_b03",

            "green": "sur_refl_b04",

            "swir1": "sur_refl_b06",

            "swir2": "sur_refl_b07",

        }

    @staticmethod
    def qa_band():

        return "state_1km"
