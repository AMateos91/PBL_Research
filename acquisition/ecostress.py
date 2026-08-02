from __future__ import annotations

from .satellite import Satellite


class ECOSTRESS(Satellite):

    SHORT_NAME = "ECO_L2T_LSTE"

    PLATFORM = "International Space Station"

    SENSOR = "ECOSTRESS"

    @staticmethod
    def bands():

        return {

            "lst": "LST",

            "emissivity": "EmisWB",

        }

    @staticmethod
    def qa_band():

        return "QC"
