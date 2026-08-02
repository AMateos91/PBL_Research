from __future__ import annotations

from .satellite import Satellite


class GEDI(Satellite):

    SHORT_NAME = "GEDI02_A"

    PLATFORM = "International Space Station"

    SENSOR = "GEDI LiDAR"

    @staticmethod
    def bands():

        return {

            "canopy_height": "rh100",

            "relative_height": "rh98",

            "elevation": "elev_lowestmode",

            "quality_flag": "quality_flag",

        }

    @staticmethod
    def qa_band():

        return "degrade_flag"
