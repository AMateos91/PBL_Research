from __future__ import annotations

from .satellite import Satellite


class MERRA2(Satellite):

    SHORT_NAME = "M2T1NXSLV"

    PLATFORM = "GEOS"

    SENSOR = "MERRA-2 Reanalysis"

    @staticmethod
    def bands():

        return {

            "pbl_height": "PBLH",

            "air_temperature": "T",

            "specific_humidity": "QV",

            "surface_pressure": "PS",

            "wind_u": "U",

            "wind_v": "V",

            "latent_heat_flux": "EFLUX",

            "sensible_heat_flux": "HFLUX",

        }

    @staticmethod
    def qa_band():

        return ""
