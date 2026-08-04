import pytest

from pbl_research.acquisition.base import Satellite


def test_satellite_creation():

    satellite = Satellite()

    assert satellite is not None
