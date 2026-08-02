from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

from numpy.typing import NDArray


BoundingBox: TypeAlias = tuple[
    float,
    float,
    float,
    float,
]

Coordinate: TypeAlias = tuple[
    float,
    float,
]

Resolution: TypeAlias = int | float

FloatRaster: TypeAlias = NDArray[np.float32]

IntRaster: TypeAlias = NDArray[np.int16]

UIntRaster: TypeAlias = NDArray[np.uint16]

MaskRaster: TypeAlias = NDArray[np.bool_]

RasterArray: TypeAlias = NDArray[np.generic]

XarrayDataset: TypeAlias = xr.Dataset

XarrayDataArray: TypeAlias = xr.DataArray

PandasDataFrame: TypeAlias = pd.DataFrame

FilePath: TypeAlias = Path

Metadata: TypeAlias = dict[
    str,
    Any,
]

BandMap: TypeAlias = dict[
    str,
    RasterArray,
]

FeatureMap: TypeAlias = dict[
    str,
    RasterArray,
]

TimeRange: TypeAlias = tuple[
    str,
    str,
]

ChunkSize: TypeAlias = dict[
    str,
    int,
]
