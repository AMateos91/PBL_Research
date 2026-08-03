from enum import StrEnum
from enum import Enum


class ResamplingMethod(Enum):

    NEAREST = "nearest"

    BILINEAR = "bilinear"

    CUBIC = "cubic"

    AVERAGE = "average"

    MODE = "mode"

    LANCZOS = "lanczos"


class CRS(Enum):

    WGS84 = "EPSG:4326"

    WEB_MERCATOR = "EPSG:3857"

    UTM_15N = "EPSG:32615"

    UTM_16N = "EPSG:32616"

    UTM_17N = "EPSG:32617"

    UTM_18N = "EPSG:32618"

    UTM_19N = "EPSG:32619"


class Dataset(StrEnum):

    HLSL30 = "HLSL30"
    HLSS30 = "HLSS30"
    LANDSAT = "LANDSAT"
    MODIS = "MODIS"
    VIIRS = "VIIRS"
    ECOSTRESS = "ECOSTRESS"
    GEDI = "GEDI"
    MERRA2 = "MERRA2"
    ACTIVATE = "ACTIVATE"


class SpectralBand(StrEnum):

    COASTAL = "B01"
    BLUE = "B02"
    GREEN = "B03"
    RED = "B04"
    RED_EDGE_1 = "B05"
    RED_EDGE_2 = "B06"
    RED_EDGE_3 = "B07"
    NIR = "B08"
    NARROW_NIR = "B8A"
    SWIR1 = "B11"
    SWIR2 = "B12"


class SpectralIndex(StrEnum):

    NDVI = "NDVI"
    NDWI = "NDWI"
    EVI = "EVI"
    SAVI = "SAVI"
    MSAVI = "MSAVI"
    NBR = "NBR"
    ALBEDO = "ALBEDO"
    LAI = "LAI"


class MeteorologicalVariable(StrEnum):

    AIR_TEMPERATURE = "T"
    SURFACE_TEMPERATURE = "LST"
    SPECIFIC_HUMIDITY = "QV"
    RELATIVE_HUMIDITY = "RH"
    SURFACE_PRESSURE = "PS"
    WIND_U = "U"
    WIND_V = "V"
    PBL_HEIGHT = "PBLH"
    LATENT_HEAT = "LE"
    SENSIBLE_HEAT = "H"
    NET_RADIATION = "RN"


class FileFormat(StrEnum):

    NETCDF = ".nc"
    HDF5 = ".h5"
    HDF = ".hdf"
    TIFF = ".tif"
    GEOTIFF = ".tiff"
    ZARR = ".zarr"
    PARQUET = ".parquet"
    CSV = ".csv"


class CRS(StrEnum):

    WGS84 = "EPSG:4326"
    WEB_MERCATOR = "EPSG:3857"


class Platform(StrEnum):

    LANDSAT = "Landsat"
    SENTINEL2 = "Sentinel-2"
    TERRA = "Terra"
    AQUA = "Aqua"
    ISS = "ISS"


DEFAULT_CRS = CRS.WGS84

DEFAULT_RESOLUTION = 30

DEFAULT_RANDOM_SEED = 42

DEFAULT_CHUNK_SIZE = 1024

DEFAULT_FLOAT_DTYPE = "float32"

DEFAULT_INT_DTYPE = "int16"
