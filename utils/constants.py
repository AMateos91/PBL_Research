from enum import StrEnum
from enum import Enum


class Band(Enum):

    BLUE = "blue"

    GREEN = "green"

    RED = "red"

    RED_EDGE = "red_edge"

    NIR = "nir"

    SWIR = "swir"

    SWIR2 = "swir2"

    THERMAL = "thermal"


class CRS(Enum):

    WGS84 = "EPSG:4326"

    WEB_MERCATOR = "EPSG:3857"

    UTM_15N = "EPSG:32615"

    UTM_16N = "EPSG:32616"

    UTM_17N = "EPSG:32617"

    UTM_18N = "EPSG:32618"

    UTM_19N = "EPSG:32619"


class ResamplingMethod(Enum):

    NEAREST = "nearest"

    BILINEAR = "bilinear"

    CUBIC = "cubic"

    AVERAGE = "average"

    MODE = "mode"

    LANCZOS = "lanczos"


class ScalingMethod(Enum):

    STANDARD = "standard"

    MINMAX = "minmax"

    ROBUST = "robust"


class Framework(Enum):

    NUMPY = "numpy"

    PYTORCH = "pytorch"

    TENSORFLOW = "tensorflow"


class TemporalMethod(Enum):

    NEAREST = "nearest"

    PAD = "pad"

    BACKFILL = "backfill"
    

class NormalizationMethod(Enum):

    MINMAX = "minmax"

    ZSCORE = "zscore"

    NONE = "none"
    

class Compatibility(Enum):

    OVERRIDE = "override"

    IDENTICAL = "identical"

    EQUALS = "equals"

    NO_CONFLICTS = "no_conflicts"

    BROADCAST_EQUALS = "broadcast_equals"


class Variable(Enum):

    ELEVATION = "elevation"

    SLOPE = "slope"

    ASPECT = "aspect"

    ALBEDO = "albedo"

    THERMAL = "thermal"

    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    NIR = "nir"
    SWIR = "swir"
    SWIR2 = "swir2"

    NDVI = "ndvi"
    EVI = "evi"
    SAVI = "savi"

    NDWI = "ndwi"
    NDMI = "ndmi"
    MSI = "msi"

    NDBI = "ndbi"

    SURFACE_TEMPERATURE = "surface_temperature"

    TEMPERATURE_ANOMALY = "temperature_anomaly"

    ALBEDO = "albedo"

    ELEVATION = "elevation"
    SLOPE = "slope"
    ASPECT = "aspect"

    U_WIND = "u_wind"

    V_WIND = "v_wind"

    WIND_SPEED = "wind_speed"

    WIND_DIRECTION = "wind_direction"

    AIR_TEMPERATURE = "air_temperature"

    DEW_POINT_TEMPERATURE = "dew_point_temperature"

    RELATIVE_HUMIDITY = "relative_humidity"

    PRESSURE = "pressure"

    VPD = "vpd"

    SHORTWAVE_RADIATION = "shortwave_radiation"

    LONGWAVE_RADIATION = "longwave_radiation"

    SENSIBLE_HEAT_FLUX = "sensible_heat_flux"

    LATENT_HEAT_FLUX = "latent_heat_flux"

    TEMPERATURE_GRADIENT = "temperature_gradient"

    BOWEN_RATIO = "bowen_ratio"

    THERMAL_ADVECTION_PROXY = "thermal_advection_proxy"

class TextureMetric(Enum):

    CONTRAST = "contrast"

    DISSIMILARITY = "dissimilarity"

    HOMOGENEITY = "homogeneity"

    ENERGY = "energy"

    CORRELATION = "correlation"

    ASM = "ASM"


class LossFunction(Enum):

    MSE = "mse"

    MAE = "mae"

    HUBER = "huber"

    CROSS_ENTROPY = "cross_entropy"

    BCE = "bce"

    BCE_LOGITS = "bce_logits"


class Metric(Enum):

    ACCURACY = "accuracy"

    PRECISION = "precision"

    RECALL = "recall"

    F1 = "f1"

    MAE = "mae"

    MSE = "mse"

    RMSE = "rmse"

    R2 = "r2"


class Optimizer(Enum):

    SGD = "sgd"

    ADAM = "adam"

    ADAMW = "adamw"

    RMSPROP = "rmsprop"


class Scheduler(Enum):

    STEP = "step"

    MULTISTEP = "multistep"

    EXPONENTIAL = "exponential"

    COSINE = "cosine"

    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class EnsembleMethod(Enum):

    MEAN = "mean"

    MEDIAN = "median"

    WEIGHTED = "weighted"

    VOTING = "voting"
    

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
