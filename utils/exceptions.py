class PBLResearchError(Exception):
    pass


class ConfigurationError(PBLResearchError):
    pass


class AuthenticationError(PBLResearchError):
    pass


class AuthorizationError(PBLResearchError):
    pass


class ConnectionError(PBLResearchError):
    pass


class DownloadError(PBLResearchError):
    pass


class DatasetError(PBLResearchError):
    pass


class DatasetNotFoundError(DatasetError):
    pass


class DatasetCorruptedError(DatasetError):
    pass


class DatasetCompatibilityError(DatasetError):
    pass


class AirborneDatasetError(DatasetError):
    pass


class SatelliteDatasetError(DatasetError):
    pass


class MetadataError(PBLResearchError):
    pass


class ValidationError(PBLResearchError):
    pass


class GeometryError(PBLResearchError):
    pass


class CoordinateReferenceSystemError(GeometryError):
    pass


class BoundingBoxError(GeometryError):
    pass


class AreaOfInterestError(GeometryError):
    pass


class RasterError(PBLResearchError):
    pass


class RasterAlignmentError(RasterError):
    pass


class RasterResolutionError(RasterError):
    pass


class BandError(RasterError):
    pass


class MissingBandError(BandError):
    pass


class InvalidBandError(BandError):
    pass


class QualityMaskError(PBLResearchError):
    pass


class CloudMaskError(QualityMaskError):
    pass


class TemporalAlignmentError(PBLResearchError):
    pass


class FeatureExtractionError(PBLResearchError):
    pass


class SpectralIndexError(FeatureExtractionError):
    pass


class ModelError(PBLResearchError):
    pass


class TrainingError(ModelError):
    pass


class PredictionError(ModelError):
    pass


class ExportError(PBLResearchError):
    pass
