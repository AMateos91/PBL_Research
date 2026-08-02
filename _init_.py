from importlib.metadata import version, PackageNotFoundError

from .config import Config
from .settings import Settings

try:
    __version__ = version("pbl_research")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "Abraham Mateos Gallego"
__license__ = "MIT"

__all__ = [
    "Config",
    "Settings",
    "__version__",
]
