from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class EnvironmentReport:

    python_version: str
    operating_system: str
    architecture: str
    executable: str
    working_directory: Path
    earthdata_credentials: bool
    cuda_available: bool
    writable_directory: bool
    installed_packages: dict[str, str]


class Environment:

    REQUIRED_PACKAGES = (
        "earthaccess",
        "numpy",
        "pandas",
        "xarray",
        "rioxarray",
        "rasterio",
        "geopandas",
        "shapely",
        "dask",
        "pyproj",
        "matplotlib",
        "scikit_learn",
    )

    @staticmethod
    def package_version(package: str) -> str | None:

        try:
            module = importlib.import_module(package)
        except ModuleNotFoundError:
            return None

        return getattr(module, "__version__", "unknown")

    @classmethod
    def installed_packages(cls) -> dict[str, str]:

        packages = {}

        for package in cls.REQUIRED_PACKAGES:

            version = cls.package_version(package)

            if version is not None:
                packages[package] = version

        return packages

    @staticmethod
    def has_earthdata_credentials() -> bool:

        return (
            os.getenv("EARTHDATA_USERNAME") is not None
            and
            os.getenv("EARTHDATA_PASSWORD") is not None
        )

    @staticmethod
    def has_cuda() -> bool:

        return shutil.which("nvidia-smi") is not None

    @staticmethod
    def is_writable(path: Path) -> bool:

        return os.access(path, os.W_OK)

    @classmethod
    def report(
        cls,
        working_directory: Path,
    ) -> EnvironmentReport:

        return EnvironmentReport(

            python_version=platform.python_version(),

            operating_system=platform.platform(),

            architecture=platform.machine(),

            executable=sys.executable,

            working_directory=working_directory.resolve(),

            earthdata_credentials=cls.has_earthdata_credentials(),

            cuda_available=cls.has_cuda(),

            writable_directory=cls.is_writable(
                working_directory
            ),

            installed_packages=cls.installed_packages(),

        )
