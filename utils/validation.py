from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .constants import Dataset
from .constants import SpectralBand
from .exceptions import BoundingBoxError
from .exceptions import ConfigurationError
from .exceptions import DatasetNotFoundError
from .exceptions import InvalidBandError


class Validator:

    @staticmethod
    def date(date: str) -> None:

        try:
            datetime.strptime(
                date,
                "%Y-%m-%d"
            )
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid date: {date}"
            ) from exc

    @staticmethod
    def time_range(
        start: str,
        end: str,
    ) -> None:

        Validator.date(start)
        Validator.date(end)

        if start > end:
            raise ConfigurationError(
                "Start date must precede end date."
            )

    @staticmethod
    def bounding_box(
        bbox: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> None:

        if len(bbox) != 4:
            raise BoundingBoxError(
                "Bounding box must contain four coordinates."
            )

        west, south, east, north = bbox

        if west >= east:
            raise BoundingBoxError(
                "Invalid longitude limits."
            )

        if south >= north:
            raise BoundingBoxError(
                "Invalid latitude limits."
            )

    @staticmethod
    def path(path: Path) -> None:

        if not path.exists():
            raise FileNotFoundError(path)

    @staticmethod
    def directory(path: Path) -> None:

        if not path.is_dir():
            raise NotADirectoryError(path)

    @staticmethod
    def dataset(name: str) -> None:

        if name not in Dataset._value2member_map_:
            raise DatasetNotFoundError(name)

    @staticmethod
    def band(name: str) -> None:

        if name not in SpectralBand._value2member_map_:
            raise InvalidBandError(name)

    @staticmethod
    def resolution(
        resolution: int | float,
    ) -> None:

        if resolution <= 0:
            raise ConfigurationError(
                "Resolution must be positive."
            )
