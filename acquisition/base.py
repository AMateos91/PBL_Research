from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from pathlib import Path

from ..utils.logger import Logger


class Acquisition(ABC):

    def __init__(self):

        self.logger = Logger.get(
            self.__class__.__name__
        )

    @abstractmethod
    def search(
        self,
        *args,
        **kwargs,
    ):
        ...

    @abstractmethod
    def download(
        self,
        *args,
        **kwargs,
    ):
        ...

    @staticmethod
    def ensure_directory(
        directory: Path,
    ) -> Path:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @staticmethod
    def as_path(
        value,
    ) -> Path:

        return Path(value).expanduser().resolve()
