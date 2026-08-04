from __future__ import annotations

from abc import ABC, abstractmethod


class Reader(ABC):

    @abstractmethod
    def read(
        self,
        *args,
        **kwargs,
    ):
        ...
