from __future__ import annotations

from abc import ABC, abstractmethod


class Exporter(ABC):

    @abstractmethod
    def export(
        self,
        *args,
        **kwargs,
    ) -> None:
        ...
