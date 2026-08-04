from __future__ import annotations

from abc import ABC, abstractmethod


class Visualizer(ABC):

    @abstractmethod
    def plot(
        self,
        *args,
        **kwargs,
    ) -> None:
        ...
