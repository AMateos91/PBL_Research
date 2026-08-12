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

class FigureExporter(Exporter):

    def export(
        self,
        figure,
        path: str,
        dpi: int = 300,
        bbox_inches: str = "tight",
    ) -> None:
        figure.canvas.draw()
        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches=bbox_inches,
        )
