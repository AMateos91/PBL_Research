from __future__ import annotations

from abc import ABC, abstractmethod


class Workflow(ABC):

    @abstractmethod
    def run(
        self,
        *args,
        **kwargs,
    ):
        ...
