from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .workflow import Workflow


T = TypeVar(
    "T",
)


class PreprocessingWorkflow(Workflow):

    def __init__(
        self,
        steps: list[Callable[[T], T]],
    ) -> None:

        self.steps = steps

    def run(
        self,
        data: T,
    ) -> T:

        for step in self.steps:

            data = step(
                data,
            )

        return data
