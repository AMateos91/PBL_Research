from __future__ import annotations

from typing import TypeVar

from .workflow import Workflow


T = TypeVar(
    "T",
)


class Pipeline(Workflow):

    def __init__(
        self,
        workflows: list[Workflow],
    ) -> None:

        self.workflows = workflows

    def run(
        self,
        data: T,
    ) -> T:

        for workflow in self.workflows:

            data = workflow.run(
                data,
            )

        return data
