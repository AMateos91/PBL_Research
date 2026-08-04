from __future__ import annotations

from .workflow import Workflow


class Pipeline(Workflow):

    def __init__(
        self,
        workflows: list[Workflow],
    ) -> None:

        self.workflows = workflows

    def run(
        self,
        data,
    ):

        for workflow in self.workflows:

            data = workflow.run(
                data,
            )

        return data
