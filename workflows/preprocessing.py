from __future__ import annotations

from .workflow import Workflow


class PreprocessingWorkflow(Workflow):

    def __init__(
        self,
        steps: list,
    ) -> None:

        self.steps = steps

    def run(
        self,
        data,
    ):

        for step in self.steps:

            data = step(
                data,
            )

        return data
