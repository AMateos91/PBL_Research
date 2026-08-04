from __future__ import annotations

from .workflow import Workflow


class TrainingWorkflow(Workflow):

    def __init__(
        self,
        trainer,
    ) -> None:

        self.trainer = trainer

    def run(
        self,
        *args,
        **kwargs,
    ):

        return self.trainer.train(
            *args,
            **kwargs,
        )
