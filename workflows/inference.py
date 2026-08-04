from __future__ import annotations

from .workflow import Workflow


class InferenceWorkflow(Workflow):

    def __init__(
        self,
        predictor,
    ) -> None:

        self.predictor = predictor

    def run(
        self,
        *args,
        **kwargs,
    ):

        return self.predictor.predict(
            *args,
            **kwargs,
        )
