from __future__ import annotations

from abc import ABC

import torch

from ..base import Model


class NeuralModel(Model, ABC):

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:

        self.model = model

    def to(
        self,
        device: str,
    ):

        self.model.to(device)

        return self

    def parameters(self):

        return self.model.parameters()

    def train(self):

        self.model.train()

    def eval(self):

        self.model.eval()

    def __call__(
        self,
        x: torch.Tensor,
    ):

        return self.model(x)

    def fit(
        self,
        *args,
        **kwargs,
    ):

        raise NotImplementedError(
            "Training is handled by ml.training.Trainer."
        )

    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        self.model.eval()

        with torch.no_grad():

            return self.model(
                x
            )

    def save(
        self,
        path: str,
    ) -> None:

        torch.save(
            self.model.state_dict(),
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
    ):

        raise NotImplementedError(
            "Each neural network must implement its own load()."
        )
