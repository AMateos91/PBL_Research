from __future__ import annotations

import torch
import torch.nn as nn

from .base import NeuralModel


class MLPModel(NeuralModel):

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = [128, 64],
        output_size: int = 1,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:

        layers = []

        in_features = input_size

        for hidden in hidden_sizes:

            layers.extend(
                [
                    nn.Linear(
                        in_features,
                        hidden,
                    ),
                    activation(),
                    nn.Dropout(
                        dropout,
                    ),
                ]
            )

            in_features = hidden

        layers.append(
            nn.Linear(
                in_features,
                output_size,
            )
        )

        model = nn.Sequential(
            *layers,
        )

        super().__init__(
            model,
        )

    @classmethod
    def load(
        cls,
        path: str,
        **kwargs,
    ) -> "MLPModel":

        instance = cls(
            **kwargs,
        )

        instance.model.load_state_dict(
            torch.load(
                path,
                map_location="cpu",
            )
        )

        instance.model.eval()

        return instance
