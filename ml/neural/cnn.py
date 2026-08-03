from __future__ import annotations

import torch
import torch.nn as nn

from .base import NeuralModel


class CNNModel(NeuralModel):

    def __init__(
        self,
        input_channels: int,
        output_size: int = 1,
    ) -> None:

        model = nn.Sequential(

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                1,
            ),

            nn.Flatten(),

            nn.Linear(
                64,
                output_size,
            ),
        )

        super().__init__(
            model,
        )

    @classmethod
    def load(
        cls,
        path: str,
        **kwargs,
    ) -> "CNNModel":

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
