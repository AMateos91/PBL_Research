from __future__ import annotations

import torch
import torch.nn as nn

from .base import NeuralModel


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(
            x,
        )


class UNetNetwork(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 1,
    ) -> None:

        super().__init__()

        self.encoder = DoubleConv(
            input_channels,
            64,
        )

        self.pool = nn.MaxPool2d(
            2,
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                64,
                64,
                kernel_size=2,
                stride=2,
            ),

            DoubleConv(
                64,
                64,
            ),

            nn.Conv2d(
                64,
                output_channels,
                kernel_size=1,
            ),

        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.encoder(
            x,
        )

        x = self.pool(
            x,
        )

        return self.decoder(
            x,
        )


class UNetModel(NeuralModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(

            UNetNetwork(

                **kwargs,

            )

        )
