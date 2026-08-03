from __future__ import annotations

import torch
import torch.nn as nn

from .base import NeuralModel


class AutoencoderNetwork(nn.Module):

    def __init__(
        self,
        input_size: int,
        latent_size: int = 64,
    ) -> None:

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_size,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                latent_size,
            ),

            nn.ReLU(),

        )

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_size,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                input_size,
            ),

        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        latent = self.encoder(
            x,
        )

        return self.decoder(
            latent,
        )


class AutoencoderModel(NeuralModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(

            AutoencoderNetwork(

                **kwargs,

            )

        )
