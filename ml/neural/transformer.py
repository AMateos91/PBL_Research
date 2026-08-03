from __future__ import annotations

import torch
import torch.nn as nn

from .base import NeuralModel


class TransformerNetwork(nn.Module):

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        output_size: int = 1,
    ) -> None:

        super().__init__()

        self.embedding = nn.Linear(
            input_size,
            d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(
            d_model,
            output_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.embedding(x)

        x = self.encoder(x)

        return self.fc(
            x[:, -1]
        )


class TransformerModel(NeuralModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            TransformerNetwork(
                **kwargs,
            )
        )
