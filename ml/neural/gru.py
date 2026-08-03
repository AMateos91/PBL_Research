from __future__ import annotations

import torch
import torch.nn as nn

from .base import NeuralModel


class GRUNetwork(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output, _ = self.gru(x)

        return self.fc(
            output[:, -1]
        )


class GRUModel(NeuralModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(
            GRUNetwork(
                **kwargs,
            )
        )
