from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from PBL_Research.ml.neural.mlp import MLPModel
from PBL_Research.ml.training import Trainer


def main() -> None:

    torch.manual_seed(42)

    input_size = 10
    output_size = 1
    samples = 1000

    x = torch.randn(
        samples,
        input_size,
    )

    y = (
        x.sum(
            dim=1,
            keepdim=True,
        )
        + 0.1 * torch.randn(
            samples,
            output_size,
        )
    )

    dataset = TensorDataset(
        x,
        y,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    model = MLPModel(
        input_size=input_size,
        hidden_sizes=[128, 64],
        output_size=output_size,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss = nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss=loss,
        device="cpu",
    )

    trainer.fit(
        train_loader=train_loader,
        epochs=10,
    )


if __name__ == "__main__":
    main()
