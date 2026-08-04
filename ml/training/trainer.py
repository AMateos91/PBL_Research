from __future__ import annotations

import torch


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss,
        device: str = "cpu",
    ) -> None:

        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.device = device

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:

        self.model.train()

        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        prediction = self.model(x)

        loss = self.loss(
            prediction,
            y,
        )

        loss.backward()

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validation_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:

        self.model.eval()

        x = x.to(self.device)
        y = y.to(self.device)

        prediction = self.model(x)

        loss = self.loss(
            prediction,
            y,
        )

        return loss.item()
