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

    def train(
        self,
        train_loader,
        epochs: int = 1,
    ) -> None:

        for epoch in range(epochs):

            running_loss = 0.0

            for x, y in train_loader:

                running_loss += self.train_step(
                    x,
                    y,
                )

            running_loss /= len(train_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {running_loss:.6f}"
            )

    @torch.no_grad()
    def validate(
        self,
        validation_loader,
    ) -> float:

        losses = []

        for x, y in validation_loader:

            losses.append(
                self.validation_step(
                    x,
                    y,
                )
            )

        return sum(losses) / len(losses)

    def fit(
        self,
        train_loader,
        validation_loader=None,
        epochs: int = 1,
    ) -> None:

        for epoch in range(epochs):

            running_loss = 0.0

            for x, y in train_loader:

                running_loss += self.train_step(
                    x,
                    y,
                )

            running_loss /= len(train_loader)

            message = (
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {running_loss:.6f}"
            )

            if validation_loader is not None:

                validation_loss = self.validate(
                    validation_loader,
                )

                message += (
                    f" - Validation Loss: "
                    f"{validation_loss:.6f}"
                )

            print(message)
