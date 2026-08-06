from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .callbacks import Callback
from .early_stopping import EarlyStopping


class Trainer:
    """
    Trainer for PyTorch neural network models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        device: str = "cpu",
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        callbacks: list[Callback] | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> None:

        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.early_stopping = early_stopping

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "validation_loss": [],
        }

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

        return float(loss.item())

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

        return float(loss.item())

    @torch.no_grad()
    def validate(
        self,
        validation_loader: DataLoader,
    ) -> float:

        losses: list[float] = []

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
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        epochs: int = 1,
    ) -> dict[str, list[float]]:

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        if self.early_stopping is not None:
            self.early_stopping.reset()

        self.history = {
            "train_loss": [],
            "validation_loss": [],
        }

        for callback in self.callbacks:
            callback.on_train_begin()

        stop_training = False

        for epoch in range(epochs):

            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            running_loss = 0.0

            progress = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
            )

            for batch, (x, y) in progress:

                for callback in self.callbacks:
                    callback.on_batch_begin(batch)

                loss = self.train_step(
                    x,
                    y,
                )

                running_loss += loss

                progress.set_postfix(
                    train_loss=f"{loss:.6f}"
                )

                for callback in self.callbacks:
                    callback.on_batch_end(
                        batch,
                        {
                            "loss": loss,
                        },
                    )

            running_loss /= len(train_loader)

            self.history["train_loss"].append(
                running_loss
            )

            logs = {
                "train_loss": running_loss,
            }

            message = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {running_loss:.6f}"
            )

            if validation_loader is not None:

                validation_loss = self.validate(
                    validation_loader,
                )

                self.history[
                    "validation_loss"
                ].append(validation_loss)

                logs[
                    "validation_loss"
                ] = validation_loss

                message += (
                    f" | Validation Loss: "
                    f"{validation_loss:.6f}"
                )

                if (
                    self.early_stopping is not None
                    and self.early_stopping.step(
                        validation_loss
                    )
                ):

                    print(
                        "Early stopping triggered."
                    )

                    stop_training = True

            if self.scheduler is not None:
                self.scheduler.step()

            for callback in self.callbacks:
                callback.on_epoch_end(
                    epoch,
                    logs,
                )

            print(message)

            if stop_training:
                break

        for callback in self.callbacks:
            callback.on_train_end()

        return self.history
