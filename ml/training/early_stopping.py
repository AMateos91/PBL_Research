from __future__ import annotations


class EarlyStopping:

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> None:

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0

    def step(
        self,
        loss: float,
    ) -> bool:

        if loss < self.best_loss - self.min_delta:

            self.best_loss = loss

            self.counter = 0

            return False

        self.counter += 1

        return self.counter >= self.patience

    def reset(
        self,
    ) -> None:

        self.best_loss = float("inf")

        self.counter = 0
