from __future__ import annotations


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> None:

        if patience <= 0:
            raise ValueError(
                "patience must be greater than zero."
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta cannot be negative."
            )

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0

    def step(
        self,
        loss: float,
    ) -> bool:
        """
        Update the early stopping state.

        Parameters
        ----------
        loss : float
            Current validation loss.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """

        if loss < self.best_loss - self.min_delta:

            self.best_loss = loss
            self.counter = 0

            return False

        self.counter += 1

        return self.counter >= self.patience

    def reset(
        self,
    ) -> None:
        """
        Reset the early stopping state.
        """

        self.best_loss = float("inf")
        self.counter = 0
