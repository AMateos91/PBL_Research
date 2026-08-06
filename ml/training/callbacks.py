from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


class Callback(ABC):
    """
    Base class for training callbacks.
    """

    def on_train_begin(
        self,
    ) -> None:
        """
        Called before training starts.
        """
        return

    def on_train_end(
        self,
    ) -> None:
        """
        Called after training finishes.
        """
        return

    def on_epoch_begin(
        self,
        epoch: int,
    ) -> None:
        """
        Called before an epoch starts.
        """
        return

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """
        Called after an epoch finishes.
        """
        return

    def on_batch_begin(
        self,
        batch: int,
    ) -> None:
        """
        Called before a batch starts.
        """
        return

    def on_batch_end(
        self,
        batch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """
        Called after a batch finishes.
        """
        return
