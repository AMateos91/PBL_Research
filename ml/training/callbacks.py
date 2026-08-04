from __future__ import annotations

from abc import ABC


class Callback(ABC):

    def on_train_begin(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_epoch_begin(
        self,
        epoch: int,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        pass

    def on_batch_begin(
        self,
        batch: int,
    ) -> None:
        pass

    def on_batch_end(
        self,
        batch: int,
        logs: dict | None = None,
    ) -> None:
        pass
