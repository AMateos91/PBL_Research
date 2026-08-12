from __future__ import annotations

import torch.optim as optim

from ..utils.constants import Optimizer


class Optimizers:

    _OPTIMIZERS = {

        Optimizer.SGD: optim.SGD,

        Optimizer.ADAM: optim.Adam,

        Optimizer.ADAMW: optim.AdamW,

        Optimizer.RMSPROP: optim.RMSprop,

    }

    @classmethod
    def get(
        cls,
        optimizer: Optimizer,
        parameters,
        **kwargs,
    ) -> optim.Optimizer:

        if optimizer not in cls._OPTIMIZERS:

            raise ValueError(
                f"Unknown optimizer: {optimizer}"
            )

        return cls._OPTIMIZERS[
            optimizer
        ](
            parameters,
            **kwargs,
        )
