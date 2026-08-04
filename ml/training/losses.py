from __future__ import annotations

import torch.nn as nn

from ...utils.constants import LossFunction


class Loss:

    _LOSSES = {

        LossFunction.MSE: nn.MSELoss,

        LossFunction.MAE: nn.L1Loss,

        LossFunction.HUBER: nn.HuberLoss,

        LossFunction.CROSS_ENTROPY: nn.CrossEntropyLoss,

        LossFunction.BCE: nn.BCELoss,

        LossFunction.BCE_LOGITS: nn.BCEWithLogitsLoss,

    }

    @classmethod
    def get(
        cls,
        loss: LossFunction,
        **kwargs,
    ) -> nn.Module:

        if loss not in cls._LOSSES:

            raise ValueError(
                f"Unknown loss function: {loss}"
            )

        return cls._LOSSES[
            loss
        ](
            **kwargs,
        )
