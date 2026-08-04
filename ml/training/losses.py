from __future__ import annotations

import torch.nn as nn


class Loss:

    _LOSSES = {

        "mse": nn.MSELoss,

        "mae": nn.L1Loss,

        "huber": nn.HuberLoss,

        "cross_entropy": nn.CrossEntropyLoss,

        "bce": nn.BCELoss,

        "bce_logits": nn.BCEWithLogitsLoss,

    }

    @classmethod
    def get(
        cls,
        name: str,
        **kwargs,
    ) -> nn.Module:

        if name not in cls._LOSSES:

            raise ValueError(
                f"Unknown loss: {name}"
            )

        return cls._LOSSES[
            name
        ](
            **kwargs,
        )
