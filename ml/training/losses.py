from __future__ import annotations

import torch.nn as nn

from ..utils.constants import LossFunction


class Loss:
    """
    Factory class for PyTorch loss functions.
    """

    _LOSSES: dict[
        LossFunction,
        type[nn.Module],
    ] = {
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
        """
        Return an instantiated PyTorch loss function.

        Parameters
        ----------
        loss : LossFunction
            Loss function identifier.

        **kwargs
            Keyword arguments forwarded to the loss constructor.

        Returns
        -------
        nn.Module
            Instantiated loss function.
        """

        try:
            loss_cls = cls._LOSSES[loss]
        except KeyError as exc:
            raise ValueError(
                f"Unknown loss function: {loss}"
            ) from exc

        return loss_cls(**kwargs)
