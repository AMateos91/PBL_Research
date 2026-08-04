from __future__ import annotations

import torch.optim.lr_scheduler as lr_scheduler

from ...utils.constants import Scheduler


class Schedulers:

    _SCHEDULERS = {

        Scheduler.STEP: lr_scheduler.StepLR,

        Scheduler.MULTISTEP: lr_scheduler.MultiStepLR,

        Scheduler.EXPONENTIAL: lr_scheduler.ExponentialLR,

        Scheduler.COSINE: lr_scheduler.CosineAnnealingLR,

        Scheduler.REDUCE_ON_PLATEAU: (
            lr_scheduler.ReduceLROnPlateau
        ),

    }

    @classmethod
    def get(
        cls,
        scheduler: Scheduler,
        optimizer,
        **kwargs,
    ):

        if scheduler not in cls._SCHEDULERS:

            raise ValueError(
                f"Unknown scheduler: {scheduler}"
            )

        return cls._SCHEDULERS[
            scheduler
        ](
            optimizer,
            **kwargs,
        )
