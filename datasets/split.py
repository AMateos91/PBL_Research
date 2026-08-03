from __future__ import annotations

import numpy as np


class Split:

    def __init__(
        self,
        train: float = 0.7,
        validation: float = 0.15,
        test: float = 0.15,
        shuffle: bool = True,
        random_state: int = 42,
    ):

        self.train = train
        self.validation = validation
        self.test = test
        self.shuffle = shuffle
        self.random_state = random_state

    def build(
        self,
        n_samples: int,
    ):

        indices = np.arange(n_samples)

        if self.shuffle:

            rng = np.random.default_rng(
                self.random_state
            )

            rng.shuffle(indices)

        train_end = int(
            self.train * n_samples
        )

        validation_end = train_end + int(
            self.validation * n_samples
        )

        return (
            indices[:train_end],
            indices[
                train_end:validation_end
            ],
            indices[validation_end:],
        )
