from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Split:
    """
    Train/validation/test dataset splitter.
    """

    def __init__(
        self,
        train: float = 0.7,
        validation: float = 0.15,
        test: float = 0.15,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:

        total = train + validation + test

        if not np.isclose(total, 1.0):
            raise ValueError(
                "train + validation + test must sum to 1."
            )

        for name, value in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        self.train = train
        self.validation = validation
        self.test = test
        self.shuffle = shuffle
        self.random_state = random_state

    def build(
        self,
        n_samples: int,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
        """
        Split sample indices into train, validation and test sets.
        """

        if n_samples <= 0:
            raise ValueError(
                "n_samples must be greater than zero."
            )

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
            indices[train_end:validation_end],
            indices[validation_end:],
        )
