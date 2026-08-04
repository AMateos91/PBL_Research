from __future__ import annotations

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)


class Validator:

    @staticmethod
    def kfold(
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> KFold:

        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    @staticmethod
    def stratified(
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> StratifiedKFold:

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
