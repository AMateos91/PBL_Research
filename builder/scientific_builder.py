from __future__ import annotations

import pandas as pd


class ScientificBuilder:

    def __init__(
        self,
    ) -> None:

        self._frames: list[pd.DataFrame] = []

    def add(
        self,
        frame: pd.DataFrame,
    ) -> "ScientificBuilder":

        if not isinstance(
            frame,
            pd.DataFrame,
        ):

            raise TypeError(
                "Expected pandas.DataFrame."
            )

        self._frames.append(
            frame.copy()
        )

        return self

    def extend(
        self,
        frames: list[pd.DataFrame],
    ) -> "ScientificBuilder":

        for frame in frames:

            self.add(
                frame,
            )

        return self

    def clear(
        self,
    ) -> None:

        self._frames.clear()

    @property
    def empty(
        self,
    ) -> bool:

        return len(
            self._frames
        ) == 0

    def build(
        self,
        reset_index: bool = True,
    ) -> pd.DataFrame:

        if self.empty:

            raise ValueError(
                "No datasets have been added."
            )

        dataset = pd.concat(

            self._frames,

            axis=0,

            ignore_index=reset_index,

        )

        dataset = dataset.drop_duplicates()

        return dataset

    def features(
        self,
        target: str,
    ) -> pd.DataFrame:

        dataset = self.build()

        if target not in dataset.columns:

            raise KeyError(
                target,
            )

        return dataset.drop(
            columns=[target],
        )

    def target(
        self,
        target: str,
    ) -> pd.Series:

        dataset = self.build()

        if target not in dataset.columns:

            raise KeyError(
                target,
            )

        return dataset[target]

    def split(
        self,
        target: str,
    ) -> tuple[pd.DataFrame, pd.Series]:

        return (

            self.features(
                target,
            ),

            self.target(
                target,
            ),

        )

    def summary(
        self,
    ) -> dict:

        dataset = self.build()

        return {

            "rows": len(
                dataset,
            ),

            "columns": len(
                dataset.columns,
            ),

            "features": list(
                dataset.columns,
            ),

        }

    def __len__(
        self,
    ) -> int:

        return len(
            self._frames,
        )

    def __repr__(
        self,
    ) -> str:

        return (

            "ScientificBuilder("
            f"datasets={len(self)}, "
            f"empty={self.empty}"
            ")"

        )
