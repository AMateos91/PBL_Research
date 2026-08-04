from __future__ import annotations

from typing import TypeVar

import h5py
import joblib
import torch

from ..base import Model
from ...utils.constants import ExportFormat

from .reader import Reader


T = TypeVar(
    "T",
    bound=Model,
)


class ModelReader(Reader):

    def read(
        self,
        path: str,
        format: ExportFormat,
        model: T | None = None,
    ) -> T:

        match format:

            case ExportFormat.JOBLIB:

                return joblib.load(
                    path,
                )

            case ExportFormat.TORCH:

                if model is None:

                    raise ValueError(
                        "A model instance must be provided."
                    )

                state_dict = torch.load(
                    path,
                    map_location="cpu",
                )

                model.model.load_state_dict(
                    state_dict,
                )

                return model

            case ExportFormat.HDF5:

                if model is None:

                    raise ValueError(
                        "A model instance must be provided."
                    )

                state_dict = {}

                with h5py.File(
                    path,
                    "r",
                ) as file:

                    for name, dataset in file.items():

                        state_dict[name] = torch.tensor(
                            dataset[()],
                        )

                model.model.load_state_dict(
                    state_dict,
                )

                return model

            case _:

                raise ValueError(
                    f"Unsupported format: {format}"
                )
