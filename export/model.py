from __future__ import annotations

import h5py
import joblib
import torch

from ..ml.base import Model
from ..utils.constants import ExportFormat

from .exporter import Exporter


class ModelExporter(Exporter):

    def export(
        self,
        model: Model,
        path: str,
        format: ExportFormat,
    ) -> None:

        match format:

            case ExportFormat.JOBLIB:

                joblib.dump(
                    model,
                    path,
                )

            case ExportFormat.TORCH:

                torch.save(
                    model.model.state_dict(),
                    path,
                )

            case ExportFormat.HDF5:

                with h5py.File(
                    path,
                    "w",
                ) as file:

                    for name, parameter in (
                        model.model.state_dict().items()
                    ):

                        file.create_dataset(
                            name,
                            data=parameter.cpu().numpy(),
                        )

            case _:

                raise ValueError(
                    f"Unsupported export format: {format}"
                )
