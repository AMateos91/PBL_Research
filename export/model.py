from __future__ import annotations

import joblib
import torch

from ..base import Model

from .exporter import Exporter


class ModelExporter(Exporter):

    def export(
        self,
        model: Model,
        path: str,
    ) -> None:

        suffix = path.split(
            ".",
        )[-1].lower()

        if suffix == "joblib":

            joblib.dump(
                model,
                path,
            )

        elif suffix in {

            "pt",

            "pth",

        }:

            torch.save(

                model.model.state_dict(),

                path,

            )

        else:

            raise ValueError(
                f"Unsupported format: {suffix}"
            )
