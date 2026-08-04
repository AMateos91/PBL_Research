from __future__ import annotations

import pandas as pd

from .exporter import Exporter


class TableExporter(Exporter):

    def export(
        self,
        table: pd.DataFrame,
        path: str,
    ) -> None:

        suffix = path.split(
            ".",
        )[-1].lower()

        if suffix == "csv":

            table.to_csv(
                path,
                index=False,
            )

        elif suffix == "xlsx":

            table.to_excel(
                path,
                index=False,
            )

        elif suffix == "parquet":

            table.to_parquet(
                path,
                index=False,
            )

        else:

            raise ValueError(
                f"Unsupported format: {suffix}"
            )
