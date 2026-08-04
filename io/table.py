from __future__ import annotations

import pandas as pd

from ...utils.constants import ExportFormat

from .reader import Reader


class TableReader(Reader):

    def read(
        self,
        path: str,
        format: ExportFormat,
        **kwargs,
    ) -> pd.DataFrame:

        match format:

            case ExportFormat.CSV:

                return pd.read_csv(
                    path,
                    **kwargs,
                )

            case ExportFormat.XLSX:

                return pd.read_excel(
                    path,
                    **kwargs,
                )

            case ExportFormat.PARQUET:

                return pd.read_parquet(
                    path,
                    **kwargs,
                )

            case _:

                raise ValueError(
                    f"Unsupported format: {format}"
                )
