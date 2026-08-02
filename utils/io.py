from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr


class IO:

    @staticmethod
    def create_directory(
        directory: Path,
    ) -> None:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    @staticmethod
    def remove_file(
        file: Path,
    ) -> None:

        if file.exists():
            file.unlink()

    @staticmethod
    def exists(
        path: Path,
    ) -> bool:

        return path.exists()

    @staticmethod
    def read_json(
        file: Path,
    ) -> dict[str, Any]:

        with file.open(
            "r",
            encoding="utf-8"
        ) as stream:

            return json.load(stream)

    @staticmethod
    def write_json(
        data: dict[str, Any],
        file: Path,
    ) -> None:

        with file.open(
            "w",
            encoding="utf-8"
        ) as stream:

            json.dump(
                data,
                stream,
                indent=4,
                ensure_ascii=False
            )

    @staticmethod
    def read_csv(
        file: Path,
    ) -> pd.DataFrame:

        return pd.read_csv(file)

    @staticmethod
    def write_csv(
        dataframe: pd.DataFrame,
        file: Path,
        index: bool = False,
    ) -> None:

        dataframe.to_csv(
            file,
            index=index
        )

    @staticmethod
    def read_netcdf(
        file: Path,
    ) -> xr.Dataset:

        return xr.open_dataset(file)

    @staticmethod
    def write_netcdf(
        dataset: xr.Dataset,
        file: Path,
    ) -> None:

        dataset.to_netcdf(file)

    @staticmethod
    def read_zarr(
        directory: Path,
    ) -> xr.Dataset:

        return xr.open_zarr(directory)

    @staticmethod
    def write_zarr(
        dataset: xr.Dataset,
        directory: Path,
    ) -> None:

        dataset.to_zarr(
            directory,
            mode="w"
        )
