from __future__ import annotations

from pathlib import Path

import h5py


class HDFReader:
    
 def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = Path(path)

        if not self.path.exists():

            raise FileNotFoundError(
                self.path
            )

        self.file: h5py.File | None = None

 def open(
        self,
    ) -> h5py.File:

        if self.file is None:

            self.file = h5py.File(
                self.path,
                "r",
            )

        return self.file

 def close(
        self,
    ) -> None:

        if self.file is not None:

            self.file.close()

            self.file = None

 def __enter__(
        self,
    ) -> "HDFReader":

        self.open()

        return self

 def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.close()

 def _walk(
        self,
        group: h5py.Group,
        prefix: str = "",
    ) -> list[tuple[str, object]]:

        items = []

        for name, obj in group.items():

            path = (
                f"{prefix}/{name}"
                if prefix
                else f"/{name}"
            )

            items.append(
                (
                    path,
                    obj,
                )
            )

            if isinstance(
                obj,
                h5py.Group,
            ):

                items.extend(
                    self._walk(
                        obj,
                        path,
                    )
                )

        return items

 @property
 def groups(
        self,
    ) -> list[str]:

        file = self.open()

        return [

            path

            for path, obj in self._walk(
                file,
            )

            if isinstance(
                obj,
                h5py.Group,
            )

        ]

 @property
 def datasets(
        self,
    ) -> list[str]:

        file = self.open()

        return [

            path

            for path, obj in self._walk(
                file,
            )

            if isinstance(
                obj,
                h5py.Dataset,
            )

        ]

 def tree(
        self,
    ) -> None:

        file = self.open()

        def visit(
            group,
            level: int = 0,
        ) -> None:

            indent = "    " * level

            for name, obj in group.items():

                if isinstance(obj, h5py.Group):

                    print(f"{indent}[Group] {name}")

                    visit(
                        obj,
                        level + 1,
                    )

                elif isinstance(obj, h5py.Dataset):

                    print(
                        f"{indent}[Dataset] {name} "
                        f"shape={obj.shape} "
                        f"dtype={obj.dtype}"
                    )

        visit(file)

 def search(
        self,
        keyword: str,
    ) -> list[str]:

        keyword = keyword.lower()

        matches = []

        for dataset in self.datasets:

            if keyword in dataset.lower():

                matches.append(
                    dataset,
                )

        return matches

 def attributes(
        self,
        path: str,
    ) -> dict:

        file = self.open()

        if path not in file:

            raise KeyError(
                f"{path} not found."
            )

        obj = file[path]

        attributes = {}

        for key, value in obj.attrs.items():

            try:

                if hasattr(
                    value,
                    "tolist",
                ):

                    value = value.tolist()

                elif isinstance(
                    value,
                    bytes,
                ):

                    value = value.decode()

            except Exception:

                pass

            attributes[key] = value

        return attributes

 def read(
        self,
        path: str,
    ):

        file = self.open()

        if path not in file:

            raise KeyError(
                f"{path} not found."
            )

        obj = file[path]

        if not isinstance(
            obj,
            h5py.Dataset,
        ):

            raise TypeError(
                f"{path} is not a dataset."
            )

        return obj[...]

 def info(
        self,
        path: str,
    ) -> dict:

        file = self.open()

        if path not in file:

            raise KeyError(
                f"{path} not found."
            )

        obj = file[path]

        if not isinstance(
            obj,
            h5py.Dataset,
        ):

            raise TypeError(
                f"{path} is not a dataset."
            )

        return {

            "shape": obj.shape,

            "dtype": str(
                obj.dtype,
            ),

            "size": obj.size,

            "ndim": obj.ndim,

            "attributes": self.attributes(
                path,
            ),

        }

 def read_all(
        self,
    ) -> dict[str, object]:

        data = {}

        for dataset in self.datasets:

            try:

                data[
                    dataset
                ] = self.read(
                    dataset,
                )

            except Exception:

                continue

        return data

 def to_xarray(
        self,
    ):

        try:

            import xarray as xr

        except ImportError as exc:

            raise ImportError(
                "xarray is required."
            ) from exc

        variables = {}

        for path in self.datasets:

            try:

                values = self.read(
                    path,
                )

                name = path.split(
                    "/"
                )[-1]

                if hasattr(
                    values,
                    "ndim",
                ):

                    if values.ndim == 1:

                        dims = (
                            "dim_0",
                        )

                    elif values.ndim == 2:

                        dims = (
                            "dim_0",
                            "dim_1",
                        )

                    elif values.ndim == 3:

                        dims = (
                            "dim_0",
                            "dim_1",
                            "dim_2",
                        )

                    else:

                        continue

                    variables[
                        name
                    ] = (
                        dims,
                        values,
                    )

            except Exception:

                continue

        return xr.Dataset(
            data_vars=variables,
        )

 def summary(
        self,
    ) -> dict:

        return {

            "file": str(
                self.path,
            ),

            "groups": len(
                self.groups,
            ),

            "datasets": len(
                self.datasets,
            ),

            "group_names": self.groups,

            "dataset_names": self.datasets,

        }

    def __repr__(
        self,
    ) -> str:

        summary = self.summary()

        return (
            f"HDFReader("
            f"file='{summary['file']}', "
            f"groups={summary['groups']}, "
            f"datasets={summary['datasets']}"
            f")"
        )
