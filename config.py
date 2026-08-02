from pathlib import Path
from typing import Any

import yaml


class Config:

    def __init__(self, config_file: str | Path):

        self.path = Path(config_file).expanduser().resolve()

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        with self.path.open(
            "r",
            encoding="utf-8"
        ) as stream:

            self._config = yaml.safe_load(stream)

    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:

        keys = key.split(".")

        value = self._config

        for k in keys:

            if not isinstance(value, dict):
                return default

            value = value.get(k)

            if value is None:
                return default

        return value

    def set(
        self,
        key: str,
        value: Any
    ) -> None:

        keys = key.split(".")

        current = self._config

        for k in keys[:-1]:

            current = current.setdefault(
                k,
                {}
            )

        current[keys[-1]] = value

    def save(self) -> None:

        with self.path.open(
            "w",
            encoding="utf-8"
        ) as stream:

            yaml.dump(
                self._config,
                stream,
                sort_keys=False,
                allow_unicode=True
            )

    @property
    def data(self):

        return self._config
