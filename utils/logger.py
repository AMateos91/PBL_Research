from __future__ import annotations

import logging

from pathlib import Path


class Logger:

    _configured = False

    @classmethod
    def configure(
        cls,
        log_directory: Path,
        level: int = logging.INFO,
        filename: str = "pbl_research.log",
    ) -> None:

        if cls._configured:
            return

        log_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        logfile = log_directory / filename

        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    logfile,
                    encoding="utf-8"
                )
            ],
            force=True
        )

        cls._configured = True

    @staticmethod
    def get(
        name: str,
    ) -> logging.Logger:

        return logging.getLogger(name)
