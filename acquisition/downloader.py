from __future__ import annotations

from pathlib import Path


class Downloader:

    def download(
        self,
        results,
        directory: Path,
    ) -> list[Path]:

        raise NotImplementedError(
            "Downloader backend not implemented."
        )
