from __future__ import annotations

from pathlib import Path

import earthaccess


class Downloader:

    def download(
        self,
        results,
        directory: Path,
    ) -> list[Path]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not results:
            return []

        return [
            Path(file)
            for file in earthaccess.download(
                results,
                local_path=str(directory),
            )
        ]
