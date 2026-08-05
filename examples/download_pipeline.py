from __future__ import annotations

from pathlib import Path

from .acquisition.earthdata import EarthData
from .utils.logger import Logger


class PBLPipeline:

    def __init__(
        self,
        output_dir: str = "data/raw",
    ):

        self.logger = Logger.get(__name__)

        self.output_dir = Path(output_dir)

        self.earthdata = EarthData()


    def run(
        self,
        **search_params,
    ) -> list[Path]:

        self.logger.info(
            "Launching PBL Research pipeline."
        )

        results = self.earthdata.search(
            **search_params
        )

        self.logger.info(
            f"Found {len(results)} granules."
        )

        files = self.earthdata.download(
            results,
            self.output_dir,
        )

        self.logger.info(
            f"Downloaded {len(files)} files."
        )

        return files


if __name__ == "__main__":

    pipeline = PBLPipeline()

    files = pipeline.run(
        short_name="HLSL30",
        temporal=("2024-06-01", "2024-06-02"),
        bounding_box=(-122.6, 37.6, -122.3, 37.9),
    )

    print("\nDownloaded files:")

    for file in files:
        print(file)
