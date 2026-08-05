from __future__ import annotations

from pathlib import Path

from PBL_Research.preprocessing.loader import Loader
from PBL_Research.preprocessing.quality import Quality
from PBL_Research.features.vegetation import Vegetation


def run_pipeline(data_dir: str = "data/raw") -> None:

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {data_path}"
        )

    files = sorted(data_path.rglob("*.tif"))

    if not files:
        raise FileNotFoundError(
            f"No GeoTIFF files found in {data_path}"
        )

    print(f"Found {len(files)} GeoTIFF files.")

    loader = Loader(masked=True)
    dataset = loader.process(files)

    print("\n=== Loader ===")
    print(dataset)

    quality = Quality(drop_invalid=True)
    dataset = quality.process(dataset)

    print("\n=== Quality ===")
    print(dataset)

    vegetation = Vegetation()
    dataset = vegetation.compute(dataset)

    print("\n=== Vegetation Features ===")
    print(dataset)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":

    DATA_DIR = "data/raw"

    run_pipeline(DATA_DIR)
