from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:

    project_name: str = "PBL Research"
    project_version: str = "0.1.0"

    root_directory: Path = Path.cwd()

    data_directory: Path = field(init=False)
    downloads_directory: Path = field(init=False)
    cache_directory: Path = field(init=False)
    output_directory: Path = field(init=False)
    logs_directory: Path = field(init=False)
    figures_directory: Path = field(init=False)
    models_directory: Path = field(init=False)

    earthdata_username: str | None = None
    earthdata_password: str | None = None

    study_area: tuple[float, float, float, float] | None = None

    start_date: str | None = None
    end_date: str | None = None

    crs: str = "EPSG:4326"
    resolution: int = 30

    workers: int = 4
    random_seed: int = 42

    def __post_init__(self):

        self.data_directory = self.root_directory / "data"
        self.downloads_directory = self.data_directory / "downloads"
        self.cache_directory = self.data_directory / "cache"

        self.output_directory = self.root_directory / "output"
        self.logs_directory = self.output_directory / "logs"
        self.figures_directory = self.output_directory / "figures"
        self.models_directory = self.output_directory / "models"

    @property
    def directories(self):

        return (
            self.data_directory,
            self.downloads_directory,
            self.cache_directory,
            self.output_directory,
            self.logs_directory,
            self.figures_directory,
            self.models_directory,
        )

    def create_directories(self):

        for directory in self.directories:
            directory.mkdir(
                parents=True,
                exist_ok=True
            )
