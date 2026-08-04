from __future__ import annotations

from pathlib import Path

import earthaccess

from ..utils.exceptions import AuthenticationError
from ..utils.logger import Logger
from .base import Acquisition


class EarthData(Acquisition):

    def __init__(self):

        self.logger = Logger.get(__name__)

        self._authenticated = False


    @property
    def authenticated(self) -> bool:

        return self._authenticated


    def login(self) -> None:

        try:

            earthaccess.login(
                strategy="environment"
            )

            self._authenticated = True

            self.logger.info(
                "Earthdata authentication successful."
            )


        except Exception as exc:

            self.logger.exception(
                "Earthdata authentication failed."
            )

            raise AuthenticationError(
                "Unable to authenticate with Earthdata."
            ) from exc



    def search(
        self,
        **kwargs,
    ):

        if not self.authenticated:

            self.login()


        return earthaccess.search_data(
            **kwargs
        )



    def download(
        self,
        results,
        directory: Path,
    ) -> list[Path]:

        if not results:
    raise ValueError("No Earthdata granules found.")

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        files = earthaccess.download(
            results,
            local_path=directory,
        )


        return [
            Path(file)
            for file in files
        ]
       
