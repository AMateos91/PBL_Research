from __future__ import annotations

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

        except Exception as exc:

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
