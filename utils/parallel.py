from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any
from typing import Callable


class Parallel:

    @staticmethod
    def threads(
        function: Callable[..., Any],
        iterable,
        workers: int = 4,
    ) -> list[Any]:

        results = []

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = [
                executor.submit(
                    function,
                    item
                )
                for item in iterable
            ]

            for future in as_completed(
                futures
            ):

                results.append(
                    future.result()
                )

        return results

    @staticmethod
    def processes(
        function: Callable[..., Any],
        iterable,
        workers: int = 4,
    ) -> list[Any]:

        results = []

        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = [
                executor.submit(
                    function,
                    item
                )
                for item in iterable
            ]

            for future in as_completed(
                futures
            ):

                results.append(
                    future.result()
                )

        return results

    @staticmethod
    def map(
        function: Callable[..., Any],
        iterable,
        workers: int = 4,
        process: bool = False,
    ) -> list[Any]:

        if process:

            return Parallel.processes(
                function,
                iterable,
                workers,
            )

        return Parallel.threads(
            function,
            iterable,
            workers,
        )
