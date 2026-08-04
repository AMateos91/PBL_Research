from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="pbl_research",
        description="Planetary Boundary Layer Research Toolkit",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file.",
    )

    parser.add_argument(
        "--workflow",
        type=str,
        choices=[
            "preprocessing",
            "training",
            "inference",
            "pipeline",
        ],
        default="pipeline",
    )

    return parser
