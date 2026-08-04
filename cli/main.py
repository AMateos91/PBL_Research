from __future__ import annotations

from .parser import build_parser


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    print(
        f"Workflow: {args.workflow}"
    )

    print(
        f"Configuration: {args.config}"
    )


if __name__ == "__main__":

    main()
