"""CLI entry point for benchr."""

import argparse
import sys
from pathlib import Path

from benchr._results import (
    execution_result_from_json,
    build_summary_data,
    _group_execution_result,
    _extract_unique_names,
)
from benchr._output import DefaultSummaryFormatter, compare_and_print


def cli() -> None:
    parser = argparse.ArgumentParser(description="benchr - benchmark comparison tool")
    sub = parser.add_subparsers(dest="command")

    compare_parser = sub.add_parser(
        "compare",
        help="Compare benchmark JSON result files; first file is the baseline",
    )
    compare_parser.add_argument(
        "files", nargs="+", type=str, help="JSON result files to compare"
    )

    args = parser.parse_args()

    if args.command == "compare":
        files = [Path(f) for f in args.files]
        for f in files:
            if not f.exists():
                print(f"Error: file not found: {f}", file=sys.stderr)
                sys.exit(1)

        names = _extract_unique_names(files)
        results = [execution_result_from_json(f.read_text()) for f in files]

        if len(results) == 1:
            data = build_summary_data(results[0], [])
            out = DefaultSummaryFormatter().format(data)
            if out:
                print(out)
        else:
            grouped = [_group_execution_result(r, n) for r, n in zip(results, names)]
            compare_and_print(grouped)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
