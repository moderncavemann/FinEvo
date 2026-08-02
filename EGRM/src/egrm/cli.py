"""Command-line entry points for provider-free EGRM checks."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .benchmark import run_controlled_fixture, write_result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="egrm-controlled")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = run_controlled_fixture(args.contract)
    if args.output:
        write_result(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["fixture_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
