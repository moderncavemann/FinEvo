#!/usr/bin/env python3
"""Expand the provider-disabled design draft into its exact 105 cells."""

from __future__ import annotations

import argparse
import json

from egrm.design import expand_scientific_design, write_design_inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    inventory = expand_scientific_design(args.contract)
    if args.output:
        write_design_inventory(inventory, args.output)
    print(json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
