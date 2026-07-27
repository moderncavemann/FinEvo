#!/usr/bin/env python3
"""Render compact and expanded V2.10.1 pilot contracts deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_10,
    PILOT_CONTRACT_ID_V2_10_1,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_1,
    PILOT_CONTRACT_SCHEMA_VERSION_V2,
    PILOT_CONTRACT_TAG_V2_10_1,
    PILOT_CONTRACT_V2_10_CANONICAL_SHA256,
    _v2_10_1_expected_qref_receipt_verifier_retry_amendment,
    canonical_contract_sha256,
    load_pilot_contract,
)


EXPERIMENTS = ROOT / "experiments"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        choices=("draft", "frozen"),
        required=True,
    )
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--test-collection-sha256")
    parser.add_argument("--compiled-source-count", type=int)
    parser.add_argument("--compiled-source-inventory-sha256")
    parser.add_argument("--sealed-manifest-inventory-sha256")
    return parser.parse_args()


def _canonical_text(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> int:
    args = _arguments()
    if args.status == "frozen" and any(
        value is None
        for value in (
            args.test_count,
            args.test_collection_sha256,
            args.compiled_source_count,
            args.compiled_source_inventory_sha256,
            args.sealed_manifest_inventory_sha256,
        )
    ):
        raise SystemExit("frozen rendering requires all expected-CI arguments")
    expected_ci = {
        "test_count": args.test_count,
        "test_collection_sha256": args.test_collection_sha256,
        "compiled_source_count": args.compiled_source_count,
        "compiled_source_inventory_sha256": (args.compiled_source_inventory_sha256),
        "sealed_manifest_inventory_sha256": (args.sealed_manifest_inventory_sha256),
    }
    overlay = {
        "schema_version": PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_1,
        "contract_id": PILOT_CONTRACT_ID_V2_10_1,
        "status": args.status,
        "base_contract": {
            "path": "pilot_v2_10.yaml",
            "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
            "contract_id": PILOT_CONTRACT_ID_V2_10,
            "canonical_sha256": PILOT_CONTRACT_V2_10_CANONICAL_SHA256,
        },
        "changes": {
            "implementation": {
                "required_git_tag": PILOT_CONTRACT_TAG_V2_10_1,
            },
            "release_requirements": {
                "tag": PILOT_CONTRACT_TAG_V2_10_1,
                "expected_ci": expected_ci,
            },
            "denominator_policy": {
                "policy_id": "finevo-pilot-v2.10.1-itt",
            },
        },
        "qref_receipt_verifier_retry_amendment": (
            _v2_10_1_expected_qref_receipt_verifier_retry_amendment(
                status=args.status,
            )
        ),
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
            "declared_sha256": "0" * 64,
        },
    }
    overlay["integrity"]["declared_sha256"] = canonical_contract_sha256(overlay)
    overlay_path = EXPERIMENTS / "pilot_v2_10_1_overlay.yaml"
    overlay_path.write_text(_canonical_text(overlay), encoding="utf-8")
    expanded = load_pilot_contract(overlay_path).to_dict()
    (EXPERIMENTS / "pilot_v2_10_1.yaml").write_text(
        _canonical_text(expanded),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
