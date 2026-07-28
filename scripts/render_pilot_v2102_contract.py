#!/usr/bin/env python3
"""Render compact and expanded V2.10.2 pilot contracts deterministically."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_10_1,
    PILOT_CONTRACT_ID_V2_10_2,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_2,
    PILOT_CONTRACT_SCHEMA_VERSION_V2,
    PILOT_CONTRACT_TAG_V2_10_2,
    PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256,
    _v2_10_2_expected_p95_consumer_adapter_retry_amendment,
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


def _stage_text(*, prefix: str, text: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=EXPERIMENTS,
        prefix=prefix,
        suffix=".json",
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _restore(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
        return
    staged = _stage_text(
        prefix=f".{path.name}.restore.",
        text=previous.decode("utf-8"),
    )
    os.replace(staged, path)


def _publish_pair(
    *,
    staged_overlay: Path,
    staged_expanded: Path,
    overlay_path: Path,
    expanded_path: Path,
) -> None:
    previous_overlay = overlay_path.read_bytes() if overlay_path.exists() else None
    previous_expanded = expanded_path.read_bytes() if expanded_path.exists() else None
    try:
        os.replace(staged_expanded, expanded_path)
        os.replace(staged_overlay, overlay_path)
    except Exception:
        _restore(expanded_path, previous_expanded)
        _restore(overlay_path, previous_overlay)
        raise


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
        "compiled_source_inventory_sha256": (
            args.compiled_source_inventory_sha256
        ),
        "sealed_manifest_inventory_sha256": (
            args.sealed_manifest_inventory_sha256
        ),
    }
    overlay = {
        "schema_version": PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_2,
        "contract_id": PILOT_CONTRACT_ID_V2_10_2,
        "status": args.status,
        "base_contract": {
            "path": "pilot_v2_10_1.yaml",
            "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
            "contract_id": PILOT_CONTRACT_ID_V2_10_1,
            "canonical_sha256": (
                PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256
            ),
        },
        "changes": {
            "implementation": {
                "required_git_tag": PILOT_CONTRACT_TAG_V2_10_2,
            },
            "release_requirements": {
                "tag": PILOT_CONTRACT_TAG_V2_10_2,
                "expected_ci": expected_ci,
            },
            "denominator_policy": {
                "policy_id": "finevo-pilot-v2.10.2-itt",
            },
        },
        "p95_consumer_adapter_retry_amendment": (
            _v2_10_2_expected_p95_consumer_adapter_retry_amendment(
                status=args.status,
            )
        ),
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
            "declared_sha256": "0" * 64,
        },
    }
    overlay["integrity"]["declared_sha256"] = canonical_contract_sha256(
        overlay
    )
    overlay_path = EXPERIMENTS / "pilot_v2_10_2_overlay.yaml"
    expanded_path = EXPERIMENTS / "pilot_v2_10_2.yaml"
    staged_overlay = _stage_text(
        prefix=".pilot_v2_10_2_overlay.",
        text=_canonical_text(overlay),
    )
    staged_expanded: Path | None = None
    try:
        expanded = load_pilot_contract(staged_overlay).to_dict()
        staged_expanded = _stage_text(
            prefix=".pilot_v2_10_2.",
            text=_canonical_text(expanded),
        )
        if load_pilot_contract(staged_expanded).to_dict() != expanded:
            raise RuntimeError("staged V2.10.2 expanded contract failed round-trip")
        _publish_pair(
            staged_overlay=staged_overlay,
            staged_expanded=staged_expanded,
            overlay_path=overlay_path,
            expanded_path=expanded_path,
        )
    finally:
        staged_overlay.unlink(missing_ok=True)
        if staged_expanded is not None:
            staged_expanded.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
