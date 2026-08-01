#!/usr/bin/env python3
"""Replay and render the V2.11.10 three-root source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import verified_memory.pilot_contract as pilot_contract_module
import verified_memory.pilot_v21110_continuation as continuation
from verified_memory.pilot_contract import load_pilot_contract, science_design_sha256
from verified_memory.pilot_v21110_continuation import build_v21110_source_manifest


TRACKED_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_10.yaml"
TRACKED_SOURCE_MANIFEST_PATH = (
    ROOT / "experiments" / "pilot_v2_11_10_source_manifest.json"
)


def _load_contract_for_source_bootstrap(contract_path: Path):
    """Parse only an unpinned V2.11.10 draft under its temporary design pin."""

    pinned = pilot_contract_module.PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256
    if pinned is not None:
        return load_pilot_contract(contract_path)
    document = json.loads(contract_path.read_text(encoding="utf-8"))
    if document.get("contract_id") != "finevo-pilot-v2.11.10" or document.get(
        "status"
    ) != "draft":
        raise ValueError(
            "only an unpinned V2.11.10 draft may cross the source bootstrap boundary"
        )
    candidate = science_design_sha256(document)
    pilot_contract_module.PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256 = candidate
    try:
        return load_pilot_contract(contract_path)
    finally:
        pilot_contract_module.PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256 = pinned


def _validated_release_roots(
    *,
    repo_root: Path,
    failed_repo_root: Path,
    authority_repo_root: Path,
) -> tuple[Path, Path, Path]:
    """Resolve three real directories and reject every aliasing combination."""

    child = continuation._real_root(repo_root, name="V2.11.10 repository")
    failed = continuation._real_root(
        failed_repo_root,
        name="V2.11.9 failed repository",
    )
    authority = continuation._real_root(
        authority_repo_root,
        name="V2.11.5 authority repository",
    )
    continuation._require_distinct_roots(
        child=child,
        failed=failed,
        authority=authority,
    )
    return child, failed, authority


def render_source_manifest(
    *,
    contract_path: Path,
    repo_root: Path,
    failed_repo_root: Path,
    authority_repo_root: Path,
) -> tuple[dict, bytes]:
    """Return the deterministic three-root manifest and canonical bytes."""

    child, failed, authority = _validated_release_roots(
        repo_root=repo_root,
        failed_repo_root=failed_repo_root,
        authority_repo_root=authority_repo_root,
    )
    contract = _load_contract_for_source_bootstrap(contract_path)
    manifest = build_v21110_source_manifest(
        contract=contract,
        repo_root=child,
        failed_repo_root=failed,
        authority_repo_root=authority,
    )
    payload = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return manifest, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=TRACKED_CONTRACT_PATH,
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--failed-repo-root",
        type=Path,
        required=True,
        help="clean detached V2.11.9 terminal-no-go checkout",
    )
    parser.add_argument(
        "--authority-repo-root",
        type=Path,
        required=True,
        help="clean detached V2.11.5 science-authority checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TRACKED_SOURCE_MANIFEST_PATH,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify output bytes instead of writing them",
    )
    args = parser.parse_args(argv)
    manifest, payload = render_source_manifest(
        contract_path=args.contract,
        repo_root=args.repo_root,
        failed_repo_root=args.failed_repo_root,
        authority_repo_root=args.authority_repo_root,
    )
    output = args.output.resolve(strict=False)
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise SystemExit("tracked V2.11.10 source manifest drifted")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    print(
        json.dumps(
            {
                "content_sha256": manifest["integrity"]["content_sha256"],
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
