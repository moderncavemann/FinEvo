from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_contract as pilot_contract_module
from verified_memory import pilot_v2119_continuation as continuation


EXACT_RECEIPT = Path(
    "experiment_results/pilot-v2.11.9/raw/parent-import/current_authority/"
    "post_gate_authority.json"
)
EXPECTED_COMMIT = "a" * 40
EXPECTED_CONTRACT_HASH = "b" * 64
CONTENT_HASH = "c" * 64


@pytest.fixture
def v2119_adapter_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    receipt = {
        "schema_version": continuation.V2119_CURRENT_AUTHORITY_SCHEMA_VERSION,
        "integrity": {"content_sha256": CONTENT_HASH},
    }
    receipt_path = tmp_path / EXACT_RECEIPT
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reservations = {
        "openai/gpt-5.2-2025-12-11": {
            "action": {"authority": {}, "reservation": {}},
            "semantic": {"authority": {}, "reservation": {}},
        }
    }
    expected_binding = {
        "receipt_path": EXACT_RECEIPT.as_posix(),
        "receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "receipt_content_sha256": CONTENT_HASH,
        "git_commit": EXPECTED_COMMIT,
        "reservations": reservations,
    }
    calls: list[dict[str, Any]] = []

    def verify(
        receipt_path: str | Path,
        *,
        repo_root: str | Path,
        expected_git_commit: str,
        expected_contract_sha256: str,
    ) -> dict[str, Any]:
        calls.append(
            {
                "receipt_path": str(receipt_path),
                "repo_root": Path(repo_root),
                "expected_git_commit": expected_git_commit,
                "expected_contract_sha256": expected_contract_sha256,
            }
        )
        return deepcopy(expected_binding)

    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256",
        EXPECTED_CONTRACT_HASH,
    )
    monkeypatch.setattr(
        continuation,
        "verified_v2119_observed_p95_authority_binding",
        verify,
    )
    return {
        "repo_root": tmp_path,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "expected_binding": expected_binding,
        "calls": calls,
    }


def test_v2119_schema_is_registered_without_replacing_v2118_adapter() -> None:
    registry = authority.DEDICATED_OBSERVED_P95_BINDING_SCHEMA_REGISTRY

    assert registry[continuation.V2119_CURRENT_AUTHORITY_SCHEMA_VERSION] == (
        "v2.11.9-continuation-authority"
    )
    assert (
        registry["finevo-pilot-v2.11.8-continuation-observed-p95-authority-v1"]
        == "v2.11.8-continuation-authority"
    )


def test_v2119_generic_adapter_uses_exact_path_contract_and_verifier(
    v2119_adapter_case: dict[str, Any],
) -> None:
    case = v2119_adapter_case

    binding = authority.verified_observed_p95_authority_binding(
        EXACT_RECEIPT.as_posix(),
        repo_root=case["repo_root"],
        expected_git_commit=EXPECTED_COMMIT,
    )
    rows = authority.verify_observed_p95_authority_receipt(
        EXACT_RECEIPT.as_posix(),
        repo_root=case["repo_root"],
        expected_git_commit=EXPECTED_COMMIT,
    )

    assert binding == case["expected_binding"]
    assert rows == case["expected_binding"]["reservations"]
    assert case["calls"] == [
        {
            "receipt_path": EXACT_RECEIPT.as_posix(),
            "repo_root": case["repo_root"].resolve(),
            "expected_git_commit": EXPECTED_COMMIT,
            "expected_contract_sha256": EXPECTED_CONTRACT_HASH,
        },
        {
            "receipt_path": EXACT_RECEIPT.as_posix(),
            "repo_root": case["repo_root"].resolve(),
            "expected_git_commit": EXPECTED_COMMIT,
            "expected_contract_sha256": EXPECTED_CONTRACT_HASH,
        },
    ]


def test_v2119_mapping_only_and_alias_paths_fail_closed(
    v2119_adapter_case: dict[str, Any],
) -> None:
    case = v2119_adapter_case
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="requires a repository-relative receipt path",
    ):
        authority.verify_observed_p95_authority_receipt(
            case["receipt"],
            repo_root=case["repo_root"],
            expected_git_commit=EXPECTED_COMMIT,
        )

    alias = Path("experiment_results/pilot-v2.11.9/raw/alias/post_gate_authority.json")
    alias_path = case["repo_root"] / alias
    alias_path.parent.mkdir(parents=True)
    alias_path.write_bytes(case["receipt_path"].read_bytes())
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="outside its exact path",
    ):
        authority.verified_observed_p95_authority_binding(
            alias.as_posix(),
            repo_root=case["repo_root"],
            expected_git_commit=EXPECTED_COMMIT,
        )


def test_v2119_missing_frozen_contract_identity_fails_closed(
    v2119_adapter_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = v2119_adapter_case
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256",
        None,
    )

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="frozen contract identity is unavailable",
    ):
        authority.verified_observed_p95_authority_binding(
            EXACT_RECEIPT.as_posix(),
            repo_root=case["repo_root"],
            expected_git_commit=EXPECTED_COMMIT,
        )


def test_v2119_verifier_failure_is_wrapped_fail_closed(
    v2119_adapter_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = v2119_adapter_case

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise continuation.PilotV2119ContinuationError("fixture drift")

    monkeypatch.setattr(
        continuation,
        "verified_v2119_observed_p95_authority_binding",
        reject,
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="failed validation: fixture drift",
    ):
        authority.verified_observed_p95_authority_binding(
            EXACT_RECEIPT.as_posix(),
            repo_root=case["repo_root"],
            expected_git_commit=EXPECTED_COMMIT,
        )


def test_v2119_binding_substitution_fails_closed(
    v2119_adapter_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = v2119_adapter_case

    def substitute(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        binding = deepcopy(case["expected_binding"])
        binding["receipt_file_sha256"] = "d" * 64
        return binding

    monkeypatch.setattr(
        continuation,
        "verified_v2119_observed_p95_authority_binding",
        substitute,
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="bytes or binding changed",
    ):
        authority.verified_observed_p95_authority_binding(
            EXACT_RECEIPT.as_posix(),
            repo_root=case["repo_root"],
            expected_git_commit=EXPECTED_COMMIT,
        )
