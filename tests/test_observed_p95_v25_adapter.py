from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
from typing import Any

import pytest

from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v25_parent_import as parent_import
from verified_memory.m0_utility import UtilityConfig
from verified_memory.pilot_checkpoint import config_from_dict
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.runner import (
    OBSERVED_P95_AUTHORITY_ID,
    OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
    OBSERVED_P95_SOURCE_KIND,
    serialized_has_sealed_observed_p95_authority,
    validate_preflight_p95_reservations,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_5.yaml"
EXPECTED_COMMIT = "a" * 40
RUNTIME_MODEL = "ollama/llama3.3:70b-instruct-q4_K_M"
SERVED_MODEL = "llama3.3:70b-instruct-q4_K_M"


def _reservation(
    *,
    prompt_tokens: float,
    completion_tokens: float,
    sample_count: int,
) -> dict[str, Any]:
    reserved_prompt = math.ceil(prompt_tokens * 1.25)
    reserved_completion = math.ceil(completion_tokens * 1.25)
    return {
        "sample_count": sample_count,
        "raw_p95": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": 0.0,
        },
        "reserved_p95": {
            "prompt_tokens": reserved_prompt,
            "completion_tokens": reserved_completion,
            "total_tokens": reserved_prompt + reserved_completion,
            "cost_usd": 0.0,
        },
        "reserve_multiplier": 1.25,
    }


@pytest.fixture
def v25_child_receipt() -> dict[str, Any]:
    contract = load_pilot_contract(CONTRACT_PATH)
    source_authority = {
        "authority_id": OBSERVED_P95_AUTHORITY_ID,
        "source_kind": OBSERVED_P95_SOURCE_KIND,
        "pilot_contract_hash": contract.canonical_hash,
        "pilot_tag": parent_import.V25_SCIENCE_TAG,
        "source_projection_schema_version": (
            OBSERVED_P95_PROJECTION_SCHEMA_VERSION
        ),
        "source_projection_file_sha256": "1" * 64,
        "source_projection_content_sha256": "2" * 64,
        "source_preflight_run_id": "fixture-v2.3-preflight",
        "source_preflight_run_spec_sha256": "3" * 64,
        "source_model_id": "llama33_local_controlled",
        "source_served_model": SERVED_MODEL,
        "source_execution_artifact_sha256": "4" * 64,
        "source_provider_call_journal_sha256": "5" * 64,
    }
    reservations = {
        RUNTIME_MODEL: {
            "action": {
                "authority": deepcopy(source_authority),
                "reservation": _reservation(
                    prompt_tokens=907.0,
                    completion_tokens=33.0,
                    sample_count=36,
                ),
            },
            "semantic": {
                "authority": deepcopy(source_authority),
                "reservation": _reservation(
                    prompt_tokens=2499.0,
                    completion_tokens=345.0,
                    sample_count=10,
                ),
            },
        }
    }
    receipt = parent_import._seal(
        {
            "schema_version": (
                parent_import.V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION
            ),
            "contract": {
                "path": "experiments/pilot_v2_5.yaml",
                "file_sha256": "6" * 64,
                "contract_id": contract.contract_id,
                "contract_sha256": contract.canonical_hash,
            },
            "git": {
                "tag": parent_import.V25_SCIENCE_TAG,
                "commit": EXPECTED_COMMIT,
            },
            "model": {
                "model_id": "llama33_local_controlled",
                "runtime_model": RUNTIME_MODEL,
                "served_model": SERVED_MODEL,
            },
            "retry_source": {"fixture_sha256": "7" * 64},
            "parent_source": {"fixture_sha256": "8" * 64},
            "reservations": reservations,
            "scientific_evidence": False,
            "evidence_use": "synthetic inherited-authority adapter fixture",
        }
    )

    ignored_root = ROOT / "experiment_results"
    ignored_root.mkdir(parents=True, exist_ok=True)
    raw_root = Path(
        tempfile.mkdtemp(
            prefix="pytest-v25-observed-adapter-",
            dir=ignored_root,
        )
    )
    receipt_path = raw_root / "observed_p95_authority_receipt.json"
    receipt_path.write_bytes(
        (
            json.dumps(
                receipt,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    try:
        yield {
            "contract": contract,
            "raw_root": raw_root,
            "receipt": receipt,
            "receipt_path": receipt_path,
            "receipt_relative": receipt_path.relative_to(ROOT).as_posix(),
        }
    finally:
        shutil.rmtree(raw_root, ignore_errors=True)


def test_v25_child_receipt_reaches_generic_binding_and_runner_config_before_provider(
    v25_child_receipt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = v25_child_receipt
    adapter_calls: list[dict[str, Any]] = []

    def verify_child(
        receipt: dict[str, Any],
        *,
        repo_root: Path,
        expected_git_commit: str,
    ) -> dict[str, Any]:
        adapter_calls.append(
            {
                "receipt": deepcopy(receipt),
                "repo_root": repo_root,
                "expected_git_commit": expected_git_commit,
            }
        )
        assert receipt == case["receipt"]
        assert repo_root == ROOT
        assert expected_git_commit == EXPECTED_COMMIT
        return deepcopy(receipt["reservations"])

    monkeypatch.setattr(
        parent_import,
        "verify_v25_inherited_p95_receipt",
        verify_child,
    )
    binding = authority.verified_observed_p95_authority_binding(
        case["receipt_relative"],
        repo_root=ROOT,
        expected_git_commit=EXPECTED_COMMIT,
    )
    assert binding["reservations"] == case["receipt"]["reservations"]
    assert binding["receipt_path"] == case["receipt_relative"]
    assert binding["receipt_content_sha256"] == case["receipt"]["integrity"][
        "content_sha256"
    ]

    runner_reservations = deepcopy(binding["reservations"])
    for by_kind in runner_reservations.values():
        for entry in by_kind.values():
            entry["authority"].update(
                {
                    "source_authority_receipt_path": binding["receipt_path"],
                    "source_authority_receipt_file_sha256": binding[
                        "receipt_file_sha256"
                    ],
                    "source_authority_receipt_content_sha256": binding[
                        "receipt_content_sha256"
                    ],
                    "source_release_commit": binding["git_commit"],
                }
            )

    release_checks: list[dict[str, Any]] = []

    def accept_release(**kwargs: Any) -> None:
        release_checks.append(dict(kwargs))
        assert kwargs == {
            "repo_root": ROOT,
            "pilot_tag": parent_import.V25_SCIENCE_TAG,
            "source_release_commit": EXPECTED_COMMIT,
        }

    provider_constructions: list[str] = []

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        provider_constructions.append("forbidden")
        raise AssertionError("provider construction is outside this config gate")

    monkeypatch.setattr(
        "verified_memory.runner._verify_local_release_identity",
        accept_release,
    )
    monkeypatch.setattr(
        orchestrator,
        "resolve_utility",
        lambda *_args, **_kwargs: UtilityConfig(),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        forbidden_provider,
    )

    spec = case["contract"].expand(stage="stage0-calibration")[0]
    config = orchestrator.config_for_spec(
        case["contract"],
        spec,
        raw_root=case["raw_root"],
        paid_provenance=SimpleNamespace(
            git_tag=parent_import.V25_SCIENCE_TAG,
            head_commit=EXPECTED_COMMIT,
        ),
        verify_bound_inputs=True,
        preflight_p95_reservations=runner_reservations,
    )
    validated = validate_preflight_p95_reservations(
        config,
        provider_model_name=RUNTIME_MODEL,
    )
    assert {
        row.call_kind for row in config.preflight_p95_reservations
    } == {"action", "semantic"}
    # The frozen Stage-0 no-memory arm dispatches actions only; both inherited
    # rows are nevertheless source-verified while the config is constructed.
    assert set(validated) == {"action"}

    payload = config.to_dict()
    restored = config_from_dict(payload)
    assert restored.to_dict() == payload
    assert serialized_has_sealed_observed_p95_authority(payload) is True
    assert provider_constructions == []
    assert release_checks
    assert len(adapter_calls) >= 5


def test_v25_generic_adapter_translates_child_validation_failure(
    v25_child_receipt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_child(*_args: Any, **_kwargs: Any) -> None:
        raise parent_import.PilotV25ParentImportError(
            "synthetic child receipt tamper"
        )

    monkeypatch.setattr(
        parent_import,
        "verify_v25_inherited_p95_receipt",
        reject_child,
    )

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match=(
            "V2.5 inherited observed-p95 receipt failed validation: "
            "synthetic child receipt tamper"
        ),
    ):
        authority.verified_observed_p95_authority_binding(
            v25_child_receipt["receipt_relative"],
            repo_root=ROOT,
            expected_git_commit=EXPECTED_COMMIT,
        )
