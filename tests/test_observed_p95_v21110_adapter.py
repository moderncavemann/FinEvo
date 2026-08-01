from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_contract as pilot_contract_module
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v21110_continuation as continuation
from verified_memory.pilot_checkpoint import config_from_dict
from verified_memory.runner import (
    OBSERVED_P95_AUTHORITY_ID,
    OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
    OBSERVED_P95_SOURCE_KIND,
    ObservedPreflightP95Reservation,
    has_sealed_observed_p95_authority,
    observed_p95_authority_repo_context,
    serialized_has_sealed_observed_p95_authority,
    validate_preflight_p95_reservations,
)


ROOT = Path(__file__).resolve().parents[1]
EXACT_RECEIPT = Path(
    "experiment_results/pilot-v2.11.10/raw/parent-import/current_authority/"
    "post_gate_authority.json"
)
CONTENT_HASH = "c" * 64
SOURCE_AUTHORITY_FIELDS = {
    "authority_id",
    "source_kind",
    "pilot_contract_hash",
    "pilot_tag",
    "source_projection_schema_version",
    "source_projection_file_sha256",
    "source_projection_content_sha256",
    "source_preflight_run_id",
    "source_preflight_run_spec_sha256",
    "source_model_id",
    "source_served_model",
    "source_execution_artifact_sha256",
    "source_provider_call_journal_sha256",
}
RECEIPT_ENVELOPE_FIELDS = {
    "source_authority_receipt_path",
    "source_authority_receipt_file_sha256",
    "source_authority_receipt_content_sha256",
    "source_release_commit",
}


def _reservation(*, call_kind: str) -> dict[str, Any]:
    return {
        "sample_count": 24 if call_kind == "action" else 8,
        "raw_p95": {
            "prompt_tokens": 80.0,
            "completion_tokens": 40.0,
            "total_tokens": 120.0,
            "cost_usd": 0.008,
        },
        "reserved_p95": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.01,
        },
        "reserve_multiplier": 1.25,
    }


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("/usr/bin/git", *arguments),
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def v21110_roundtrip_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    repo_root = tmp_path / "release"
    contract_path = repo_root / "experiments/pilot_v2_11_10.yaml"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes((ROOT / "experiments/pilot_v2_11_10.yaml").read_bytes())
    contract_path.with_name("pilot_v2_11_10_source_manifest.json").write_bytes(
        (ROOT / "experiments/pilot_v2_11_10_source_manifest.json").read_bytes()
    )
    (repo_root / ".gitignore").write_text(
        "/experiment_results/\n",
        encoding="utf-8",
    )
    _git(repo_root, "init", "--quiet")
    _git(repo_root, "config", "user.name", "FinEvo Test")
    _git(repo_root, "config", "user.email", "finevo-test@example.invalid")
    _git(repo_root, "add", ".gitignore", "experiments")
    _git(repo_root, "commit", "--quiet", "-m", "fixture release")
    release_commit = _git(repo_root, "rev-parse", "HEAD")
    _git(
        repo_root,
        "tag",
        "-a",
        continuation.V21110_SCIENCE_TAG,
        "-m",
        "fixture science tag",
    )
    contract_document = json.loads(contract_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
        pilot_contract_module.science_design_sha256(contract_document),
    )
    contract = pilot_contract_module.load_pilot_contract(contract_path)
    raw_root = repo_root.joinpath(*continuation.V21110_RAW_ROOT.parts)
    receipt_path = repo_root / EXACT_RECEIPT
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": continuation.V21110_CURRENT_AUTHORITY_SCHEMA_VERSION,
                "integrity": {"content_sha256": CONTENT_HASH},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    reservations: dict[str, dict[str, Any]] = {}
    stable_source_authorities: dict[str, dict[str, Any]] = {}
    projections: dict[str, dict[str, Any]] = {}
    for model_index, model_id in enumerate(
        ("gpt52_main", "gpt56_diagnostic"),
        start=1,
    ):
        profile = contract.provider_profiles[model_id]
        runtime = orchestrator._runtime_model_for_profile(profile)
        reservations[runtime] = {}
        stable_source_authorities[runtime] = {}
        projection_rows: dict[str, Any] = {}
        for call_kind in ("action", "semantic"):
            numeric = _reservation(call_kind=call_kind)
            stable = {
                "authority_id": OBSERVED_P95_AUTHORITY_ID,
                "source_kind": OBSERVED_P95_SOURCE_KIND,
                "source_preflight_run_id": (
                    f"fixture-{model_id}-closed-loop-preflight"
                ),
                "source_preflight_run_spec_sha256": str(model_index) * 64,
                "source_model_id": model_id,
                "source_served_model": profile.served_model,
                "source_execution_artifact_sha256": str(model_index + 2) * 64,
                "source_provider_call_journal_sha256": str(model_index + 4) * 64,
            }
            reservations[runtime][call_kind] = numeric
            stable_source_authorities[runtime][call_kind] = stable
            projection_rows[f"{profile.served_model}::{call_kind}"] = numeric
        projections[model_id] = {
            "profile_id": model_id,
            "served_model": profile.served_model,
            "runtime_model": runtime,
            "projection": projection_rows,
        }

    current_authority = {
        "integrity": {"content_sha256": CONTENT_HASH},
        "reservations": reservations,
        "stable_source_authorities": stable_source_authorities,
    }
    monkeypatch.setattr(
        continuation,
        "verify_v21110_current_authority",
        lambda **_kwargs: deepcopy(current_authority),
    )
    monkeypatch.setattr(
        continuation,
        "verified_v21110_projection",
        lambda _contract, model_id, **_kwargs: (
            deepcopy(projections[model_id]),
            repo_root / f"{model_id}-projection.json",
        ),
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256",
        contract.canonical_hash,
    )
    monkeypatch.setattr(
        orchestrator,
        "resolve_utility",
        lambda *_args, **_kwargs: orchestrator._utility_from_mapping(
            {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 0.5,
                "consumption_scale": 1.0,
                "discount_factor": 0.99,
            }
        ),
    )
    paid = SimpleNamespace(
        git_tag=continuation.V21110_SCIENCE_TAG,
        head_commit=release_commit,
        tag_commit=release_commit,
        tag_object_type="tag",
        worktree_clean=True,
    )
    return {
        "repo_root": repo_root,
        "raw_root": raw_root,
        "contract": contract,
        "paid": paid,
        "receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }


def test_v21110_registry_has_dedicated_adapter() -> None:
    assert authority.DEDICATED_OBSERVED_P95_BINDING_SCHEMA_REGISTRY[
        continuation.V21110_CURRENT_AUTHORITY_SCHEMA_VERSION
    ] == "v2.11.10-continuation-authority"


def test_v21110_producer_core13_becomes_runner_envelope17(
    v21110_roundtrip_case: dict[str, Any],
) -> None:
    case = v21110_roundtrip_case
    binding = continuation.verified_v21110_observed_p95_authority_binding(
        EXACT_RECEIPT.as_posix(),
        repo_root=case["repo_root"],
        expected_git_commit=case["paid"].head_commit,
        expected_contract_sha256=case["contract"].canonical_hash,
    )
    assert binding["receipt_file_sha256"] == case["receipt_file_sha256"]
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        runtime = orchestrator._runtime_model_for_profile(
            case["contract"].provider_profiles[model_id]
        )
        runner_rows = continuation.runner_reservations_for_v21110(
            case["contract"],
            model_id,
            repo_root=case["repo_root"],
            raw_root=case["raw_root"],
            paid=case["paid"],
        )
        assert set(runner_rows) == {runtime}
        for call_kind in ("action", "semantic"):
            producer_authority = binding["reservations"][runtime][call_kind][
                "authority"
            ]
            runner_entry = runner_rows[runtime][call_kind]
            runner_authority = runner_entry["authority"]
            stripped = {
                key: value
                for key, value in runner_authority.items()
                if key not in RECEIPT_ENVELOPE_FIELDS
            }
            assert set(producer_authority) == SOURCE_AUTHORITY_FIELDS
            assert len(producer_authority) == 13
            assert producer_authority["source_projection_schema_version"] == (
                OBSERVED_P95_PROJECTION_SCHEMA_VERSION
            )
            assert set(runner_authority) == (
                SOURCE_AUTHORITY_FIELDS | RECEIPT_ENVELOPE_FIELDS
            )
            assert len(runner_authority) == 17
            assert stripped == producer_authority
            assert ObservedPreflightP95Reservation.from_dict(
                model=runtime,
                call_kind=call_kind,
                value=runner_entry,
            ).to_dict() == runner_entry


def test_v21110_two_models_use_real_config_roundtrip_and_source_validation(
    v21110_roundtrip_case: dict[str, Any],
) -> None:
    case = v21110_roundtrip_case
    for model_id, stage_id in (
        ("gpt52_main", "experiment-b"),
        ("gpt56_diagnostic", "cross-model"),
    ):
        spec = next(
            spec
            for spec in case["contract"].expand(stage=stage_id, model=model_id)
            if spec.arm_id == "full"
        )
        runtime = orchestrator._runtime_model_for_profile(
            case["contract"].provider_profiles[model_id]
        )
        reservations = continuation.runner_reservations_for_v21110(
            case["contract"],
            model_id,
            repo_root=case["repo_root"],
            raw_root=case["raw_root"],
            paid=case["paid"],
        )
        with observed_p95_authority_repo_context(case["repo_root"]):
            config = orchestrator.config_for_spec(
                case["contract"],
                spec,
                raw_root=case["raw_root"],
                paid_provenance=case["paid"],
                authority_repo_root=case["repo_root"],
                verify_bound_inputs=True,
                preflight_p95_reservations=reservations,
            )
            payload = config.to_dict()
            restored = config_from_dict(payload)
            validated = validate_preflight_p95_reservations(
                restored,
                provider_model_name=runtime,
            )
            assert has_sealed_observed_p95_authority(config) is True
            assert has_sealed_observed_p95_authority(restored) is True
            assert serialized_has_sealed_observed_p95_authority(
                payload,
                authority_repo_root=case["repo_root"],
            ) is True
        assert restored.to_dict() == payload
        assert set(validated) == {"action", "semantic"}


def test_v21110_acceptance_runner_material_covers_36_units_and_86_cells(
    v21110_roundtrip_case: dict[str, Any],
) -> None:
    case = v21110_roundtrip_case
    material = continuation._validated_runner_config_material(
        case["contract"],
        repo_root=case["repo_root"],
        raw_root=case["raw_root"],
        paid=case["paid"],
    )
    assert material["cell_count"] == 86
    assert material["execution_unit_count"] == 36
    assert material["model_count"] == 2
    assert material["serialize_restore_validated"] is True
    assert material["source_backed_validation_performed"] is True
    assert material["provider_construction"] is False
    assert material["provider_calls"] == 0
    assert sum(
        len(run_ids) for run_ids in material["execution_unit_run_ids"].values()
    ) == 86
    assert sum(
        len(run_ids) == 11
        for run_ids in material["execution_unit_run_ids"].values()
    ) == 5
