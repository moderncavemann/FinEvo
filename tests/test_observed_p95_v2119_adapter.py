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
from verified_memory import pilot_v2119_continuation as continuation
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


EXACT_RECEIPT = Path(
    "experiment_results/pilot-v2.11.9/raw/parent-import/current_authority/"
    "post_gate_authority.json"
)
EXPECTED_COMMIT = "a" * 40
EXPECTED_CONTRACT_HASH = "b" * 64
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


def _reservation(*, call_kind: str) -> dict[str, Any]:
    sample_count = 24 if call_kind == "action" else 8
    return {
        "sample_count": sample_count,
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


@pytest.fixture
def v2119_roundtrip_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Build a provider-free release fixture with the real adapter boundary."""

    source_root = Path(__file__).resolve().parents[1]
    repo_root = tmp_path / "release"
    experiments = repo_root / "experiments"
    experiments.mkdir(parents=True)
    contract_path = experiments / "pilot_v2_11_9.yaml"
    contract_path.write_bytes(
        (source_root / "experiments/pilot_v2_11_9.yaml").read_bytes()
    )
    manifest_path = experiments / "pilot_v2_11_9_source_manifest.json"
    manifest_path.write_bytes(
        (
            source_root
            / "experiments/pilot_v2_11_9_source_manifest.json"
        ).read_bytes()
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
        continuation.V2119_SCIENCE_TAG,
        "-m",
        "fixture science tag",
    )

    contract = pilot_contract_module.load_pilot_contract(contract_path)
    raw_root = repo_root.joinpath(*continuation.V2119_RAW_ROOT.parts)
    receipt_path = repo_root / EXACT_RECEIPT
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    continuation.V2119_CURRENT_AUTHORITY_SCHEMA_VERSION
                ),
                "integrity": {"content_sha256": CONTENT_HASH},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt_file_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

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
        "verify_v2119_current_authority",
        lambda **_kwargs: deepcopy(current_authority),
    )
    monkeypatch.setattr(
        continuation,
        "verified_v2119_projection",
        lambda _contract, model_id, **_kwargs: (
            deepcopy(projections[model_id]),
            repo_root / f"{model_id}-projection.json",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "verified_v2119_projection",
        lambda _contract, model_id, **_kwargs: (
            deepcopy(projections[model_id]),
            repo_root / f"{model_id}-projection.json",
        ),
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
        git_tag=continuation.V2119_SCIENCE_TAG,
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
        "release_commit": release_commit,
        "receipt_file_sha256": receipt_file_sha256,
    }


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


def test_v2119_producer_runner_authority_is_exact_13_to_17_to_13(
    v2119_roundtrip_case: dict[str, Any],
) -> None:
    """The receipt owns 13 source fields; only the runner owns its 4-field envelope."""

    case = v2119_roundtrip_case
    binding = continuation.verified_v2119_observed_p95_authority_binding(
        EXACT_RECEIPT.as_posix(),
        repo_root=case["repo_root"],
        expected_git_commit=case["release_commit"],
        expected_contract_sha256=case["contract"].canonical_hash,
    )
    assert binding["receipt_path"] == EXACT_RECEIPT.as_posix()
    assert binding["receipt_file_sha256"] == case["receipt_file_sha256"]
    assert binding["receipt_content_sha256"] == CONTENT_HASH

    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        runtime = orchestrator._runtime_model_for_profile(
            case["contract"].provider_profiles[model_id]
        )
        runner_rows = continuation.runner_reservations_for_v2119(
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
                name: value
                for name, value in runner_authority.items()
                if name not in RECEIPT_ENVELOPE_FIELDS
            }

            assert set(producer_authority) == SOURCE_AUTHORITY_FIELDS
            assert producer_authority["source_projection_schema_version"] == (
                OBSERVED_P95_PROJECTION_SCHEMA_VERSION
            )
            assert set(runner_authority) == (
                SOURCE_AUTHORITY_FIELDS | RECEIPT_ENVELOPE_FIELDS
            )
            assert stripped == producer_authority
            assert runner_authority["source_authority_receipt_path"] == (
                binding["receipt_path"]
            )
            assert runner_authority["source_authority_receipt_file_sha256"] == (
                binding["receipt_file_sha256"]
            )
            assert (
                runner_authority["source_authority_receipt_content_sha256"]
                == binding["receipt_content_sha256"]
            )
            assert runner_authority["source_release_commit"] == (
                binding["git_commit"]
            )
            assert ObservedPreflightP95Reservation.from_dict(
                model=runtime,
                call_kind=call_kind,
                value=runner_entry,
            ).to_dict() == runner_entry


def test_v2119_gpt52_and_gpt56_scientific_configs_roundtrip_and_validate(
    v2119_roundtrip_case: dict[str, Any],
) -> None:
    """Exercise the same config constructor and validator used before dispatch."""

    case = v2119_roundtrip_case
    model_stages = (
        ("gpt52_main", "experiment-b"),
        ("gpt56_diagnostic", "cross-model"),
    )
    for model_id, stage_id in model_stages:
        spec = next(
            spec
            for spec in case["contract"].expand(stage=stage_id, model=model_id)
            if spec.arm_id == "full"
        )
        runtime = orchestrator._runtime_model_for_profile(
            case["contract"].provider_profiles[model_id]
        )
        reservations = orchestrator._runner_p95_reservations(
            case["contract"],
            model_id,
            raw_root=case["raw_root"],
            paid=case["paid"],
            authority_repo_root=case["repo_root"],
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

        assert restored.to_dict() == payload
        assert set(validated) == {"action", "semantic"}
        assert serialized_has_sealed_observed_p95_authority(
            payload,
            authority_repo_root=case["repo_root"],
        ) is True


def test_v2119_legacy_17_field_producer_fails_source_backed_validation(
    v2119_roundtrip_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reintroducing the V2.11.9 producer bug must stop config construction."""

    case = v2119_roundtrip_case
    binding = continuation.verified_v2119_observed_p95_authority_binding(
        EXACT_RECEIPT.as_posix(),
        repo_root=case["repo_root"],
        expected_git_commit=case["release_commit"],
        expected_contract_sha256=case["contract"].canonical_hash,
    )
    legacy = deepcopy(binding)
    receipt_envelope = {
        "source_authority_receipt_path": binding["receipt_path"],
        "source_authority_receipt_file_sha256": binding["receipt_file_sha256"],
        "source_authority_receipt_content_sha256": binding[
            "receipt_content_sha256"
        ],
        "source_release_commit": binding["git_commit"],
    }
    for by_kind in legacy["reservations"].values():
        for call_kind in ("action", "semantic"):
            by_kind[call_kind]["authority"].update(receipt_envelope)
            assert set(by_kind[call_kind]["authority"]) == (
                SOURCE_AUTHORITY_FIELDS | RECEIPT_ENVELOPE_FIELDS
            )
    monkeypatch.setattr(
        continuation,
        "verified_v2119_observed_p95_authority_binding",
        lambda *_args, **_kwargs: deepcopy(legacy),
    )
    monkeypatch.setattr(
        orchestrator,
        "verified_v2119_observed_p95_authority_binding",
        lambda *_args, **_kwargs: deepcopy(legacy),
    )

    model_id = "gpt52_main"
    spec = next(
        spec
        for spec in case["contract"].expand(
            stage="experiment-b",
            model=model_id,
        )
        if spec.arm_id == "full"
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="runner authority layering differs from the flat source binding",
    ):
        orchestrator._runner_p95_reservations(
            case["contract"],
            model_id,
            raw_root=case["raw_root"],
            paid=case["paid"],
            authority_repo_root=case["repo_root"],
        )
    runtime = orchestrator._runtime_model_for_profile(
        case["contract"].provider_profiles[model_id]
    )
    reservations = {runtime: deepcopy(legacy["reservations"][runtime])}
    with observed_p95_authority_repo_context(case["repo_root"]):
        with pytest.raises(
            ValueError,
            match=(
                "source-backed observed p95 source authority differs for "
                "openai/gpt-5.2-2025-12-11::action"
            ),
        ):
            orchestrator.config_for_spec(
                case["contract"],
                spec,
                raw_root=case["raw_root"],
                paid_provenance=case["paid"],
                authority_repo_root=case["repo_root"],
                verify_bound_inputs=True,
                preflight_p95_reservations=reservations,
            )


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
