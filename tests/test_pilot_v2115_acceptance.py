from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v2115_acceptance as acceptance
from verified_memory import runner
from verified_memory.pilot_budget import RunProjection
from verified_memory.pilot_contract import PilotContract, load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"
V2114_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_4.yaml"


def _paid(contract: PilotContract) -> orchestrator.GitProvenance:
    return orchestrator.GitProvenance(
        git_tag=str(contract.implementation["required_git_tag"]),
        head_commit="c" * 40,
        tag_commit="c" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )


def _reservation(*, call_kind: str) -> dict[str, Any]:
    raw_prompt = 800.0 if call_kind == "action" else 1_200.0
    raw_completion = 200.0 if call_kind == "action" else 400.0
    raw_cost = 0.01 if call_kind == "action" else 0.02
    return {
        "sample_count": 24 if call_kind == "action" else 8,
        "raw_p95": {
            "prompt_tokens": raw_prompt,
            "completion_tokens": raw_completion,
            "total_tokens": raw_prompt + raw_completion,
            "cost_usd": raw_cost,
        },
        "reserved_p95": {
            "prompt_tokens": int(raw_prompt * 1.25),
            "completion_tokens": int(raw_completion * 1.25),
            "total_tokens": int((raw_prompt + raw_completion) * 1.25),
            "cost_usd": raw_cost * 1.25,
        },
        "reserve_multiplier": 1.25,
    }


def _stable_authority(*, model_id: str, served_model: str) -> dict[str, str]:
    return {
        "authority_id": "finevo-closed-loop-observed-p95-v1",
        "source_kind": "sealed-closed-loop-observed-p95",
        "source_projection_schema_version": "finevo-pilot-projection-p95-v1",
        "source_preflight_run_id": f"source-preflight-{model_id}",
        "source_preflight_run_spec_sha256": "1" * 64,
        "source_model_id": model_id,
        "source_served_model": served_model,
        "source_execution_artifact_sha256": "2" * 64,
        "source_provider_call_journal_sha256": "3" * 64,
    }


def _source_generation_authority(
    *,
    model_id: str,
    served_model: str,
) -> dict[str, str]:
    return {
        **_stable_authority(model_id=model_id, served_model=served_model),
        "pilot_contract_hash": "4" * 64,
        "pilot_tag": "pilot-v2.11.4-science",
        "source_projection_file_sha256": "5" * 64,
        "source_projection_content_sha256": "6" * 64,
        "source_authority_receipt_path": (
            "experiment_results/pilot-v2.11.4/raw/long-context-preflight/"
            "post_gate_authority.json"
        ),
        "source_authority_receipt_file_sha256": "7" * 64,
        "source_authority_receipt_content_sha256": "8" * 64,
        "source_release_commit": "9" * 40,
    }


def _current_generation_authority(
    contract: PilotContract,
    *,
    model_id: str,
    served_model: str,
) -> dict[str, str]:
    return {
        **_stable_authority(model_id=model_id, served_model=served_model),
        "pilot_contract_hash": contract.canonical_hash,
        "pilot_tag": str(contract.implementation["required_git_tag"]),
        "source_projection_file_sha256": "a" * 64,
        "source_projection_content_sha256": "b" * 64,
    }


def _runtime_rows(
    authority: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    return {
        call_kind: {
            "authority": dict(authority),
            "reservation": _reservation(call_kind=call_kind),
        }
        for call_kind in ("action", "semantic")
    }


def _v2115_binding_fixtures(
    contract: PilotContract,
    paid: orchestrator.GitProvenance,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    per_model_authority: dict[str, dict[str, Any]] = {}
    per_model_projection: dict[str, dict[str, Any]] = {}
    global_reservations: dict[str, Any] = {}
    for model_id in acceptance.V2115_ALLOWED_MODELS:
        profile = contract.provider_profiles[model_id]
        runtime_model = orchestrator._runtime_model_for_profile(profile)
        source_rows = _runtime_rows(
            _source_generation_authority(
                model_id=model_id,
                served_model=profile.served_model,
            )
        )
        current_rows = _runtime_rows(
            _current_generation_authority(
                contract,
                model_id=model_id,
                served_model=profile.served_model,
            )
        )
        per_model_authority[model_id] = {
            "receipt_path": (
                "experiment_results/pilot-v2.11.5/raw/long-context-preflight/"
                f"imported_observed_p95/{model_id}/"
                "observed_p95_authority_receipt.json"
            ),
            "receipt_file_sha256": "d" * 64,
            "receipt_content_sha256": "e" * 64,
            "git_commit": paid.head_commit,
            "reservations": {runtime_model: source_rows},
        }
        payload = {
            "model_id": model_id,
            "served_model": profile.served_model,
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_commit": paid.head_commit,
            },
            "projection": {
                f"{profile.served_model}::{call_kind}": _reservation(
                    call_kind=call_kind
                )
                for call_kind in ("action", "semantic")
            },
        }
        per_model_projection[model_id] = {
            "profile_id": model_id,
            "served_model": profile.served_model,
            "runtime_model": runtime_model,
            "git_commit": paid.head_commit,
            "source_contract_id": orchestrator.V2112_CONTRACT_ID,
            "reservations": {runtime_model: source_rows},
            "payload": payload,
        }
        global_reservations[runtime_model] = current_rows
    global_binding = {
        "receipt_path": acceptance.V2115_POST_GATE_RELATIVE_PATH.as_posix(),
        "receipt_file_sha256": "a" * 64,
        "receipt_content_sha256": "b" * 64,
        "git_commit": paid.head_commit,
        "reservations": global_reservations,
    }
    return per_model_authority, per_model_projection, global_binding


def _model_from_import_path(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.parent.name in {"gpt52_main", "gpt56_diagnostic"}:
        return candidate.parent.name
    raise AssertionError(f"unexpected V2.11.5 model path: {path}")


def test_v2115_denominator_is_exact_and_draft_fails_closed() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    stage_counts = {
        stage_id: len(contract.expand(stage=stage_id))
        for stage_id in contract.stage_ids
    }
    scientific = tuple(
        spec
        for stage_id in acceptance.V2115_SCIENTIFIC_STAGE_IDS
        for spec in contract.expand(stage=stage_id)
    )
    provider = tuple(
        spec
        for spec in scientific
        if spec.execution_mode != "offline_candidate_admission"
    )

    assert stage_counts == acceptance.V2115_EXPECTED_STAGE_CELL_COUNTS
    assert len(contract.expand()) == 136
    assert len(scientific) == 131
    assert len(provider) == 126
    assert sum(stage_counts[stage] for stage in acceptance.V2115_OPERATIONAL_STAGE_IDS) == 5
    if contract.status == "frozen":
        acceptance._require_contract(contract, ROOT)
    else:
        with pytest.raises(
            acceptance.PilotV2115AcceptanceError,
            match="exact frozen V2.11.5 contract denominator",
        ):
            acceptance._require_contract(contract, ROOT)


def test_v2115_contract_shape_passes_acceptance_when_release_status_is_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = replace(load_pilot_contract(CONTRACT_PATH), status="frozen")
    monkeypatch.setattr(acceptance, "load_pilot_contract", lambda _path: contract)

    acceptance._require_contract(contract, ROOT)


def test_v2115_cross_generation_comparator_has_exact_closed_allowlists() -> None:
    source_fields = {
        "pilot_contract_hash",
        "pilot_tag",
        "source_projection_file_sha256",
        "source_projection_content_sha256",
        "source_authority_receipt_path",
        "source_authority_receipt_file_sha256",
        "source_authority_receipt_content_sha256",
        "source_release_commit",
    }
    current_fields = {
        "pilot_contract_hash",
        "pilot_tag",
        "source_projection_file_sha256",
        "source_projection_content_sha256",
    }
    stable_fields = set(_stable_authority(model_id="gpt52_main", served_model="m"))

    assert set(orchestrator._V2115_SOURCE_GENERATION_AUTHORITY_FIELDS) == source_fields
    assert set(orchestrator._V2115_CURRENT_GENERATION_AUTHORITY_FIELDS) == current_fields
    assert set(orchestrator._V2115_STABLE_CROSS_GENERATION_AUTHORITY_FIELDS) == stable_fields


def test_v2115_cross_generation_comparator_accepts_only_registered_generation_drift() -> None:
    contract = SimpleNamespace(
        canonical_hash="a" * 64,
        implementation={"required_git_tag": "pilot-v2.11.5-science"},
    )
    source = _runtime_rows(
        _source_generation_authority(
            model_id="gpt52_main",
            served_model="gpt-5.2-2025-12-11",
        )
    )
    current = _runtime_rows(
        _current_generation_authority(
            contract,
            model_id="gpt52_main",
            served_model="gpt-5.2-2025-12-11",
        )
    )

    assert orchestrator._v2115_cross_generation_reservations_match(
        source_runtime_reservations=source,
        current_runtime_reservations=current,
    )

    for field in orchestrator._V2115_SOURCE_GENERATION_AUTHORITY_FIELDS:
        changed = deepcopy(source)
        changed["action"]["authority"][field] = "registered-generation-drift"
        assert orchestrator._v2115_cross_generation_reservations_match(
            source_runtime_reservations=changed,
            current_runtime_reservations=current,
        ), field

    outside_allowlist = deepcopy(source)
    outside_allowlist["action"]["authority"]["unregistered_generation_field"] = True
    assert not orchestrator._v2115_cross_generation_reservations_match(
        source_runtime_reservations=outside_allowlist,
        current_runtime_reservations=current,
    )

    stable_drift = deepcopy(source)
    stable_drift["semantic"]["authority"]["source_model_id"] = "wrong-model"
    assert not orchestrator._v2115_cross_generation_reservations_match(
        source_runtime_reservations=stable_drift,
        current_runtime_reservations=current,
    )

    numeric_drift = deepcopy(current)
    numeric_drift["action"]["reservation"]["sample_count"] += 1
    assert not orchestrator._v2115_cross_generation_reservations_match(
        source_runtime_reservations=source,
        current_runtime_reservations=numeric_drift,
    )


def test_v2114_whole_object_failure_remains_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V2114_CONTRACT_PATH)
    paid = _paid(contract)
    model_id = "gpt52_main"
    profile = contract.provider_profiles[model_id]
    runtime_model = orchestrator._runtime_model_for_profile(profile)
    source_rows = _runtime_rows(
        _source_generation_authority(
            model_id=model_id,
            served_model=profile.served_model,
        )
    )
    current_rows = _runtime_rows(
        {
            **_stable_authority(
                model_id=model_id,
                served_model=profile.served_model,
            ),
            "pilot_contract_hash": contract.canonical_hash,
            "pilot_tag": str(contract.implementation["required_git_tag"]),
            "source_projection_file_sha256": "a" * 64,
            "source_projection_content_sha256": "b" * 64,
        }
    )
    authority = {
        "receipt_path": "fixture-source.json",
        "receipt_file_sha256": "d" * 64,
        "receipt_content_sha256": "e" * 64,
        "git_commit": paid.head_commit,
        "reservations": {runtime_model: source_rows},
    }
    payload = {
        "model_id": model_id,
        "served_model": profile.served_model,
        "bindings": {
            "contract_sha256": contract.canonical_hash,
            "git_commit": paid.head_commit,
        },
        "projection": {
            f"{profile.served_model}::{kind}": _reservation(call_kind=kind)
            for kind in ("action", "semantic")
        },
    }
    projection = {
        "profile_id": model_id,
        "served_model": profile.served_model,
        "runtime_model": runtime_model,
        "git_commit": paid.head_commit,
        "source_contract_id": orchestrator.V2112_CONTRACT_ID,
        "reservations": {runtime_model: source_rows},
        "payload": payload,
    }
    global_binding = {"reservations": {runtime_model: current_rows}}
    monkeypatch.setattr(
        orchestrator,
        "verified_v2114_observed_p95_authority_binding",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(
        orchestrator,
        "verified_v2114_observed_p95_projection_binding",
        lambda *_args, **_kwargs: projection,
    )
    monkeypatch.setattr(
        orchestrator,
        "_verified_observed_p95_binding",
        lambda *_args, **_kwargs: global_binding,
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="V2.11.4 resealed gpt52_main p95 identity/binding drifted",
    ):
        orchestrator._load_verified_projection(
            contract,
            model_id,
            raw_root=tmp_path,
            paid=paid,
            authority_repo_root=ROOT,
        )


def test_v2115_real_config_audit_covers_126_cells_and_five_d_groups_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real config construction and the real reservation comparator.

    Only the already-verified receipt/file adapters and utility artifact reader
    are replaced with deterministic fixtures.  In particular, neither
    ``_runner_p95_reservations`` nor
    ``_v2115_cross_generation_reservations_match`` is mocked.
    """

    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    authorities, projections, global_binding = _v2115_binding_fixtures(
        contract, paid
    )
    raw_root = tmp_path / "experiment_results" / "pilot-v2.11.5" / "raw"
    raw_root.mkdir(parents=True)

    monkeypatch.setattr(
        orchestrator,
        "verified_v2115_observed_p95_authority_binding",
        lambda path, **_kwargs: authorities[_model_from_import_path(path)],
    )
    monkeypatch.setattr(
        orchestrator,
        "verified_v2115_observed_p95_projection_binding",
        lambda path, **_kwargs: projections[_model_from_import_path(path)],
    )
    monkeypatch.setattr(
        orchestrator,
        "_verified_observed_p95_binding",
        lambda *_args, **_kwargs: global_binding,
    )
    monkeypatch.setattr(
        orchestrator,
        "resolve_utility",
        lambda *_args, **_kwargs: orchestrator._utility_from_mapping(
            {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 1.0,
                "consumption_scale": 1.0,
                "discount_factor": 0.99,
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "_verify_source_backed_observed_p95_rows",
        lambda *_args, **_kwargs: None,
    )

    with acceptance._provider_boundary_stack():
        configs, d_groups = acceptance._audit_configs_and_d_groups(
            contract,
            repo_root=ROOT,
            raw_root=raw_root,
            paid=paid,
        )

    assert configs["provider_config_count"] == 126
    assert len(configs["config_sha256_by_run"]) == 126
    assert configs["roundtrip_exact"] is True
    assert configs["sealed_observed_p95_authority"] is True
    assert d_groups["group_count"] == 5
    assert d_groups["cells_per_group"] == 11
    assert len(d_groups["groups"]) == 5


def test_v2115_projection_audit_covers_81_units_and_5816_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    def normal_projection(
        _contract: Any,
        spec: Any,
        **_kwargs: Any,
    ) -> RunProjection:
        calls = {"action": spec.num_agents * spec.episode_length}
        if contract.arms[spec.arm_id]["parameters"].get(
            "semantic_actor_exposure", True
        ):
            calls["semantic"] = 16
        total = sum(calls.values())
        return RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=total / 1_000.0,
            completions=total,
            storage_bytes=20_000_000,
            basis={"calls_by_kind": calls},
        )

    def d_projection(
        _contract: Any,
        representative: Any,
        **_kwargs: Any,
    ) -> RunProjection:
        calls = {"action": 288, "semantic": 8}
        return RunProjection(
            run_id=f"d-group-{representative.environment_seed}",
            stage_bucket=representative.budget_bucket,
            cost_usd=sum(calls.values()) / 1_000.0,
            completions=sum(calls.values()),
            storage_bytes=80_000_000,
            basis={"calls_by_kind": calls},
        )

    operational_runs: dict[str, Any] = {}
    for stage_id in acceptance.V2115_OPERATIONAL_STAGE_IDS:
        for spec in contract.expand(stage=stage_id):
            operational_runs[spec.run_id] = {
                "stage_bucket": spec.budget_bucket,
                "status": "complete",
                "actual": {
                    "cost_usd": 0.0,
                    "completions": 0,
                    "storage_bytes": 0,
                },
            }
    boundary = contract.v2115_forward_boundary
    assert boundary is not None
    caps = orchestrator._budget_caps(contract)
    budget = SimpleNamespace(
        caps=caps,
        snapshot=lambda: {
            "parent_debit": boundary["parent_budget_debit"],
            "caps": caps.to_dict(),
            "runs": operational_runs,
        },
    )
    monkeypatch.setattr(orchestrator, "projection_from_preflight", normal_projection)
    monkeypatch.setattr(orchestrator, "_d_group_projection", d_projection)
    monkeypatch.setattr(
        orchestrator,
        "_assert_projection_matrix_fits",
        lambda *_args, **_kwargs: None,
    )

    with acceptance._provider_boundary_stack():
        result = acceptance._audit_projections(
            contract,
            repo_root=ROOT,
            raw_root=ROOT / "experiment_results" / "pilot-v2.11.5" / "raw",
            paid=SimpleNamespace(),
            run_ledger=SimpleNamespace(),
            budget_ledger=budget,
        )

    assert result["projection_unit_count"] == 81
    assert result["fresh_calls_by_kind"] == {"action": 4_848, "semantic": 968}
    assert result["fresh_provider_calls"] == 5_816
    assert result["fresh_projected_completions"] == 5_816
    assert result["fresh_projected_storage_bytes"] == 1_830_000_000
    assert result["fresh_calls_by_model"] == boundary["matrix"]["fresh_calls_by_model"]


def test_v2115_zero_provider_boundary_blocks_all_canonical_factories() -> None:
    provider_factory = acceptance.canonical_llm_providers.create_llm_provider
    multi_model = acceptance.canonical_llm_providers.MultiModelLLM
    catalog_validator = (
        acceptance.canonical_provider_catalog.validate_live_provider_catalog
    )

    with acceptance._provider_boundary_stack():
        for call in (
            lambda: acceptance.canonical_llm_providers.create_llm_provider(),
            lambda: acceptance.canonical_llm_providers.MultiModelLLM(),
            lambda: acceptance.canonical_provider_catalog.validate_live_provider_catalog(),
        ):
            with pytest.raises(
                acceptance.PilotV2115AcceptanceError,
                match="zero-provider acceptance attempted",
            ):
                call()

    assert acceptance.canonical_llm_providers.create_llm_provider is provider_factory
    assert acceptance.canonical_llm_providers.MultiModelLLM is multi_model
    assert (
        acceptance.canonical_provider_catalog.validate_live_provider_catalog
        is catalog_validator
    )


def test_v2115_namespace_rejects_v2114_scientific_reuse(tmp_path: Path) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    reused = (
        raw_root
        / "unexpected"
        / (
            "finevo-pilot-v2.11.4--experiment-c--gpt52_main--full--"
            "registered-rate-shock--stage0-selected--s1099057501"
        )
        / "decoded.json"
    )
    reused.parent.mkdir(parents=True)
    reused.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        acceptance.PilotV2115AcceptanceError,
        match="pre-science raw namespace contains scientific artifacts",
    ):
        acceptance._audit_pre_science_namespace(raw_root, contract)
