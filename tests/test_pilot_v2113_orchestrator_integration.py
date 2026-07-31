from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import runner
from verified_memory.pilot_budget import ParentBudgetDebit
from verified_memory.pilot_contract import PilotContract, load_pilot_contract
from verified_memory.pilot_v2113_parent_import import (
    V2113_CAPABILITY_WRAPPER_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_3.yaml"
MODELS = {"gpt52_main", "gpt56_diagnostic"}


def _contract():
    return load_pilot_contract(CONTRACT_PATH)


def _paid() -> orchestrator.GitProvenance:
    return orchestrator.GitProvenance(
        git_tag="pilot-v2.11.3-science",
        head_commit="c" * 40,
        tag_commit="c" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )


def _surrogate_provenance(
    contract: PilotContract,
    git_commit: str,
    git_tag: str,
) -> dict[str, Any]:
    return {
        "git_tag": git_tag,
        "resolved_git_commit": git_commit,
        "commit_resolution": contract.implementation["commit_resolution"],
        "p0_base_commit": contract.implementation["p0_base_commit"],
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def _forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("V2.11.3 operational import constructed a provider")


def _ledgers(
    contract: PilotContract,
    raw_root: Path,
) -> tuple[orchestrator.PilotRunLedger, orchestrator.PilotBudgetLedger]:
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    boundary = contract.v2113_forward_boundary
    assert boundary is not None
    budget = orchestrator.PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=ParentBudgetDebit.from_dict(boundary["parent_budget_debit"]),
    )
    return ledger, budget


def _science_specs(contract: PilotContract) -> tuple[Any, ...]:
    return tuple(
        spec
        for stage_id in orchestrator._scientific_stage_ids(contract)
        for spec in contract.expand(stage=stage_id)
    )


def test_v2113_registers_fresh_136_cell_itt_and_131_science_cells() -> None:
    contract = _contract()
    all_specs = tuple(contract.expand())
    operational = tuple(
        spec
        for spec in all_specs
        if spec.execution_mode
        in {
            "parent_authority_import",
            "capability_authority_import",
            "preflight_authority_import",
        }
    )
    science = tuple(spec for spec in all_specs if spec not in operational)

    assert len(all_specs) == 136
    assert len({spec.run_id for spec in all_specs}) == 136
    assert len(operational) == 5
    assert len(science) == 131
    assert (
        sum(spec.execution_mode == "offline_candidate_admission" for spec in science)
        == 5
    )
    assert (
        sum(spec.execution_mode != "offline_candidate_admission" for spec in science)
        == 126
    )
    assert tuple(orchestrator._scientific_stage_ids(contract)) == (
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
        "cross-model",
    )


def test_v2113_imported_preflight_is_projection_source_not_paid_preflight() -> None:
    contract = _contract()

    for model_id in MODELS:
        assert orchestrator._preflight_stage_for_model(contract, model_id) == (
            "long-context-preflight"
        )
        specs = contract.expand(stage="long-context-preflight", model=model_id)
        assert len(specs) == 1
        assert specs[0].execution_mode == "preflight_authority_import"
        assert specs[0].execution_mode != "closed_loop_preflight"

    assert orchestrator._stage_execution_modes(
        contract, "long-context-preflight"
    ) == frozenset({"preflight_authority_import"})
    assert orchestrator._is_capability_stage(contract, "long-context-preflight")

    operational = tuple(
        spec
        for spec in contract.expand()
        if spec.execution_mode
        in {
            "parent_authority_import",
            "capability_authority_import",
            "preflight_authority_import",
        }
    )
    assert len(operational) == 5
    assert all(
        orchestrator._max_call_projection(contract, spec) == (0, 0, 0)
        for spec in operational
    )


def test_v2113_capability_import_materializes_two_zero_provider_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    paid = _paid()
    raw_root = tmp_path / "raw"
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    boundary = contract.v2113_forward_boundary
    assert boundary is not None
    budget = orchestrator.PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=ParentBudgetDebit.from_dict(boundary["parent_budget_debit"]),
    )
    wrappers = {
        model_id: {
            "schema_version": V2113_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            "capability": {
                "capability_pass": True,
                "interface_pass": True,
            },
            "integrity": {"content_sha256": model_id * 2},
        }
        for model_id in MODELS
    }
    monkeypatch.setattr(orchestrator, "_assert_prerequisites", lambda *_a, **_k: {})
    monkeypatch.setattr(
        orchestrator,
        "verify_v2113_parent_import_receipt",
        lambda *_a, **_k: {"receipt": "verified"},
    )
    monkeypatch.setattr(
        orchestrator,
        "capability_wrappers_from_v2113_receipt",
        lambda *_a, **_k: wrappers,
    )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", _forbidden_provider)
    monkeypatch.setattr(PilotContract, "validate_provenance", _surrogate_provenance)

    result = orchestrator._execute_v2113_capability_import_stage(
        contract,
        contract.expand(stage="capability-gate"),
        raw_root=raw_root,
        repo_root=ROOT,
        paid=paid,
        run_ledger=ledger,
        budget_ledger=budget,
    )

    assert result["status"] == "complete"
    assert result["go_models"] == list(contract.models_for_stage("capability-gate"))
    for spec in contract.expand(stage="capability-gate"):
        assert ledger.status(spec.run_id) == "complete"
        assert budget.snapshot()["runs"][spec.run_id]["actual"]["completions"] == 0


@pytest.mark.parametrize("failure_point", ("prerequisite", "parent-source"))
def test_v2113_capability_early_failure_terminalizes_full_science_itt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    contract = _contract()
    raw_root = tmp_path / "raw"
    ledger, budget = _ledgers(contract, raw_root)
    if failure_point == "prerequisite":
        monkeypatch.setattr(
            orchestrator,
            "_assert_prerequisites",
            lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("deliberate early capability failure")
            ),
        )
    else:
        monkeypatch.setattr(orchestrator, "_assert_prerequisites", lambda *_a, **_k: {})
        monkeypatch.setattr(
            orchestrator,
            "verify_v2113_parent_import_receipt",
            lambda *_a, **_k: (_ for _ in ()).throw(
                orchestrator.PilotV2113ParentImportError(
                    "deliberate capability source failure"
                )
            ),
        )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", _forbidden_provider)

    result = orchestrator._execute_v2113_capability_import_stage(
        contract,
        contract.expand(stage="capability-gate"),
        raw_root=raw_root,
        repo_root=ROOT,
        paid=_paid(),
        run_ledger=ledger,
        budget_ledger=budget,
    )

    assert result["status"] == "complete-with-no-go"
    assert result["failure"]["provider_construction"] is False
    assert result["failure"]["provider_calls"] == 0
    assert all(
        ledger.status(spec.run_id) == "integrity-stopped"
        for spec in contract.expand(stage="capability-gate")
    )
    science = _science_specs(contract)
    assert len(science) == 131
    assert all(ledger.status(spec.run_id) == "integrity-stopped" for spec in science)
    assert all(
        row["actual"]["completions"] == 0 and row["actual"]["cost_usd"] == 0.0
        for row in budget.snapshot()["runs"].values()
    )
    assert (
        orchestrator._execute_v2113_capability_import_stage(
            contract,
            contract.expand(stage="capability-gate"),
            raw_root=raw_root,
            repo_root=ROOT,
            paid=_paid(),
            run_ledger=ledger,
            budget_ledger=budget,
        )
        == result
    )


def test_v2113_capability_semantic_no_go_propagates_all_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    raw_root = tmp_path / "raw"
    ledger, budget = _ledgers(contract, raw_root)
    model_ids = tuple(contract.models_for_stage("capability-gate"))
    wrappers = {
        model_id: {
            "schema_version": V2113_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            "capability": {
                "capability_pass": index == 0,
                "interface_pass": index == 0,
            },
            "integrity": {"content_sha256": (str(index + 1) * 64)},
        }
        for index, model_id in enumerate(model_ids)
    }
    monkeypatch.setattr(orchestrator, "_assert_prerequisites", lambda *_a, **_k: {})
    monkeypatch.setattr(
        orchestrator,
        "verify_v2113_parent_import_receipt",
        lambda *_a, **_k: {"receipt": "verified"},
    )
    monkeypatch.setattr(
        orchestrator,
        "capability_wrappers_from_v2113_receipt",
        lambda *_a, **_k: wrappers,
    )
    monkeypatch.setattr(PilotContract, "validate_provenance", _surrogate_provenance)
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", _forbidden_provider)

    result = orchestrator._execute_v2113_capability_import_stage(
        contract,
        contract.expand(stage="capability-gate"),
        raw_root=raw_root,
        repo_root=ROOT,
        paid=_paid(),
        run_ledger=ledger,
        budget_ledger=budget,
    )

    assert result["status"] == "complete-with-no-go"
    assert result["failure"]["error_type"] == "V2113CapabilitySemanticNoGo"
    assert result["failure"]["provider_calls"] == 0
    capability_statuses = {
        ledger.status(spec.run_id) for spec in contract.expand(stage="capability-gate")
    }
    assert capability_statuses == {"complete", "capability-no-go"}
    science = _science_specs(contract)
    assert len(science) == 131
    assert all(ledger.status(spec.run_id) == "capability-no-go" for spec in science)
    assert all(
        ledger.status(spec.run_id) == "capability-no-go"
        for spec in contract.expand(stage="long-context-preflight")
    )


@pytest.mark.parametrize(
    ("stage_id", "executor_name"),
    (
        ("parent-import", "_execute_v2113_parent_import_stage"),
        ("capability-gate", "_execute_v2113_capability_import_stage"),
        (
            "long-context-preflight",
            "_execute_v2113_preflight_authority_import_stage",
        ),
    ),
)
def test_v2113_operational_stages_route_before_provider_or_catalog_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
    executor_name: str,
) -> None:
    expected = {"status": "complete", "selected": executor_name}
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_a, **_k: _paid(),
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda root, _paid_value: root / "release_attestation.json",
    )
    monkeypatch.setattr(
        orchestrator,
        "_parent_budget_debit",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        orchestrator,
        executor_name,
        lambda *_a, **_k: calls.append(executor_name) or expected,
    )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", _forbidden_provider)
    monkeypatch.setattr(
        orchestrator, "validate_live_provider_catalog", _forbidden_provider
    )

    result = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id=stage_id,
        resume=False,
        raw_root=tmp_path / "raw",
        repo_root=tmp_path,
        parent_repo_root=(
            tmp_path / "immutable-v2112" if stage_id == "parent-import" else None
        ),
    )

    assert result == expected
    assert calls == [executor_name]


def test_v2113_calibration_uses_explicit_release_root_not_module_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    paid = _paid()
    release_root = tmp_path / "clean-release-b"
    raw_root = release_root / "experiment_results" / "pilot-v2.11.3" / "raw"
    receipt_path = raw_root / "parent-import" / "parent_import_receipt.json"
    orchestrator._atomic_json(receipt_path, {"fixture": True})
    roots: list[Path] = []
    receipt = {"integrity": {"content_sha256": "a" * 64}}
    boundary = contract.v2113_forward_boundary
    assert boundary is not None

    def validate(_value: Any, **kwargs: Any) -> dict[str, Any]:
        roots.append(Path(kwargs["repo_root"]))
        return receipt

    def calibration(_value: Any, **kwargs: Any) -> dict[str, Any]:
        roots.append(Path(kwargs["repo_root"]))
        allowlist = boundary["calibration_allowlist"]
        return {
            "calibration": {
                "q_ref": allowlist["q_ref"],
                "selected_utility_profile": {
                    "profile_id": allowlist["utility_profile_id"],
                    "rho": 1.0,
                    "labor_weight": 2.0,
                    "inverse_frisch": 0.5,
                    "consumption_scale": allowlist["q_ref"],
                    "discount_factor": 0.99,
                },
                "stage0_absolute_flow_utility_threshold": {
                    "value": allowlist["absolute_flow_utility_threshold"],
                    "treatment_outcomes_inspected": False,
                },
            },
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
            "imported_effect_cells": 0,
            "scientific_evidence": False,
        }

    monkeypatch.setattr(
        orchestrator,
        "validate_v2113_parent_import_receipt",
        validate,
    )
    monkeypatch.setattr(
        orchestrator,
        "calibration_wrapper_from_v2113_receipt",
        calibration,
    )

    resolved = orchestrator._load_verified_v2113_calibration(
        contract,
        raw_root=raw_root,
        paid=paid,
        release_repo_root=release_root,
    )

    assert resolved["q_ref"] == boundary["calibration_allowlist"]["q_ref"]
    assert roots == [release_root, release_root]
    assert release_root != ROOT


@pytest.mark.parametrize("failure_point", ("prerequisite", "parent-source"))
def test_v2113_preflight_early_failure_terminalizes_full_science_itt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    contract = _contract()
    raw_root = tmp_path / "raw"
    ledger, budget = _ledgers(contract, raw_root)
    if failure_point == "prerequisite":
        monkeypatch.setattr(
            orchestrator,
            "_assert_prerequisites",
            lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("deliberate early preflight failure")
            ),
        )
    else:
        monkeypatch.setattr(orchestrator, "_assert_prerequisites", lambda *_a, **_k: {})
        monkeypatch.setattr(
            orchestrator,
            "verify_v2113_parent_import_receipt",
            lambda *_a, **_k: (_ for _ in ()).throw(
                orchestrator.PilotV2113ParentImportError(
                    "deliberate preflight source failure"
                )
            ),
        )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", _forbidden_provider)

    result = orchestrator._execute_v2113_preflight_authority_import_stage(
        contract,
        contract.expand(stage="long-context-preflight"),
        raw_root=raw_root,
        repo_root=ROOT,
        paid=_paid(),
        run_ledger=ledger,
        budget_ledger=budget,
    )

    assert result["status"] == "complete-with-no-go"
    assert result["failure"]["provider_construction"] is False
    assert result["failure"]["provider_calls"] == 0
    assert all(
        ledger.status(spec.run_id) == "integrity-stopped"
        for spec in contract.expand(stage="long-context-preflight")
    )
    science = _science_specs(contract)
    assert len(science) == 131
    assert all(ledger.status(spec.run_id) == "integrity-stopped" for spec in science)
    assert all(
        row["actual"]["completions"] == 0 and row["actual"]["cost_usd"] == 0.0
        for row in budget.snapshot()["runs"].values()
    )


def test_v2113_preflight_import_failure_is_zero_provider_and_terminalizes_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    paid = _paid()
    raw_root = tmp_path / "raw"
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    boundary = contract.v2113_forward_boundary
    assert boundary is not None
    budget = orchestrator.PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=ParentBudgetDebit.from_dict(boundary["parent_budget_debit"]),
    )
    wrappers = {model_id: {"model_id": model_id} for model_id in MODELS}

    monkeypatch.setattr(orchestrator, "_assert_prerequisites", lambda *_a, **_k: {})
    monkeypatch.setattr(
        orchestrator,
        "verify_v2113_parent_import_receipt",
        lambda *_a, **_k: {"receipt": "verified"},
    )
    monkeypatch.setattr(
        orchestrator,
        "preflight_wrappers_from_v2113_receipt",
        lambda *_a, **_k: wrappers,
    )
    monkeypatch.setattr(
        orchestrator,
        "persist_v2113_resealed_observed_p95_authority",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("deliberate reseal failure")
        ),
    )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", _forbidden_provider)

    result = orchestrator._execute_v2113_preflight_authority_import_stage(
        contract,
        contract.expand(stage="long-context-preflight"),
        raw_root=raw_root,
        repo_root=ROOT,
        paid=paid,
        run_ledger=ledger,
        budget_ledger=budget,
    )

    assert result["status"] == "complete-with-no-go"
    assert result["go_models"] == []
    assert result["failure"]["provider_construction"] is False
    assert result["failure"]["provider_calls"] == 0
    assert all(
        ledger.status(spec.run_id) == "integrity-stopped"
        for spec in contract.expand(stage="long-context-preflight")
    )
    science = tuple(
        spec
        for stage_id in orchestrator._scientific_stage_ids(contract)
        for spec in contract.expand(stage=stage_id)
    )
    assert len(science) == 131
    assert all(ledger.status(spec.run_id) == "integrity-stopped" for spec in science)
    budget_rows = budget.snapshot()["runs"]
    assert all(
        row["actual"]["completions"] == 0 and row["actual"]["cost_usd"] == 0.0
        for row in budget_rows.values()
    )


def test_v2113_runner_transform_failure_precedes_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    spec = contract.expand(stage="experiment-a")[0]
    projection = orchestrator.RunProjection(
        run_id=spec.run_id,
        stage_bucket=spec.budget_bucket,
        cost_usd=1.0,
        completions=1,
        storage_bytes=1_000_000,
        basis={
            "method": "test",
            "run_call_limit": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
        },
    )
    transformed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_projection",
        lambda *_a, **_k: (
            {"served_model": "sentinel", "projection": {}},
            tmp_path / "projection.json",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_verified_observed_p95_binding",
        lambda *_a, **_k: {"base": "authority"},
    )

    def reject_transform(binding: dict[str, Any]) -> None:
        transformed.append(binding)
        raise orchestrator.PilotV2113GateError("deliberate transformer failure")

    monkeypatch.setattr(
        orchestrator,
        "runner_reservations_from_v2113_gate_binding",
        reject_transform,
    )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden_provider)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="runner p95 authority transformation failed",
    ):
        orchestrator._execute_actor_run(
            contract,
            spec,
            raw_root=tmp_path / "raw",
            paid=_paid(),
            projection=projection,
            budget=orchestrator._run_budget_from_projection(projection),
            authority_repo_root=ROOT,
        )

    assert transformed == [{"base": "authority"}]


def test_v2113_scientific_config_uses_alternate_clean_authority_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    spec = contract.expand(stage="experiment-a")[0]
    release_root = tmp_path / "alternate-clean-release"
    release_root.mkdir()
    observed_roots: list[Path | None] = []
    provider_calls: list[str] = []

    def construct_config(**kwargs: Any) -> dict[str, Any]:
        observed_roots.append(runner._OBSERVED_P95_AUTHORITY_REPO_ROOT.get())
        return kwargs

    monkeypatch.setattr(orchestrator, "VerifiedRunConfig", construct_config)
    monkeypatch.setattr(
        orchestrator,
        "resolve_utility",
        lambda *_a, **_k: orchestrator.UtilityConfig(),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        lambda *_a, **_k: provider_calls.append("constructed"),
    )

    config = orchestrator.config_for_spec(
        contract,
        spec,
        raw_root=release_root / "experiment_results" / "pilot-v2.11.3" / "raw",
        paid_provenance=_paid(),
        authority_repo_root=release_root,
        verify_bound_inputs=True,
        preflight_p95_reservations={},
    )

    assert config["scientific_scope"] == "preregistered_mechanism_micro_pilot"
    assert observed_roots == [release_root.resolve()]
    assert runner._OBSERVED_P95_AUTHORITY_REPO_ROOT.get() is None
    assert provider_calls == []
    assert release_root.resolve() != ROOT
