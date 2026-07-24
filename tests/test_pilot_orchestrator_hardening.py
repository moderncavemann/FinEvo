from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verified_memory.budget import (
    BudgetExceeded,
    BudgetLimits,
    RunBudget,
    UsageRecord,
)
from verified_memory.m2_episodic import EvidenceLinkedEpisodicTrack
from verified_memory.failure_artifacts import verify_failure_receipt
from verified_memory.pilot_budget import (
    PilotBudgetError,
    PilotBudgetLedger,
    RunProjection,
    preflight_p95,
)
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import _write_package_files
from verified_memory.pilot_orchestrator import (
    PILOT_PROJECTION_SCHEMA_VERSION,
    PILOT_STAGE_RECEIPT_SCHEMA_VERSION,
    GitProvenance,
    PilotOrchestrationError,
    PilotRunLedger,
    _assert_prerequisites,
    _build_experiment_c_sensitivity,
    _derive_stage0_absolute_flow_threshold,
    _execute_actor_run,
    _execute_d_seed,
    _execute_q_ref,
    _exclusive_real_stage_lock,
    _finalize_budget_safely,
    _load_verified_q_ref,
    _load_verified_experiment_c_sensitivity,
    _load_verified_stage0_selection,
    _recover_or_stop_interrupted_reservation,
    _remaining_core_projections,
    _remaining_scientific_projections,
    _run_budget_from_projection,
    _seal_bound_payload,
    _write_experiment_c_sensitivity,
    _write_stage_receipt,
    conservative_projection,
    execute_stage,
    projection_from_preflight,
)
from verified_memory.pilot_provider_catalog import (
    PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
)
from verified_memory.runner import (
    RunnerFailure,
    VerifiedRunConfig,
    VerifiedRunError,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v1.yaml"


def _release_attestation(commit: str) -> dict:
    payload = {
        "schema_version": "finevo-pilot-release-attestation-v1",
        "status": "pass",
        "head_commit": commit,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        **payload,
        "attestation_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _paid(contract) -> GitProvenance:
    commit = "1" * 40
    return GitProvenance(
        git_tag="pilot-v1",
        head_commit=commit,
        tag_commit=commit,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding=contract.validate_provenance(commit, "pilot-v1"),
        release_attestation=_release_attestation(commit),
    )


def _zero_projection(_contract, spec, **_kwargs) -> RunProjection:
    return RunProjection(
        run_id=spec.run_id,
        stage_bucket=spec.budget_bucket,
        cost_usd=0.0,
        completions=1,
        storage_bytes=100_000,
        basis={
            "method": "test-zero",
            "prompt_tokens": 10,
            "completion_tokens": 10,
        },
    )


def _runner_failure(message: str = "fixture parse failure") -> VerifiedRunError:
    return VerifiedRunError(
        message,
        failure=RunnerFailure(
            schema_version="verified-runner-failure-v1",
            error_stage="action-parse",
            call_kind="action",
            decision_t=5,
            agent_id=2,
            error_type="JSONDecodeError",
            message=message,
            prompt_hash="a" * 64,
            raw_output_hash="b" * 64,
            provider="openai",
            model="gpt-5.2-2025-12-11",
            attempts=1,
        ),
    )


def test_first_invocation_registers_all_172_and_propagates_main_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_paid_provenance",
        lambda *args, **kwargs: paid,
    )

    def catalog(_contract, *, model_ids):
        model_id = model_ids[0]
        if model_id == "gpt52_main":
            raise RuntimeError("main model interface is incompatible")
        return {
            "schema_version": PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "status": "pass",
            "paid_completions": 0,
            "rows": [{"model_id": model_id, "status": "pass"}],
        }

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        catalog,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.conservative_projection",
        _zero_projection,
    )

    def complete_gate(_contract, spec, *, raw_root, budget, **_kwargs):
        artifact = raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json"
        _write_json(artifact, {"fixture": True})
        return "complete", artifact, budget, {"go": True, "reason": None}

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._execute_capability_preflight",
        complete_gate,
    )
    raw = tmp_path / "raw"
    receipt = execute_stage(
        contract_path=CONTRACT_PATH,
        stage_id="capability-preflight",
        resume=True,
        raw_root=raw,
        repo_root=ROOT,
    )
    assert receipt["status"] == "complete-with-no-go"
    assert receipt["go"] is False
    assert receipt["complete_cell_count"] == 5

    ledger = json.loads((raw / "run_ledger.json").read_text(encoding="utf-8"))
    assert len(ledger["runs"]) == 172
    rows = ledger["runs"]
    main_capability = contract.expand(
        stage="capability-preflight",
        model="gpt52_main",
    )[0]
    assert rows[main_capability.run_id]["status"] == "capability-no-go"
    for stage_id in (
        "stage0-calibration",
        "experiment-a",
        "experiment-b",
        "experiment-c",
        "experiment-d",
    ):
        assert {
            rows[spec.run_id]["status"]
            for spec in contract.expand(stage=stage_id)
        } == {"capability-no-go"}
    assert {
        rows[spec.run_id]["status"]
        for spec in contract.expand(stage="cross-model-sentinels")
    } <= {"scheduled", "capability-no-go"}

    with pytest.raises(PilotOrchestrationError, match="GPT-5.2"):
        _assert_prerequisites(
            contract,
            "stage0-calibration",
            raw_root=raw,
        )
    with pytest.raises(PilotOrchestrationError, match="prerequisites failed"):
        execute_stage(
            contract_path=CONTRACT_PATH,
            stage_id="stage0-calibration",
            resume=True,
            raw_root=raw,
            repo_root=ROOT,
        )
    blocked = json.loads(
        (raw / "stage0-calibration" / "stage_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert blocked["status"] == "prerequisite-no-go"
    assert blocked["go"] is False
    assert {
        rowspec["status"]
        for rowspec in json.loads(
            (raw / "run_ledger.json").read_text(encoding="utf-8")
        )["runs"].values()
        if rowspec["spec"]["stage_id"] == "cross-model-sentinels"
    } <= {"capability-no-go", "integrity-stopped"}


def test_ordinary_failed_a_cell_does_not_block_d_execution_prerequisite(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    ledger = PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    prerequisite_stages = ("experiment-a", "experiment-b", "experiment-c")
    specs = tuple(
        spec
        for stage_id in prerequisite_stages
        for spec in contract.expand(stage=stage_id)
    )
    ledger.register(specs)
    failed_a = contract.expand(stage="experiment-a")[0]
    for spec in specs:
        ledger.finalize(
            spec.run_id,
            status="failed" if spec.run_id == failed_a.run_id else "complete",
            artifact=None,
            failure=(
                {"error_type": "VerifiedRunError", "message": "provider failure"}
                if spec.run_id == failed_a.run_id
                else None
            ),
        )

    a_receipt_path = _write_stage_receipt(
        contract,
        "experiment-a",
        raw_root=raw,
        ledger=ledger,
        status="complete-with-no-go",
    )
    _write_stage_receipt(
        contract,
        "experiment-b",
        raw_root=raw,
        ledger=ledger,
        status="complete",
    )
    _write_stage_receipt(
        contract,
        "experiment-c",
        raw_root=raw,
        ledger=ledger,
        status="complete-with-no-go",
        artifacts={
            "zero_api_rule_sensitivity_failure": {
                "error_type": "PilotOrchestrationError",
                "message": "one B-full source is unavailable",
            },
        },
    )

    a_receipt = json.loads(a_receipt_path.read_text(encoding="utf-8"))
    assert a_receipt["terminal"] is True
    assert a_receipt["denominator_terminal"] is True
    assert a_receipt["status_counts"] == {"complete": 19, "failed": 1}
    assert a_receipt["scientific_matrix_complete"] is False
    assert a_receipt["go"] is False
    assert a_receipt["execution_progression_go"] is True

    # Neither an ordinary ITT failure nor C's descriptive sensitivity no-go is
    # an execution prerequisite gate.  The 4/5 claim threshold is evaluated
    # only when the evidence package is built.
    _assert_prerequisites(
        contract,
        "experiment-d",
        raw_root=raw,
    )


def test_experiment_a_stage_end_keeps_descendants_scheduled_after_one_run_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    raw = tmp_path / "raw"
    _write_json(
        raw / "stage0-calibration" / "stage_receipt.json",
        {
            "schema_version": PILOT_STAGE_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "stage_id": "stage0-calibration",
            "status": "complete",
            "terminal": True,
            "registered_run_count": 14,
            "complete_cell_count": 14,
            "status_counts": {"complete": 14},
            "go": True,
            "go_models": [],
        },
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_paid_provenance",
        lambda *args, **kwargs: paid,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        lambda _contract, *, model_ids: {
            "schema_version": PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "status": "pass",
            "paid_completions": 0,
            "rows": [{"model_id": model_ids[0], "status": "pass"}],
        },
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._remaining_core_projections",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.projection_from_preflight",
        _zero_projection,
    )
    failed_run_id: str | None = None

    def actor(_contract, spec, *, raw_root, budget, **_kwargs):
        nonlocal failed_run_id
        if failed_run_id is None:
            failed_run_id = spec.run_id
            raise _runner_failure("one registered action parse failure")
        manifest = raw_root / spec.stage_id / "runs" / spec.run_id / "manifest.json"
        _write_json(manifest, {"fixture": True})
        return manifest, budget, {"fixture": True}

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._execute_actor_run",
        actor,
    )
    receipt = execute_stage(
        contract_path=CONTRACT_PATH,
        stage_id="experiment-a",
        resume=True,
        raw_root=raw,
        repo_root=ROOT,
    )

    assert failed_run_id is not None
    assert receipt["status"] == "complete-with-no-go"
    assert receipt["status_counts"] == {"complete": 19, "failed": 1}
    assert receipt["execution_progression_go"] is True
    rows = json.loads(
        (raw / "run_ledger.json").read_text(encoding="utf-8")
    )["runs"]
    for stage_id in (
        "experiment-b",
        "experiment-c",
        "experiment-d",
        "cross-model-sentinels",
    ):
        assert {
            rows[spec.run_id]["status"]
            for spec in contract.expand(stage=stage_id)
        } == {"scheduled"}


@pytest.mark.parametrize("hard_status", ["budget-stopped", "integrity-stopped"])
def test_budget_or_integrity_cell_blocks_downstream_execution(
    tmp_path: Path,
    hard_status: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    ledger = PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    a_specs = contract.expand(stage="experiment-a")
    ledger.register(a_specs)
    for index, spec in enumerate(a_specs):
        ledger.finalize(
            spec.run_id,
            status=hard_status if index == 0 else "complete",
            artifact=None,
            failure=(
                {"error_type": "FixtureHardStop"}
                if index == 0
                else None
            ),
        )
    receipt_path = _write_stage_receipt(
        contract,
        "experiment-a",
        raw_root=raw,
        ledger=ledger,
        status="complete-with-no-go",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["denominator_terminal"] is True
    assert receipt["hard_stop_cell_count"] == 1
    assert receipt["execution_progression_go"] is False

    with pytest.raises(PilotOrchestrationError, match="execution no-go"):
        _assert_prerequisites(
            contract,
            "experiment-d",
            raw_root=raw,
        )


def test_standard_post_reservation_failure_is_sealed_and_storage_accounted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    failure_error = _runner_failure()
    failed_run_id: str | None = None

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_paid_provenance",
        lambda *args, **kwargs: paid,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        lambda _contract, *, model_ids: {
            "schema_version": PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "status": "pass",
            "paid_completions": 0,
            "rows": [{"model_id": model_ids[0], "status": "pass"}],
        },
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.conservative_projection",
        _zero_projection,
    )

    def capability(_contract, spec, *, raw_root, budget, **_kwargs):
        nonlocal failed_run_id
        if failed_run_id is None:
            failed_run_id = spec.run_id
            raise failure_error
        artifact = raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json"
        _write_json(artifact, {"fixture": True})
        return "complete", artifact, budget, {"go": True, "reason": None}

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._execute_capability_preflight",
        capability,
    )
    raw = tmp_path / "raw"
    execute_stage(
        contract_path=CONTRACT_PATH,
        stage_id="capability-preflight",
        resume=True,
        raw_root=raw,
        repo_root=ROOT,
    )

    assert failed_run_id is not None
    run_state = json.loads(
        (raw / "run_ledger.json").read_text(encoding="utf-8")
    )["runs"][failed_run_id]
    failure_manifest = Path(run_state["artifact"])
    assert failure_manifest.name == "failure_manifest.json"
    assert failure_manifest.parent.name == "failure_receipt"
    assert not (failure_manifest.parent / "manifest.json").exists()
    receipt = verify_failure_receipt(failure_manifest.parent)
    assert receipt["error"]["details"] == failure_error.failure.to_dict()
    assert run_state["failure"]["details"] == failure_error.failure.to_dict()
    assert receipt["scope"].endswith("/capability_probe")
    assert receipt["config"]["contract_sha256"] == contract.canonical_hash
    assert receipt["config"]["projection"]["run_id"] == failed_run_id
    assert receipt["config"]["run_specs"] == [
        run_state["spec"],
    ]
    assert set(receipt["config"]["provider_request_profiles"]) == {
        run_state["spec"]["model_id"],
    }
    assert receipt["budget_snapshot"]["budget_id"] == (
        f"{failed_run_id}-budget"
    )
    assert receipt["provenance"]["paid_provenance"] == paid.to_dict()
    assert receipt["provenance"]["scientific_evidence"] is False

    run_dir = failure_manifest.parent.parent
    accounted_size = sum(
        path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
    )
    budget_state = json.loads(
        (raw / "budget_ledger.json").read_text(encoding="utf-8")
    )["runs"][failed_run_id]
    assert budget_state["status"] == "failed"
    assert budget_state["actual"]["storage_bytes"] == accounted_size


def test_d_group_failure_receipt_is_shared_and_preserves_runner_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    actor_spec = contract.expand(
        stage="experiment-a",
        model="gpt52_main",
        arm="full",
    )[0]
    actor_projection = RunProjection(
        run_id=actor_spec.run_id,
        stage_bucket="core",
        cost_usd=0.01,
        completions=1,
        storage_bytes=100_000,
        basis={
            "method": "fixture",
            "run_call_limit": 1,
            "hosted_completion_cap_counted": True,
            "prompt_tokens": 10,
            "completion_tokens": 10,
        },
    )
    actor_budget = _run_budget_from_projection(actor_projection)
    actor_config = VerifiedRunConfig(
        run_id=actor_spec.run_id,
        episode_length=12,
        enable_semantic=True,
    )
    actor_events: list[tuple[str, str]] = []
    with monkeypatch.context() as actor_patch:
        actor_patch.setattr(
            "verified_memory.pilot_orchestrator._runner_p95_reservations",
            lambda *_args, **_kwargs: {"sealed": {}},
        )
        actor_patch.setattr(
            "verified_memory.pilot_orchestrator.config_for_spec",
            lambda *_args, **_kwargs: actor_config,
        )

        def validate_actor_authority(config, *, provider_model_name):
            assert config is actor_config
            assert config.enable_semantic is True
            actor_events.append(("validate-action-semantic", provider_model_name))
            return {}

        def actor_provider_after_validation(*_args, **_kwargs):
            actor_events.append(("provider", "constructed"))
            raise RuntimeError("actor provider construction sentinel")

        actor_patch.setattr(
            "verified_memory.pilot_orchestrator."
            "validate_preflight_p95_reservations",
            validate_actor_authority,
        )
        actor_patch.setattr(
            "verified_memory.pilot_orchestrator._provider_for_profile",
            actor_provider_after_validation,
        )
        with pytest.raises(
            RuntimeError,
            match="actor provider construction sentinel",
        ):
            _execute_actor_run(
                contract,
                actor_spec,
                raw_root=tmp_path / "actor-raw",
                paid=paid,
                projection=actor_projection,
                budget=actor_budget,
            )
    assert actor_events == [
        (
            "validate-action-semantic",
            "openai/gpt-5.2-2025-12-11",
        ),
        ("provider", "constructed"),
    ]

    seed = contract.seeds["sets"]["main"][0]
    specs = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.environment_seed == seed
    )
    assert len(specs) == 11
    raw = tmp_path / "raw"
    run_ledger = PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    run_ledger.register(specs)
    budget_ledger = PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    projection_id = f"fixture-d-group-{seed}"
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        lambda *_args, **_kwargs: RunProjection(
            run_id=projection_id,
            stage_bucket="core",
            cost_usd=0.0,
            completions=1,
            storage_bytes=100_000,
            basis={
                "method": "fixture",
                "run_call_limit": 1,
                "hosted_completion_cap_counted": True,
                "prompt_tokens": 10,
                "completion_tokens": 10,
            },
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._load_verified_stage0_selection",
        lambda *_args, **_kwargs: {
            "selected_utility": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 1.0,
                "consumption_scale": 1.0,
                "discount_factor": 0.99,
            }
        },
    )
    failure_error = _runner_failure("D branch provider failure")
    d_events: list[tuple[str, str]] = []

    def validate_d_authority(config, *, provider_model_name):
        assert config.enable_semantic is True
        d_events.append(("validate-action-semantic", provider_model_name))
        return {}

    def fail_before_provider_dispatch(*_args, **_kwargs):
        # The failure is deliberately injected after the durable group
        # reservation but before a provider object (and therefore any call)
        # exists.
        reserved = budget_ledger.snapshot()["runs"][projection_id]
        assert reserved["status"] == "reserved"
        assert reserved["actual"] is None
        d_events.append(("provider", "constructed"))
        raise failure_error

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator."
        "validate_preflight_p95_reservations",
        validate_d_authority,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        fail_before_provider_dispatch,
    )

    _execute_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        diagnostic=False,
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )

    state = run_ledger.snapshot()["runs"]
    artifacts = {state[spec.run_id]["artifact"] for spec in specs}
    assert len(artifacts) == 1
    failure_manifest = Path(artifacts.pop())
    assert failure_manifest.parent.name == "failure_receipt"
    receipt = verify_failure_receipt(failure_manifest.parent)
    assert receipt["error"]["details"] == failure_error.failure.to_dict()
    assert (
        receipt["scope"]
        == "finevo-pilot/experiment-d/shared-checkpoint-group"
    )
    assert receipt["config"]["contract_sha256"] == contract.canonical_hash
    assert receipt["config"]["projection"]["run_id"] == projection_id
    assert receipt["config"]["run_specs"] == [
        spec.to_dict() for spec in specs
    ]
    assert receipt["budget_snapshot"]["budget_id"] == (
        f"{projection_id}-budget"
    )
    assert receipt["provenance"]["paid_provenance"] == paid.to_dict()
    assert receipt["provenance"]["scientific_evidence"] is False
    assert {
        state[spec.run_id]["status"] for spec in specs
    } == {"failed"}
    assert all(
        state[spec.run_id]["failure"]["details"]
        == failure_error.failure.to_dict()
        for spec in specs
    )
    assert d_events == [
        (
            "validate-action-semantic",
            "openai/gpt-5.2-2025-12-11",
        ),
        ("provider", "constructed"),
    ]

    group_dir = failure_manifest.parent.parent
    accounted_size = sum(
        path.stat().st_size for path in group_dir.rglob("*") if path.is_file()
    )
    budget_state = budget_ledger.snapshot()["runs"][projection_id]
    assert budget_state["status"] == "failed"
    assert budget_state["actual"]["storage_bytes"] == accounted_size


def test_d_reservation_rejection_does_not_create_a_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    seed = contract.seeds["sets"]["main"][0]
    specs = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.environment_seed == seed
    )
    raw = tmp_path / "raw"
    run_ledger = PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    run_ledger.register(specs)
    budget_ledger = PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    projection_id = f"fixture-d-group-{seed}"
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        lambda *_args, **_kwargs: RunProjection(
            run_id=projection_id,
            stage_bucket="core",
            cost_usd=0.0,
            completions=1,
            storage_bytes=100_000,
            basis={
                "method": "fixture",
                "run_call_limit": 1,
                "hosted_completion_cap_counted": True,
                "prompt_tokens": 10,
                "completion_tokens": 10,
            },
        ),
    )

    def reject_reservation(_projection: RunProjection) -> None:
        raise PilotBudgetError("injected reservation rejection")

    monkeypatch.setattr(budget_ledger, "reserve", reject_reservation)
    with pytest.raises(PilotBudgetError, match="reservation rejection"):
        _execute_d_seed(
            contract,
            specs,
            raw_root=raw,
            paid=paid,
            diagnostic=False,
            budget_ledger=budget_ledger,
            run_ledger=run_ledger,
        )

    assert budget_ledger.snapshot()["runs"] == {}
    assert {
        run_ledger.snapshot()["runs"][spec.run_id]["status"]
        for spec in specs
    } == {"scheduled"}
    assert not tuple(raw.rglob("failure_manifest.json"))

    continuation_treatments = (
        "matched-a",
        "matched-b",
        "no-memory",
        "shuffled-episodic",
        "wrong-context",
        "erroneous-verified",
        "erroneous-unverified",
    )
    narrative_ids = tuple(
        sorted(
            str(spec.narrative_id)
            for spec in specs
            if spec.arm_id == "narrative-content"
        )
    )
    continuation_fixture = {
        "matched_replay_equal": True,
        "checkpoint_hash": "a" * 64,
        "prefix_hash": "b" * 64,
        "branches": {
            treatment: {
                "metrics": {},
                "delta_vs_matched_a": {},
                "trajectory": [{"decisions": []}],
                "provider_call_journal": {},
            }
            for treatment in continuation_treatments
        },
    }
    narrative_fixture = {
        "metrics": {narrative_id: {} for narrative_id in narrative_ids},
        "semantic_equivalence_within_one_action_bin": {},
        "aligned_vs_opposite_delta": {},
        "claim_boundary": "fixture",
        "branches": {
            narrative_id: {
                "trajectory": [{"decisions": []}],
                "provider_call_journal": {},
            }
            for narrative_id in narrative_ids
        },
    }
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._load_verified_stage0_selection",
        lambda *_args, **_kwargs: {
            "selected_utility": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 1.0,
                "consumption_scale": 1.0,
                "discount_factor": 0.99,
            }
        },
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator."
        "validate_preflight_p95_reservations",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.build_pilot_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {"schema_version": "fixture-checkpoint"}
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.run_pilot_continuations",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: json.loads(json.dumps(continuation_fixture))
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.run_pilot_narratives",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: json.loads(json.dumps(narrative_fixture))
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_continuation_causal_bindings",
        lambda *_args, **_kwargs: {"fixture": "continuation"},
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_narrative_causal_bindings",
        lambda *_args, **_kwargs: {"fixture": "narrative"},
    )

    from verified_memory import pilot_orchestrator as orchestrator_module

    real_terminal_writer = orchestrator_module.write_terminal_summary

    reconcile_raw = tmp_path / "reconcile-raw"
    reconcile_runs = PilotRunLedger(
        reconcile_raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    reconcile_runs.register(specs)
    reconcile_budget = PilotBudgetLedger(
        reconcile_raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    reconcile_projection_id = f"reconcile-d-group-{seed}"
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        lambda *_args, **_kwargs: RunProjection(
            run_id=reconcile_projection_id,
            stage_bucket="core",
            cost_usd=0.0,
            completions=1,
            storage_bytes=10_000_000,
            basis={
                "method": "fixture",
                "run_call_limit": 1,
                "hosted_completion_cap_counted": True,
                "prompt_tokens": 10,
                "completion_tokens": 10,
            },
        ),
    )
    reconciliation_flags: list[tuple[bool, str]] = []

    def observe_terminal_before_reconciliation(path, **kwargs):
        state = reconcile_budget.snapshot()["runs"][reconcile_projection_id]
        reconciliation_flags.append(
            (bool(kwargs["scientific_evidence"]), str(state["status"]))
        )
        return real_terminal_writer(path, **kwargs)

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.write_terminal_summary",
        observe_terminal_before_reconciliation,
    )
    real_reconcile_finalize = reconcile_budget.finalize
    reconcile_finalize_statuses: list[str] = []

    def reject_complete_reconciliation(run_id: str, *, status: str, **kwargs):
        reconcile_finalize_statuses.append(status)
        if status == "complete":
            raise PilotBudgetError("injected D reconciliation failure")
        return real_reconcile_finalize(run_id, status=status, **kwargs)

    monkeypatch.setattr(
        reconcile_budget,
        "finalize",
        reject_complete_reconciliation,
    )
    _execute_d_seed(
        contract,
        specs,
        raw_root=reconcile_raw,
        paid=paid,
        diagnostic=False,
        budget_ledger=reconcile_budget,
        run_ledger=reconcile_runs,
    )
    assert reconciliation_flags == [(False, "reserved")] * len(specs)
    assert reconcile_finalize_statuses == ["complete", "integrity-stopped"]
    assert {
        reconcile_runs.status(spec.run_id) for spec in specs
    } == {"integrity-stopped"}
    assert not tuple(
        (reconcile_raw / "experiment-d" / "summaries").glob("*.json")
    )
    assert not (
        reconcile_raw
        / "experiment-d"
        / "checkpoints"
        / f"s{seed}"
        / "pending_non_scientific_summaries"
    ).exists()
    for source_name in ("continuations.json", "narratives.json"):
        source = json.loads(
            (
                reconcile_raw
                / "experiment-d"
                / "checkpoints"
                / f"s{seed}"
                / source_name
            ).read_text(encoding="utf-8")
        )
        assert source["scientific_evidence"] is False
    reconciliation_artifact = Path(
        reconcile_runs.snapshot()["runs"][specs[0].run_id]["artifact"]
    )
    reconciliation_receipt = verify_failure_receipt(
        reconciliation_artifact.parent
    )
    assert reconciliation_receipt["provenance"]["scientific_evidence"] is False

    ledger_finalize_raw = tmp_path / "ledger-finalize-raw"
    ledger_finalize_runs = PilotRunLedger(
        ledger_finalize_raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    ledger_finalize_runs.register(specs)
    ledger_finalize_budget = PilotBudgetLedger(
        ledger_finalize_raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    ledger_finalize_projection_id = f"ledger-finalize-d-group-{seed}"
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        lambda *_args, **_kwargs: RunProjection(
            run_id=ledger_finalize_projection_id,
            stage_bucket="core",
            cost_usd=0.0,
            completions=1,
            storage_bytes=10_000_000,
            basis={
                "method": "fixture",
                "run_call_limit": 1,
                "hosted_completion_cap_counted": True,
                "prompt_tokens": 10,
                "completion_tokens": 10,
            },
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.write_terminal_summary",
        real_terminal_writer,
    )
    real_ledger_finalize_budget = ledger_finalize_budget.finalize
    ledger_finalize_budget_statuses: list[str] = []

    def observe_ledger_finalize_budget(
        run_id: str,
        *,
        status: str,
        **kwargs,
    ):
        ledger_finalize_budget_statuses.append(status)
        return real_ledger_finalize_budget(
            run_id,
            status=status,
            **kwargs,
        )

    monkeypatch.setattr(
        ledger_finalize_budget,
        "finalize",
        observe_ledger_finalize_budget,
    )
    real_finalize_many = ledger_finalize_runs.finalize_many
    ledger_finalize_many_statuses: list[tuple[str, ...]] = []

    def fail_first_complete_batch(finalizations):
        statuses = tuple(str(row["status"]) for row in finalizations)
        ledger_finalize_many_statuses.append(statuses)
        if set(statuses) == {"complete"}:
            raise OSError("injected complete run-ledger batch failure")
        return real_finalize_many(finalizations)

    monkeypatch.setattr(
        ledger_finalize_runs,
        "finalize_many",
        fail_first_complete_batch,
    )
    _execute_d_seed(
        contract,
        specs,
        raw_root=ledger_finalize_raw,
        paid=paid,
        diagnostic=False,
        budget_ledger=ledger_finalize_budget,
        run_ledger=ledger_finalize_runs,
    )
    assert ledger_finalize_budget_statuses == ["complete"]
    assert ledger_finalize_many_statuses == [
        ("complete",) * len(specs),
        ("integrity-stopped",) * len(specs),
    ]
    ledger_finalize_state = ledger_finalize_runs.snapshot()["runs"]
    assert {
        ledger_finalize_state[spec.run_id]["status"] for spec in specs
    } == {"integrity-stopped"}
    assert not tuple(
        (ledger_finalize_raw / "experiment-d" / "summaries").glob("*.json")
    )
    ledger_finalize_group = (
        ledger_finalize_raw
        / "experiment-d"
        / "checkpoints"
        / f"s{seed}"
    )
    for source_name in ("continuations.json", "narratives.json"):
        source = json.loads(
            (ledger_finalize_group / source_name).read_text(encoding="utf-8")
        )
        assert source["scientific_evidence"] is False
    ledger_finalize_artifacts = {
        ledger_finalize_state[spec.run_id]["artifact"] for spec in specs
    }
    assert len(ledger_finalize_artifacts) == 1
    ledger_finalize_receipt = verify_failure_receipt(
        Path(ledger_finalize_artifacts.pop()).parent
    )
    assert (
        ledger_finalize_receipt["scope"]
        == "finevo-pilot/experiment-d/post-reconciliation-publication"
    )
    assert (
        ledger_finalize_receipt["provenance"]["scientific_evidence"] is False
    )

    interruption_raw = tmp_path / "interruption-raw"
    interruption_runs = PilotRunLedger(
        interruption_raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    interruption_runs.register(specs)
    interruption_budget = PilotBudgetLedger(
        interruption_raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    interruption_projection_id = f"interruption-d-group-{seed}"
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        lambda *_args, **_kwargs: RunProjection(
            run_id=interruption_projection_id,
            stage_bucket="core",
            cost_usd=0.0,
            completions=1,
            storage_bytes=10_000_000,
            basis={
                "method": "fixture",
                "run_call_limit": 1,
                "hosted_completion_cap_counted": True,
                "prompt_tokens": 10,
                "completion_tokens": 10,
            },
        ),
    )
    interruption_calls = {
        "provider": 0,
        "checkpoint": 0,
        "continuation": 0,
        "narrative": 0,
    }

    def interruption_provider(*_args, **_kwargs):
        interruption_calls["provider"] += 1
        return object()

    def interruption_checkpoint(*_args, **_kwargs):
        interruption_calls["checkpoint"] += 1
        return SimpleNamespace(
            to_dict=lambda: {"schema_version": "fixture-checkpoint"}
        )

    def interruption_continuation(*_args, **_kwargs):
        interruption_calls["continuation"] += 1
        return SimpleNamespace(
            to_dict=lambda: json.loads(json.dumps(continuation_fixture))
        )

    def interruption_narrative(*_args, **_kwargs):
        interruption_calls["narrative"] += 1
        return SimpleNamespace(
            to_dict=lambda: json.loads(json.dumps(narrative_fixture))
        )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        interruption_provider,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.build_pilot_checkpoint",
        interruption_checkpoint,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.run_pilot_continuations",
        interruption_continuation,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.run_pilot_narratives",
        interruption_narrative,
    )
    scientific_write_count = 0

    def interrupt_fifth_scientific_writer(path, **kwargs):
        nonlocal scientific_write_count
        if kwargs["scientific_evidence"]:
            scientific_write_count += 1
            if scientific_write_count == 5:
                raise KeyboardInterrupt(
                    "injected fifth scientific publication interruption"
                )
        return real_terminal_writer(path, **kwargs)

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.write_terminal_summary",
        interrupt_fifth_scientific_writer,
    )
    with pytest.raises(
        KeyboardInterrupt,
        match="fifth scientific publication interruption",
    ):
        _execute_d_seed(
            contract,
            specs,
            raw_root=interruption_raw,
            paid=paid,
            diagnostic=False,
            budget_ledger=interruption_budget,
            run_ledger=interruption_runs,
        )
    assert interruption_calls == {
        "provider": 1,
        "checkpoint": 1,
        "continuation": 1,
        "narrative": 1,
    }
    assert (
        interruption_budget.snapshot()["runs"][interruption_projection_id][
            "status"
        ]
        == "complete"
    )
    assert {
        interruption_runs.status(spec.run_id) for spec in specs
    } == {"scheduled"}
    interruption_summaries = (
        interruption_raw / "experiment-d" / "summaries"
    )
    assert len(tuple(interruption_summaries.glob("*.json"))) == 4
    interruption_group = (
        interruption_raw
        / "experiment-d"
        / "checkpoints"
        / f"s{seed}"
    )
    for source_name in ("continuations.json", "narratives.json"):
        source = json.loads(
            (interruption_group / source_name).read_text(encoding="utf-8")
        )
        assert source["scientific_evidence"] is True

    def provider_must_not_resume(*_args, **_kwargs):
        raise AssertionError("resume must not construct a provider")

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        provider_must_not_resume,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.write_terminal_summary",
        real_terminal_writer,
    )
    _execute_d_seed(
        contract,
        specs,
        raw_root=interruption_raw,
        paid=paid,
        diagnostic=False,
        budget_ledger=interruption_budget,
        run_ledger=interruption_runs,
    )
    assert interruption_calls == {
        "provider": 1,
        "checkpoint": 1,
        "continuation": 1,
        "narrative": 1,
    }
    interruption_state = interruption_runs.snapshot()["runs"]
    assert {
        interruption_state[spec.run_id]["status"] for spec in specs
    } == {"integrity-stopped"}
    assert not tuple(interruption_summaries.glob("*.json"))
    for source_name in ("continuations.json", "narratives.json"):
        source = json.loads(
            (interruption_group / source_name).read_text(encoding="utf-8")
        )
        assert source["scientific_evidence"] is False

    promotion_raw = tmp_path / "promotion-raw"
    promotion_runs = PilotRunLedger(
        promotion_raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    promotion_runs.register(specs)
    promotion_budget = PilotBudgetLedger(
        promotion_raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    promotion_projection_id = f"promotion-d-group-{seed}"
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        lambda *_args, **_kwargs: RunProjection(
            run_id=promotion_projection_id,
            stage_bucket="core",
            cost_usd=0.0,
            completions=1,
            storage_bytes=10_000_000,
            basis={
                "method": "fixture",
                "run_call_limit": 1,
                "hosted_completion_cap_counted": True,
                "prompt_tokens": 10,
                "completion_tokens": 10,
            },
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        lambda *_args, **_kwargs: object(),
    )
    promotion_writer_flags: list[bool] = []
    huge_post_reconcile_message = "x" * (2 * 1024 * 1024)

    def fail_first_scientific_promotion(path, **kwargs):
        scientific = bool(kwargs["scientific_evidence"])
        promotion_writer_flags.append(scientific)
        if scientific:
            raise OSError(huge_post_reconcile_message)
        return real_terminal_writer(path, **kwargs)

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.write_terminal_summary",
        fail_first_scientific_promotion,
    )
    real_promotion_finalize = promotion_budget.finalize
    promotion_finalize_statuses: list[str] = []

    def observe_promotion_finalize(run_id: str, *, status: str, **kwargs):
        promotion_finalize_statuses.append(status)
        return real_promotion_finalize(run_id, status=status, **kwargs)

    monkeypatch.setattr(
        promotion_budget,
        "finalize",
        observe_promotion_finalize,
    )
    _execute_d_seed(
        contract,
        specs,
        raw_root=promotion_raw,
        paid=paid,
        diagnostic=False,
        budget_ledger=promotion_budget,
        run_ledger=promotion_runs,
    )
    assert promotion_writer_flags == [False] * len(specs) + [True]
    assert promotion_finalize_statuses == ["complete"]
    assert (
        promotion_budget.snapshot()["runs"][promotion_projection_id]["status"]
        == "complete"
    )
    promotion_state = promotion_runs.snapshot()["runs"]
    assert {
        promotion_state[spec.run_id]["status"] for spec in specs
    } == {"integrity-stopped"}
    assert {
        promotion_state[spec.run_id]["failure"]["error_type"]
        for spec in specs
    } == {"PostReconciliationPublicationFailure"}
    assert not tuple(
        (promotion_raw / "experiment-d" / "summaries").glob("*.json")
    )
    promotion_artifact = Path(
        promotion_state[specs[0].run_id]["artifact"]
    )
    promotion_receipt = verify_failure_receipt(promotion_artifact.parent)
    assert (
        promotion_receipt["scope"]
        == "finevo-pilot/experiment-d/post-reconciliation-publication"
    )
    assert promotion_receipt["provenance"]["scientific_evidence"] is False
    assert promotion_receipt["error"]["message_bytes"] == len(
        huge_post_reconcile_message.encode("utf-8")
    )
    assert promotion_receipt["error"]["message_sha256"] == hashlib.sha256(
        huge_post_reconcile_message.encode("utf-8")
    ).hexdigest()
    assert promotion_receipt["error"]["message_truncated"] is True
    assert len(
        promotion_receipt["error"]["message"].encode("utf-8")
    ) <= 2048
    promotion_group = (
        promotion_raw
        / "experiment-d"
        / "checkpoints"
        / f"s{seed}"
    )
    promotion_group_size = sum(
        path.stat().st_size
        for path in promotion_group.rglob("*")
        if path.is_file()
    )
    promotion_budget_row = promotion_budget.snapshot()["runs"][
        promotion_projection_id
    ]
    assert (
        promotion_group_size
        <= promotion_budget_row["actual"]["storage_bytes"]
    )
    for source_name in ("continuations.json", "narratives.json"):
        source = json.loads(
            (promotion_group / source_name).read_text(encoding="utf-8")
        )
        assert source["scientific_evidence"] is False


def test_remaining_core_projection_covers_a_b_c_and_d_once_per_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    ledger = PilotRunLedger(
        tmp_path / "run-ledger.json",
        contract_hash=contract.canonical_hash,
    )
    ledger.register(contract.expand())
    d_calls: list[int] = []

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.projection_from_preflight",
        lambda _contract, spec, **kwargs: RunProjection(
            run_id=spec.run_id,
            stage_bucket="core",
            cost_usd=0.01,
            completions=1,
            storage_bytes=100,
            basis={
                "method": "fixture",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
        ),
    )

    def d_projection(_contract, representative, **_kwargs):
        d_calls.append(representative.environment_seed)
        return RunProjection(
            run_id=f"d-group-{representative.environment_seed}",
            stage_bucket="core",
            cost_usd=0.02,
            completions=2,
            storage_bytes=200,
            basis={
                "method": "fixture-d",
                "prompt_tokens": 2,
                "completion_tokens": 2,
            },
        )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        d_projection,
    )
    projections = _remaining_core_projections(
        contract,
        raw_root=tmp_path,
        paid=paid,
        run_ledger=ledger,
    )
    assert len(projections) == 65
    assert len(d_calls) == 5
    assert len(set(d_calls)) == 5
    assert len([row for row in projections if row.run_id.startswith("d-group-")]) == 5
    offline_ids = {
        spec.run_id
        for spec in contract.expand(stage="experiment-c")
        if spec.execution_mode == "offline_candidate_admission"
    }
    assert {
        row.run_id for row in projections if row.completions == 0
    } == offline_ids

    completed = contract.expand(stage="experiment-a")[0]
    ledger.finalize(completed.run_id, status="complete", artifact="fixture")
    assert len(
        _remaining_core_projections(
            contract,
            raw_root=tmp_path,
            paid=paid,
            run_ledger=ledger,
        )
    ) == 64


def test_pre_scientific_projection_covers_stage0_core_and_only_capable_cross_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    ledger = PilotRunLedger(
        tmp_path / "run-ledger.json",
        contract_hash=contract.canonical_hash,
    )
    ledger.register(contract.expand())
    capable = {"gpt52_main", "gpt56_upper"}
    for spec in contract.expand(stage="capability-preflight"):
        ledger.finalize(
            spec.run_id,
            status=(
                "complete"
                if spec.model_id in capable
                else "capability-no-go"
            ),
            artifact=None,
        )
    for spec in contract.expand(stage="cross-model-sentinels"):
        if spec.model_id not in capable:
            ledger.finalize(
                spec.run_id,
                status="capability-no-go",
                artifact=None,
            )

    projected_specs = []

    def projection(_contract, spec, **_kwargs):
        projected_specs.append(spec)
        return RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=0.01,
            completions=1,
            storage_bytes=100,
            basis={
                "method": "fixture",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
        )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.projection_from_preflight",
        projection,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        lambda _contract, representative, **_kwargs: RunProjection(
            run_id=f"d-group-{representative.environment_seed}",
            stage_bucket="core",
            cost_usd=0.02,
            completions=2,
            storage_bytes=200,
            basis={
                "method": "fixture-d",
                "prompt_tokens": 2,
                "completion_tokens": 2,
            },
        ),
    )

    projections = _remaining_scientific_projections(
        contract,
        raw_root=tmp_path,
        paid=paid,
        run_ledger=ledger,
    )

    # 14 Stage-0 + 60 A/B/C cells + five shared D seed groups +
    # six GPT-5.6 full/no-memory cross-model cells.
    assert len(projections) == 85
    assert {
        spec.stage_id
        for spec in projected_specs
    } == {
        "stage0-calibration",
        "experiment-a",
        "experiment-b",
        "experiment-c",
        "cross-model-sentinels",
    }
    projected_cross = [
        spec
        for spec in projected_specs
        if spec.stage_id == "cross-model-sentinels"
    ]
    assert len(projected_cross) == 6
    assert {spec.model_id for spec in projected_cross} == {"gpt56_upper"}
    offline_ids = {
        spec.run_id
        for spec in contract.expand(stage="experiment-c")
        if spec.execution_mode == "offline_candidate_admission"
    }
    assert {
        row.run_id for row in projections if row.completions == 0
    } == offline_ids


def test_nonterminal_cross_cell_without_capability_pass_is_not_silently_dropped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    ledger = PilotRunLedger(
        tmp_path / "run-ledger.json",
        contract_hash=contract.canonical_hash,
    )
    ledger.register(contract.expand())
    for spec in contract.expand(stage="capability-preflight"):
        ledger.finalize(
            spec.run_id,
            status=(
                "complete"
                if spec.model_id == "gpt52_main"
                else "capability-no-go"
            ),
            artifact=None,
        )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.projection_from_preflight",
        _zero_projection,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        lambda _contract, representative, **_kwargs: RunProjection(
            run_id=f"d-group-{representative.environment_seed}",
            stage_bucket="core",
            cost_usd=0.0,
            completions=1,
            storage_bytes=1,
            basis={
                "method": "fixture-d",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
        ),
    )

    with pytest.raises(PilotOrchestrationError, match="lacks a completed"):
        _remaining_scientific_projections(
            contract,
            raw_root=tmp_path,
            paid=paid,
            run_ledger=ledger,
        )


def test_hosted_completion_cap_excludes_local_and_scripted_calls_but_keeps_run_limits() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    hosted = contract.expand(
        stage="capability-preflight",
        model="gpt52_main",
    )[0]
    local = contract.expand(
        stage="capability-preflight",
        model="llama33_local_sentinel",
    )[0]
    qref = contract.expand(stage="q-ref-resolution")[0]

    hosted_projection = conservative_projection(contract, hosted)
    local_projection = conservative_projection(contract, local)
    qref_projection = conservative_projection(contract, qref)

    assert hosted_projection.completions == 46
    assert hosted_projection.basis["run_call_limit"] == 46
    assert hosted_projection.basis["hosted_completion_cap_counted"] is True
    assert local_projection.completions == 0
    assert local_projection.basis["run_call_limit"] == 46
    assert local_projection.basis["hosted_completion_cap_counted"] is False
    assert qref_projection.completions == 0
    assert qref_projection.basis["run_call_limit"] == 48
    assert qref_projection.basis["hosted_completion_cap_counted"] is False
    assert _run_budget_from_projection(local_projection).limits.max_calls == 46
    assert _run_budget_from_projection(qref_projection).limits.max_calls == 48


def test_p95_projection_counts_only_hosted_calls_globally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    payload = {
        "projection": {
            "served::action": {
                "reserved_p95": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "cost_usd": 0.0,
                }
            },
            "served::semantic": {
                "reserved_p95": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "total_tokens": 24,
                    "cost_usd": 0.0,
                }
            },
        }
    }
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._load_verified_projection",
        lambda *args, **kwargs: (payload, tmp_path / "projection.json"),
    )
    local = contract.expand(
        stage="cross-model-sentinels",
        model="llama33_local_sentinel",
        arm="full",
    )[0]
    hosted = contract.expand(
        stage="cross-model-sentinels",
        model="gpt56_upper",
        arm="full",
    )[0]

    local_projection = projection_from_preflight(
        contract,
        local,
        raw_root=tmp_path,
    )
    hosted_projection = projection_from_preflight(
        contract,
        hosted,
        raw_root=tmp_path,
    )
    assert local_projection.completions == 0
    assert local_projection.basis["run_call_limit"] == 64
    assert hosted_projection.completions == 64
    assert hosted_projection.basis["run_call_limit"] == 64


def test_real_stage_lock_blocks_second_process_window_and_releases_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def body(**kwargs):
        calls.append(kwargs["stage_id"])
        if len(calls) == 1:
            raise RuntimeError("fixture body failure")
        return {"status": "complete"}

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._execute_stage_locked",
        body,
    )
    raw = tmp_path / "raw"
    with _exclusive_real_stage_lock(raw, stage_id="holder"):
        with pytest.raises(PilotOrchestrationError, match="execution lock"):
            execute_stage(
                contract_path=CONTRACT_PATH,
                stage_id="stage0-calibration",
                resume=True,
                raw_root=raw,
                repo_root=ROOT,
            )
    assert calls == []

    with pytest.raises(RuntimeError, match="fixture body failure"):
        execute_stage(
            contract_path=CONTRACT_PATH,
            stage_id="stage0-calibration",
            resume=True,
            raw_root=raw,
            repo_root=ROOT,
        )
    assert execute_stage(
        contract_path=CONTRACT_PATH,
        stage_id="stage0-calibration",
        resume=True,
        raw_root=raw,
        repo_root=ROOT,
    ) == {"status": "complete"}
    assert calls == ["stage0-calibration", "stage0-calibration"]


def test_stage0_projects_every_scientific_stage_before_first_dispatch_and_stops_globally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    raw = tmp_path / "raw"
    run_ledger = PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    run_ledger.register(contract.expand())
    for spec in contract.expand(stage="capability-preflight"):
        run_ledger.finalize(
            spec.run_id,
            status="complete",
            artifact=None,
        )
    _write_json(
        raw / "capability-preflight" / "stage_receipt.json",
        {
            "schema_version": PILOT_STAGE_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "stage_id": "capability-preflight",
            "status": "complete",
            "terminal": True,
            "registered_run_count": 6,
            "complete_cell_count": 6,
            "status_counts": {"complete": 6},
            "go": True,
            "go_models": [
                spec.model_id
                for spec in contract.expand(stage="capability-preflight")
            ],
        },
    )
    _write_json(
        raw / "q-ref-resolution" / "stage_receipt.json",
        {
            "schema_version": PILOT_STAGE_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "stage_id": "q-ref-resolution",
            "status": "complete",
            "terminal": True,
            "registered_run_count": 1,
            "complete_cell_count": 1,
            "status_counts": {"complete": 1},
            "go": True,
            "go_models": [],
        },
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_paid_provenance",
        lambda *args, **kwargs: paid,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        lambda _contract, *, model_ids: {
            "schema_version": PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "status": "pass",
            "paid_completions": 0,
            "rows": [{"model_id": model_ids[0], "status": "pass"}],
        },
    )
    observed = []

    def over_cap(*args, **kwargs):
        observed.append("full-matrix-projected")
        return (
            RunProjection(
                run_id="all-remaining-scientific",
                stage_bucket="calibration",
                cost_usd=3.01,
                completions=7_470,
                storage_bytes=1,
                basis={
                    "method": "full-scientific-fixture",
                    "run_call_limit": 7_470,
                    "hosted_completion_cap_counted": True,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                },
            ),
        )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._remaining_scientific_projections",
        over_cap,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._execute_actor_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("scientific dispatch occurred before full-matrix gate")
        ),
    )

    with pytest.raises(PilotOrchestrationError, match="full stage projection"):
        execute_stage(
            contract_path=CONTRACT_PATH,
            stage_id="stage0-calibration",
            resume=True,
            raw_root=raw,
            repo_root=ROOT,
        )
    assert observed == ["full-matrix-projected"]
    ledger = json.loads((raw / "run_ledger.json").read_text(encoding="utf-8"))
    scientific_statuses = {
        row["status"]
        for row in ledger["runs"].values()
        if row["spec"]["stage_id"]
        in {
            "stage0-calibration",
            "experiment-a",
            "experiment-b",
            "experiment-c",
            "experiment-d",
            "cross-model-sentinels",
        }
    }
    assert scientific_statuses == {"budget-stopped"}
    receipt = json.loads(
        (raw / "stage0-calibration" / "stage_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "budget-stopped"
    assert receipt["failure"]["projection_scope"] == (
        "all-remaining-stage0-a-b-c-d-and-capability-eligible-cross"
    )


def test_experiment_a_projects_all_remaining_core_before_any_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    raw = tmp_path / "raw"
    _write_json(
        raw / "stage0-calibration" / "stage_receipt.json",
        {
            "schema_version": PILOT_STAGE_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "stage_id": "stage0-calibration",
            "status": "complete",
            "terminal": True,
            "registered_run_count": 14,
            "complete_cell_count": 14,
            "status_counts": {"complete": 14},
            "go": True,
            "go_models": [],
        },
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_paid_provenance",
        lambda *args, **kwargs: paid,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        lambda _contract, *, model_ids: {
            "schema_version": PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "status": "pass",
            "paid_completions": 0,
            "rows": [{"model_id": model_ids[0], "status": "pass"}],
        },
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._remaining_core_projections",
        lambda *args, **kwargs: (
            RunProjection(
                run_id="all-remaining-core",
                stage_bucket="core",
                cost_usd=13.01,
                completions=1,
                storage_bytes=1,
                basis={
                    "method": "full-core-fixture",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                },
            ),
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._execute_actor_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider dispatch occurred before the full-core gate")
        ),
    )

    with pytest.raises(PilotOrchestrationError, match="full stage projection"):
        execute_stage(
            contract_path=CONTRACT_PATH,
            stage_id="experiment-a",
            resume=True,
            raw_root=raw,
            repo_root=ROOT,
        )
    ledger = json.loads((raw / "run_ledger.json").read_text(encoding="utf-8"))
    core_statuses = {
        row["status"]
        for row in ledger["runs"].values()
        if row["spec"]["stage_id"] in {
            "experiment-a",
            "experiment-b",
            "experiment-c",
            "experiment-d",
        }
    }
    assert core_statuses == {"budget-stopped"}
    assert json.loads(
        (raw / "experiment-a" / "stage_receipt.json").read_text(
            encoding="utf-8"
        )
    )["go"] is False


def test_budget_overage_records_observed_actual_before_itt_completion(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="experiment-a")[0]
    projection = RunProjection(
        run_id=spec.run_id,
        stage_bucket="core",
        cost_usd=0.01,
        completions=1,
        storage_bytes=1,
        basis={
            "method": "tiny-overage-fixture",
            "prompt_tokens": 10,
            "completion_tokens": 10,
        },
    )
    budget_ledger = PilotBudgetLedger(
        tmp_path / "budget.json",
        contract_hash=contract.canonical_hash,
    )
    budget_ledger.reserve(projection)
    run_budget = _run_budget_from_projection(projection)
    reservation = run_budget.reserve_call(
        estimated_usage=UsageRecord(1, 1, 0.005),
        label="fixture",
        model="fixture/model",
    )
    with pytest.raises(BudgetExceeded):
        run_budget.complete_call(
            reservation,
            UsageRecord(1, 1, 0.02),
        )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "observed.bin").write_bytes(b"0123456789")

    status, failure, actual = _finalize_budget_safely(
        budget_ledger,
        projection,
        run_dir=run_dir,
        budget=run_budget,
        status="complete",
    )
    assert status == "integrity-stopped"
    assert failure is not None
    assert actual == {
        "cost_usd": 0.02,
        "completions": 1,
        "storage_bytes": 10,
    }
    budget_row = budget_ledger.snapshot()["runs"][spec.run_id]
    assert budget_row["status"] == "integrity-stopped"
    assert budget_row["actual"] == actual
    assert budget_row["failure"]["observed_actual"] == actual
    assert budget_row["actual"] != {
        "cost_usd": projection.cost_usd,
        "completions": projection.completions,
        "storage_bytes": projection.storage_bytes,
    }

    run_ledger = PilotRunLedger(
        tmp_path / "runs.json",
        contract_hash=contract.canonical_hash,
    )
    run_ledger.register((spec,))
    run_ledger.finalize(
        spec.run_id,
        status=status,
        artifact=None,
        failure=failure,
    )
    assert run_ledger.status(spec.run_id) == "integrity-stopped"


def test_resume_never_redispatches_when_budget_finalized_before_itt(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="experiment-a")[0]
    projection = _zero_projection(contract, spec)
    budget = PilotBudgetLedger(
        tmp_path / "budget.json",
        contract_hash=contract.canonical_hash,
    )
    budget.reserve(projection)
    budget.finalize(
        spec.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=0,
    )
    runs = PilotRunLedger(
        tmp_path / "runs.json",
        contract_hash=contract.canonical_hash,
    )
    runs.register((spec,))
    assert _recover_or_stop_interrupted_reservation(budget, runs, spec) is True
    row = runs.snapshot()["runs"][spec.run_id]
    assert row["status"] == "integrity-stopped"
    assert row["failure"]["error_type"] == "BudgetFinalizedBeforeITT"


def test_projection_reader_recomputes_p95_and_rejects_resealed_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    cap_spec = contract.expand(
        stage="capability-preflight",
        model="gpt52_main",
    )[0]
    target = contract.expand(
        stage="experiment-a",
        model="gpt52_main",
        arm="full",
    )[0]
    run_dir = (
        tmp_path
        / "capability-preflight"
        / "runs"
        / cap_spec.run_id
    )
    manifest = run_dir / "preflight" / "manifest.json"
    capability_path = run_dir / "capability.json"
    provenance = manifest.parent / "provenance.json"
    served = contract.provider_profiles["gpt52_main"].served_model
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cost_usd": 0.01,
    }
    capability = {
        "contract_sha256": contract.canonical_hash,
        "run_spec": cap_spec.to_dict(),
        "rows": [
            {
                "category": "utility-ranking",
                "served_model": served,
                "usage": usage,
            },
            {
                "category": "rule-proposal",
                "served_model": served,
                "usage": usage,
            },
        ],
    }
    _write_json(capability_path, capability)
    _write_json(
        manifest,
        {"git": {"commit": paid.head_commit, "dirty": False}},
    )
    _write_json(
        provenance,
        {
            "details": {
                "contract_sha256": contract.canonical_hash,
                "run_spec": cap_spec.to_dict(),
            }
        },
    )

    class Result:
        @staticmethod
        def stream(name: str):
            if name != "api_usage":
                raise KeyError(name)
            return (
                {
                    "response_model": served,
                    "call_kind": "action",
                    "usage": usage,
                },
                {
                    "response_model": served,
                    "call_kind": "semantic",
                    "usage": usage,
                },
            )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_manifest",
        lambda path: SimpleNamespace(
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.load_verified_run_artifacts",
        lambda path: Result(),
    )
    source_rows = [
        {
            "response_model": served,
            "call_kind": (
                "capability-proposal"
                if row["category"] == "rule-proposal"
                else "capability-choice"
            ),
            "usage": row["usage"],
        }
        for row in capability["rows"]
    ] + list(Result.stream("api_usage"))
    projection = preflight_p95(source_rows, reserve_multiplier=1.25)
    payload = _seal_bound_payload(
        {
            "schema_version": PILOT_PROJECTION_SCHEMA_VERSION,
            "model_id": "gpt52_main",
            "served_model": served,
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": paid.git_tag,
                "git_commit": paid.head_commit,
                "source_manifest": str(manifest),
                "source_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "source_capability": str(capability_path),
                "source_capability_sha256": hashlib.sha256(
                    capability_path.read_bytes()
                ).hexdigest(),
            },
            "projection": projection,
        }
    )
    projection_path = run_dir / "projection_p95.json"
    _write_json(projection_path, payload)
    assert projection_from_preflight(
        contract,
        target,
        raw_root=tmp_path,
        paid=paid,
    ).completions > 0

    key = f"{served}::action"
    payload["projection"][key]["reserved_p95"]["cost_usd"] += 1.0
    _write_json(projection_path, _seal_bound_payload(payload))
    with pytest.raises(PilotOrchestrationError, match="sealed api_usage p95"):
        projection_from_preflight(
            contract,
            target,
            raw_root=tmp_path,
            paid=paid,
        )


def test_qref_recompute_and_stage0_exact_7x2_binding_reject_tamper(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    raw = tmp_path / "raw"
    spec = contract.expand(stage="q-ref-resolution")[0]
    projection = RunProjection(
        run_id=spec.run_id,
        stage_bucket="calibration",
        cost_usd=0.0,
        completions=48,
        storage_bytes=20_000_000,
        basis={
            "method": "scripted-qref",
            "prompt_tokens": 500_000,
            "completion_tokens": 100_000,
        },
    )
    _execute_q_ref(
        contract,
        spec,
        raw_root=raw,
        paid=paid,
        projection=projection,
    )
    valid = _load_verified_q_ref(contract, raw_root=raw, paid=paid)
    qref_path = raw / "q-ref-resolution" / "q_ref_resolution.json"
    original = json.loads(qref_path.read_text(encoding="utf-8"))
    tampered = json.loads(json.dumps(original))
    tampered["q_ref"] += 1.0
    qref_path.chmod(0o644)
    _write_json(qref_path, _seal_bound_payload(tampered))
    with pytest.raises(PilotOrchestrationError, match="sealed runner source"):
        _load_verified_q_ref(contract, raw_root=raw, paid=paid)

    _write_json(qref_path, original)
    selection = _seal_bound_payload(
        {
            "schema_version": "finevo-stage0-selection-v1",
            "selected_profile_id": "center",
            "selected_utility": {},
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": paid.git_tag,
                "git_commit": paid.head_commit,
                "q_ref_content_sha256": valid["integrity"]["content_sha256"],
                "q_ref_file_sha256": hashlib.sha256(
                    qref_path.read_bytes()
                ).hexdigest(),
                # A resealed selection still fails if one of the exact
                # preregistered 7x2 source manifests is missing.
                "source_manifests": [{} for _ in range(13)],
            },
        }
    )
    _write_json(
        raw / "stage0-calibration" / "stage0_selection.json",
        selection,
    )
    with pytest.raises(PilotOrchestrationError, match="exact 7x2"):
        _load_verified_stage0_selection(contract, raw_root=raw, paid=paid)


def test_stage0_absolute_threshold_uses_only_selected_two_seed_ledgers() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    selected_specs = tuple(
        spec
        for spec in contract.expand(stage="stage0-calibration")
        if spec.utility_profile_id == "center"
    )

    class Result:
        def __init__(self, rows):
            self._rows = rows

        def stream(self, name: str):
            assert name == "utility_ledger"
            return tuple(self._rows)

    sources = []
    for source_index, spec in enumerate(selected_specs):
        rows = [
            {
                "period": period,
                "agent_id": agent_id,
                "flow_utility": float(source_index * 2 + period / 100),
            }
            for period in range(12)
            for agent_id in range(4)
        ]
        sources.append(
            (
                spec,
                Result(rows),
                {
                    "manifest": f"/sealed/{spec.run_id}/manifest.json",
                    "manifest_sha256": str(source_index + 1) * 64,
                },
            )
        )
    threshold = _derive_stage0_absolute_flow_threshold(
        contract,
        selected_profile_id="center",
        sources=sources,
    )
    assert threshold["row_count"] == 96
    assert threshold["source_seeds"] == list(
        contract.seeds["sets"]["calibration"]
    )
    assert len(threshold["source_manifests"]) == 2
    assert threshold["treatment_outcomes_inspected"] is False

    changed_rows = list(sources[0][1]._rows)
    changed_rows[0] = {**changed_rows[0], "flow_utility": 99.0}
    changed_sources = [
        (sources[0][0], Result(changed_rows), sources[0][2]),
        sources[1],
    ]
    recomputed = _derive_stage0_absolute_flow_threshold(
        contract,
        selected_profile_id="center",
        sources=changed_sources,
    )
    original_hashes = {
        row["run_id"]: row["utility_ledger_sha256"]
        for row in threshold["source_manifests"]
    }
    recomputed_hashes = {
        row["run_id"]: row["utility_ledger_sha256"]
        for row in recomputed["source_manifests"]
    }
    assert (
        recomputed_hashes[sources[0][0].run_id]
        != original_hashes[sources[0][0].run_id]
    )
    assert recomputed["source_matrix_sha256"] != threshold["source_matrix_sha256"]


def test_stage0_consumer_recomputes_and_rejects_resealed_threshold_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    raw = tmp_path / "raw"
    qref_path = raw / "q-ref-resolution" / "q_ref_resolution.json"
    qref = {
        "q_ref": 1.0,
        "integrity": {"content_sha256": "a" * 64},
    }
    _write_json(qref_path, qref)
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._load_verified_q_ref",
        lambda *args, **kwargs: qref,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.summarize_run",
        lambda *args, **kwargs: {},
    )
    base_selection = {
        "schema_version": "finevo-stage0-selection-v1",
        "selected_profile_id": "center",
        "selected_utility": {"consumption_scale": 1.0},
    }
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.select_stage0_profile",
        lambda *args, **kwargs: dict(base_selection),
    )

    class Result:
        def __init__(self, spec):
            self.config = {
                "run_id": spec.run_id,
                "seed": spec.environment_seed,
                "max_labor_hours": 168.0,
            }
            self.records = {}
            self._rows = [
                {
                    "period": period,
                    "agent_id": agent_id,
                    "flow_utility": (
                        float(period)
                        + float(spec.environment_seed % 10) / 10.0
                    ),
                }
                for period in range(12)
                for agent_id in range(4)
            ]

        def stream(self, name: str):
            assert name == "utility_ledger"
            return tuple(self._rows)

    results: dict[str, Result] = {}
    source_manifests = []
    for spec in contract.expand(stage="stage0-calibration"):
        run_dir = raw / "stage0-calibration" / "runs" / spec.run_id
        manifest_path = run_dir / "manifest.json"
        _write_json(
            manifest_path,
            {"git": {"commit": paid.head_commit, "dirty": False}},
        )
        _write_json(
            run_dir / "provenance.json",
            {
                "details": {
                    "contract_sha256": contract.canonical_hash,
                    "run_spec": spec.to_dict(),
                }
            },
        )
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        source_manifests.append(
            {
                "run_id": spec.run_id,
                "utility_profile_id": spec.utility_profile_id,
                "environment_seed": spec.environment_seed,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_hash,
            }
        )
        results[spec.run_id] = Result(spec)

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_manifest",
        lambda run_dir: SimpleNamespace(
            manifest_sha256=hashlib.sha256(
                (Path(run_dir) / "manifest.json").read_bytes()
            ).hexdigest()
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.load_verified_run_artifacts",
        lambda run_dir: results[Path(run_dir).name],
    )
    sources = [
        (
            spec,
            results[spec.run_id],
            next(
                row
                for row in source_manifests
                if row["run_id"] == spec.run_id
            ),
        )
        for spec in contract.expand(stage="stage0-calibration")
    ]
    threshold = _derive_stage0_absolute_flow_threshold(
        contract,
        selected_profile_id="center",
        sources=sources,
    )
    selection = _seal_bound_payload(
        {
            **base_selection,
            "absolute_flow_utility_threshold": threshold,
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": paid.git_tag,
                "git_commit": paid.head_commit,
                "q_ref_content_sha256": qref["integrity"]["content_sha256"],
                "q_ref_file_sha256": hashlib.sha256(
                    qref_path.read_bytes()
                ).hexdigest(),
                "source_manifests": sorted(
                    source_manifests,
                    key=lambda row: row["run_id"],
                ),
            },
        }
    )
    selection_path = raw / "stage0-calibration" / "stage0_selection.json"
    _write_json(selection_path, selection)
    verified = _load_verified_stage0_selection(
        contract,
        raw_root=raw,
        paid=paid,
    )
    assert verified["absolute_flow_utility_threshold"] == threshold

    tampered = json.loads(json.dumps(selection))
    tampered["absolute_flow_utility_threshold"]["value"] += 1.0
    _write_json(selection_path, _seal_bound_payload(tampered))
    with pytest.raises(PilotOrchestrationError, match="sealed sources"):
        _load_verified_stage0_selection(
            contract,
            raw_root=raw,
            paid=paid,
        )


def _complete_episode_rows(run_id: str, seed: int) -> list[dict]:
    rows: list[dict] = []
    for agent_id in range(4):
        track = EvidenceLinkedEpisodicTrack(
            run_id=run_id,
            seed=seed,
            agent_id=agent_id,
        )
        for decision_t in range(12):
            decision_id = track.begin_episode(
                decision_t=decision_t,
                pre_state={"interest_rate": 0.03},
                context_id=f"context-{agent_id}-{decision_t}",
                context_vector=(0.03, float(decision_t)),
                retrieved_episode_ids=(),
                selected_rule_ids=(),
                proposed_action={
                    "labor_hours": 8.0,
                    "consumption_fraction": 0.5,
                },
                executed_action={
                    "labor_hours": 8.0,
                    "consumption_fraction": 0.5,
                },
            )
            rows.append(
                track.finalize_episode(
                    decision_id,
                    outcome_t=decision_t + 1,
                    next_state={"interest_rate": 0.03},
                    outcome={"wealth_change": 0.0},
                    reward=float(decision_t),
                    flow_utility=float(decision_t),
                ).to_dict()
            )
    return rows


def test_c_sensitivity_is_zero_api_recomputed_and_resealed_tamper_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    raw = tmp_path / "raw"
    selection_path = raw / "stage0-calibration" / "stage0_selection.json"
    _write_json(selection_path, {"fixture": "stage0"})
    stage0 = {
        "absolute_flow_utility_threshold": {
            "value": 5.5,
            "field": "flow_utility",
            "aggregation": "median",
            "source_matrix_sha256": "a" * 64,
        },
        "integrity": {"content_sha256": "b" * 64},
    }
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._load_verified_stage0_selection",
        lambda *args, **kwargs: stage0,
    )

    results: dict[str, object] = {}
    for spec in contract.expand(
        stage="experiment-b",
        model="gpt52_main",
        arm="full",
    ):
        run_dir = raw / "experiment-b" / "runs" / spec.run_id
        _write_json(
            run_dir / "manifest.json",
            {"git": {"commit": paid.head_commit, "dirty": False}},
        )
        _write_json(
            run_dir / "provenance.json",
            {
                "details": {
                    "contract_sha256": contract.canonical_hash,
                    "run_spec": spec.to_dict(),
                }
            },
        )
        episodes = _complete_episode_rows(
            spec.run_id,
            spec.environment_seed,
        )
        proposals = [
            {"agent_id": agent_id, "current_t": current_t}
            for current_t in (3, 6, 9, 12)
            for agent_id in range(4)
        ]

        class Result:
            def __init__(self):
                self.config = {
                    "run_id": spec.run_id,
                    "seed": spec.environment_seed,
                    "semantic_proposal_after": 3,
                    "semantic_proposal_interval": 3,
                    "min_candidate_support": 2,
                    "activation_min_support": 3,
                    "activation_min_margin": 1.0,
                    "activation_confidence_threshold": 0.60,
                    "retirement_patience": 2,
                    "retirement_confidence_threshold": 0.45,
                }
                self._streams = {
                    "episodes": episodes,
                    "semantic_proposals": proposals,
                    "semantic_rules": [],
                }

            def stream(self, name: str):
                return tuple(self._streams[name])

        results[spec.run_id] = Result()

    def verify_fixture(run_dir: Path):
        manifest = run_dir / "manifest.json"
        return SimpleNamespace(
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
        )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_manifest",
        verify_fixture,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.load_verified_run_artifacts",
        lambda run_dir: results[Path(run_dir).name],
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline sensitivity attempted provider construction")
        ),
    )

    output = _write_experiment_c_sensitivity(
        contract,
        raw_root=raw,
        paid=paid,
    )
    value = _load_verified_experiment_c_sensitivity(
        contract,
        raw_root=raw,
        paid=paid,
    )
    assert output == raw / "experiment-c" / "rule_sensitivity.json"
    assert value["provider_calls"] == 0
    assert value["descriptive_only"] is True
    assert value["effectiveness_gate"] is False
    assert value["source_run_count"] == 5
    assert len(value["aggregate_cells"]) == 9
    assert all(cell["source_run_count"] == 5 for cell in value["aggregate_cells"])

    sensitivity_control = {
        "pass": True,
        "provider_calls": 0,
        "grid_cell_count": 9,
        "content_sha256": value["integrity"]["content_sha256"],
    }
    package_root = tmp_path / "published-package"
    _, checksums_path, scientific_complete = _write_package_files(
        package_root,
        contract_path=CONTRACT_PATH,
        contract=contract,
        rows=[],
        denominator={
            "expected_count": len(contract.expand()),
            "observed_count": 0,
            "pass": False,
            "status_counts": {"failed": len(contract.expand())},
        },
        common_commit=paid.head_commit,
        gates={
            "experiment_a": {
                "status": "no-go",
                "claim_action": "retain route traceability only",
            },
            "experiment_c": {
                "status": "no-go",
                "claim_action": "withdraw rule-reliability claim",
            },
            "experiment_d": {
                "status": "no-go",
                "claim_action": "retain prompt sensitivity only",
            },
            "narrative": {
                "status": "no-go",
                "claim_boundary": "no semantic-response claim",
            },
        },
        capability={},
        cross_model={},
        release_controls={
            "pass": False,
            "release_attestation": {},
            "stage0_selection": {},
            "budget_ledger": {
                "raw_root_storage_bytes": sum(
                    path.stat().st_size
                    for path in raw.rglob("*")
                    if path.is_file()
                ),
            },
            "experiment_c_sensitivity": sensitivity_control,
        },
        rule_sensitivity=value,
    )
    published_sensitivity = (
        package_root / "experiment_c_rule_sensitivity.json"
    )
    assert json.loads(published_sensitivity.read_text(encoding="utf-8")) == value
    checksum_rows = json.loads(
        checksums_path.read_text(encoding="utf-8")
    )["files"]
    sensitivity_checksum = next(
        row
        for row in checksum_rows
        if row["path"] == "experiment_c_rule_sensitivity.json"
    )
    assert sensitivity_checksum["sha256"] == hashlib.sha256(
        published_sensitivity.read_bytes()
    ).hexdigest()
    report = (package_root / "reviewer_report.md").read_text(encoding="utf-8")
    assert value["integrity"]["content_sha256"] in report
    assert "Registered zero-API 3×3 sensitivity control" in report
    assert scientific_complete is False

    tampered = json.loads(json.dumps(value))
    tampered["aggregate_cells"][0]["ever_active_count"] = 999
    output.chmod(0o644)
    _write_json(output, _seal_bound_payload(tampered))
    with pytest.raises(PilotOrchestrationError, match="differs from sealed"):
        _load_verified_experiment_c_sensitivity(
            contract,
            raw_root=raw,
            paid=paid,
        )
