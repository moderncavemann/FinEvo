from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from scripts.render_pilot_v2116_contract import (
    _parse_with_bootstrap_design_pin,
    build_contract,
)
from verified_memory import pilot_contract as pilot_contract_module
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v2116_continuation as continuation
from verified_memory.pilot_budget import PilotBudgetError, PilotBudgetLedger
from verified_memory.pilot_contract import (
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_6.yaml"
PARENT_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"
SOURCE_MANIFEST_PATH = ROOT / "experiments" / "pilot_v2_11_6_source_manifest.json"
MAIN_SEEDS = (1099057501, 1421875452, 1769977770, 959809858, 617806385)
EXPECTED_D_PLAN_HASHES = {
    1099057501: "ca1cf08c31f0abbd66f638026604110dc4fb4d881269bdc61bc23449c06b8276",
    1421875452: "685d4401014c2bf2358eae785e369274c30cf19561409c571f993f536bcaf5a0",
    1769977770: "8f01895e3211d822ad5bf282faf9c939d7853fc12b0a18689f8d281cb9261486",
    959809858: "4cda0b534fdc0b93d77d43d724d6d36ecdbd5047fb567628a205e346be892eb6",
    617806385: "0fd8e4050a6498cc980691c5190845e2f26a0c615a786d477d470df628aac8f7",
}
EXPECTED_REVIEWED_CHANGED = {
    "_build_experiment_c_sensitivity",
    "_d_group_projection",
    "_execute_d_seed",
    "_execute_stage_locked",
    "_load_verified_experiment_c_sensitivity",
    "_load_verified_projection",
    "_load_verified_q_ref",
    "_load_verified_stage0_selection",
    "_observed_p95_authority_receipt_path",
    "_parent_budget_debit",
    "_runner_p95_reservations",
    "run_development_fake_matrix",
    "_v2_control_gate_ok",
    "_verified_observed_p95_binding",
    "_write_experiment_c_sensitivity",
}
EXPECTED_REVIEWED_NEW = {
    "_execute_v2116_parent_import_stage",
    "_load_verified_v2116_calibration",
    "_v2116_parent_import_projection",
    "build_v2116_experiment_d_group_plan",
}


def _contract(monkeypatch: pytest.MonkeyPatch) -> PilotContract:
    try:
        return load_pilot_contract(CONTRACT_PATH)
    except PilotContractError:
        document = build_contract(ROOT, status="draft")
        monkeypatch.setattr(
            pilot_contract_module,
            "PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256",
            science_design_sha256(document),
        )
        return PilotContract.from_dict(document)


def _reseal_contract(document: dict[str, Any]) -> None:
    document["integrity"]["declared_sha256"] = canonical_contract_sha256(document)


def _run_snapshot_before_science(contract: PilotContract) -> dict[str, Any]:
    parent_id = contract.expand(stage="parent-import")[0].run_id
    return {
        "runs": {
            spec.run_id: {
                "spec": spec.to_dict(),
                "status": "complete" if spec.run_id == parent_id else "scheduled",
            }
            for spec in contract.expand()
        }
    }


def _budget_snapshot_before_science(
    contract: PilotContract, *, parent_status: str = "complete"
) -> dict[str, Any]:
    parent_id = contract.expand(stage="parent-import")[0].run_id
    row: dict[str, Any] = {
        "status": parent_status,
        "stage_bucket": "parent_v2115",
        "reservation": orchestrator._v2116_parent_import_projection(
            contract.expand(stage="parent-import")[0]
        ).to_dict(),
    }
    if parent_status == "complete":
        row["actual"] = {
            "cost_usd": 0.0,
            "completions": 0,
            "storage_bytes": 1,
        }
    return {"runs": {parent_id: row}}


def test_tracked_source_manifest_is_a_non_null_honest_successor() -> None:
    assert (
        pilot_contract_module.PILOT_V2_11_6_SOURCE_MANIFEST_FILE_SHA256
        is not None
    )
    assert (
        pilot_contract_module.PILOT_V2_11_6_SOURCE_MANIFEST_CONTENT_SHA256
        is not None
    )
    payload = SOURCE_MANIFEST_PATH.read_bytes()
    manifest = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        pilot_contract_module.PILOT_V2_11_6_SOURCE_MANIFEST_FILE_SHA256
    )
    assert manifest["integrity"]["content_sha256"] == (
        pilot_contract_module.PILOT_V2_11_6_SOURCE_MANIFEST_CONTENT_SHA256
    )
    unsigned = deepcopy(manifest)
    unsigned["integrity"].pop("content_sha256")
    assert canonical_sha256(unsigned) == manifest["integrity"]["content_sha256"]

    equivalence = manifest["remaining_science_implementation_equivalence"]
    assert equivalence is not None
    assert equivalence["policy"] == (
        "science-core-equal-with-explicit-successor-adapter-v1"
    )
    assert equivalence["equivalence_claim"] == (
        "science_core_equal_with_explicit_successor_adapter"
    )
    assert equivalence["full_runtime_byte_identity_claimed"] is False
    assert {
        row["path"] for row in equivalence["byte_identical_files"]
    } == set(continuation._BYTE_IDENTICAL_SCIENCE_PATHS)
    assert {
        "verified_memory/actions.py",
        "verified_memory/m1_context.py",
        "verified_memory/prompts.py",
        "verified_memory/system.py",
    } <= {
        row["path"] for row in equivalence["byte_identical_files"]
    }
    assert set(equivalence["reviewed_changed_function_sha256"]) == (
        EXPECTED_REVIEWED_CHANGED
    )
    assert set(equivalence["reviewed_new_function_sha256"]) == EXPECTED_REVIEWED_NEW
    assert equivalence["removed_top_level_functions"] == []
    assert {
        row["seed"]: row["normalized_plan_sha256"]
        for row in equivalence["experiment_d_normalized_plan_receipts"]
    } == EXPECTED_D_PLAN_HASHES
    runtime = manifest["current_runtime_sources"]
    assert set(runtime["pilot_contract_reviewed_changed_node_sha256"]) == (
        continuation._REVIEWED_CHANGED_CONTRACT_NODES
    )
    assert set(runtime["pilot_contract_reviewed_new_node_sha256"]) == (
        continuation._REVIEWED_NEW_CONTRACT_NODES
    )
    assert runtime["pilot_contract_removed_top_level_nodes"] == []
    assert runtime["terminal_artifact_writer"]["parent_equal"] is True


def test_d_normalized_plans_match_v2115_for_all_five_frozen_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _contract(monkeypatch)
    parent = load_pilot_contract(PARENT_CONTRACT_PATH)

    rows = continuation._normalized_d_plan_receipts(child, parent)

    assert tuple(row["seed"] for row in rows) == MAIN_SEEDS
    assert {
        row["seed"]: row["normalized_plan_sha256"] for row in rows
    } == EXPECTED_D_PLAN_HASHES


def test_runtime_maps_exactly_all_86_parent_remaining_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _contract(monkeypatch)
    parent = load_pilot_contract(PARENT_CONTRACT_PATH)
    parent_specs = tuple(
        spec
        for stage in ("experiment-d", "experiment-b", "cross-model")
        for spec in parent.expand(stage=stage)
    )
    child_specs = tuple(
        spec
        for stage in ("experiment-d", "experiment-b", "cross-model")
        for spec in child.expand(stage=stage)
    )

    mapped = {
        continuation._normalize_parent_spec(spec.to_dict())["run_id"]:
        continuation._normalize_parent_spec(spec.to_dict())
        for spec in parent_specs
    }

    assert len(mapped) == len(child_specs) == 86
    assert mapped == {spec.run_id: spec.to_dict() for spec in child_specs}
    assert not any("--experiment-a--" in run_id for run_id in mapped)
    assert not any("--experiment-c--" in run_id for run_id in mapped)


@pytest.mark.parametrize("stage_id", ("experiment-a", "experiment-c"))
@pytest.mark.parametrize("field", ("file_sha256", "content_sha256"))
def test_resealed_parent_a_c_receipt_binding_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
    field: str,
) -> None:
    document = build_contract(ROOT, status="draft")
    document["v2116_continuation_boundary"]["parent_stage_receipts"][stage_id][
        field
    ] = "0" * 64
    _reseal_contract(document)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256",
        science_design_sha256(document),
    )

    with pytest.raises(PilotContractError, match="continuation boundary drifted"):
        PilotContract.from_dict(document)


def test_parent_import_projection_is_exactly_zero_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    spec = contract.expand(stage="parent-import")[0]

    projection = orchestrator._v2116_parent_import_projection(spec)

    assert projection.run_id == spec.run_id
    assert projection.stage_bucket == "parent_v2115"
    assert (projection.cost_usd, projection.completions) == (0.0, 0)
    assert projection.basis["provider_construction"] is False
    assert projection.basis["provider_calls"] == 0
    assert projection.basis["mapped_parent_scheduled_cells"] == 86
    assert projection.basis["imported_parent_terminal_cells_as_child_rows"] == 0
    assert projection.basis["imported_effect_cells"] == 0


def test_immutable_write_once_handles_partial_os_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "sealed.json"
    value = continuation._seal({"payload": "partial-write-loop"})
    original_write = continuation.os.write
    write_sizes: list[int] = []

    def partial_write(descriptor: int, payload: Any) -> int:
        bounded = payload[: min(3, len(payload))]
        written = original_write(descriptor, bounded)
        write_sizes.append(written)
        return written

    monkeypatch.setattr(continuation.os, "write", partial_write)
    continuation._write_once(target, value)

    expected = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    assert target.read_bytes() == expected
    assert len(write_sizes) > 1
    assert not (tmp_path / ".sealed.json.pending").exists()


def test_immutable_write_once_recovers_only_an_unpublished_stale_pending_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sealed.json"
    pending = tmp_path / ".sealed.json.pending"
    pending.write_bytes(b"stale-unpublished-bytes")
    value = continuation._seal({"payload": "fresh-exact-bytes"})

    continuation._write_once(target, value)

    assert json.loads(target.read_text(encoding="utf-8")) == value
    assert not pending.exists()


@pytest.mark.parametrize("collision", ("target", "target-symlink", "pending-symlink"))
def test_immutable_write_once_refuses_mismatched_or_symlink_collisions(
    tmp_path: Path,
    collision: str,
) -> None:
    target = tmp_path / "sealed.json"
    pending = tmp_path / ".sealed.json.pending"
    outside = tmp_path / "outside.json"
    outside.write_text('{"outside":true}\n', encoding="utf-8")
    if collision == "target":
        target.write_text('{"tampered":true}\n', encoding="utf-8")
        message = "immutable artifact drifted"
    elif collision == "target-symlink":
        target.symlink_to(outside)
        message = "immutable artifact drifted"
    else:
        pending.symlink_to(outside)
        message = "pending path drifted"

    with pytest.raises(continuation.PilotV2116ContinuationError, match=message):
        continuation._write_once(target, continuation._seal({"payload": "new"}))

    assert outside.read_text(encoding="utf-8") == '{"outside":true}\n'


def test_immutable_write_once_losing_publish_race_refuses_mismatched_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "sealed.json"
    pending = tmp_path / ".sealed.json.pending"

    def lose_race(_source: Any, destination: Any, **_kwargs: Any) -> None:
        Path(destination).write_text('{"raced":"mismatch"}\n', encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(continuation.os, "link", lose_race)
    with pytest.raises(
        continuation.PilotV2116ContinuationError,
        match="immutable artifact drifted",
    ):
        continuation._write_once(target, continuation._seal({"payload": "expected"}))

    assert target.read_text(encoding="utf-8") == '{"raced":"mismatch"}\n'
    assert not pending.exists()


def test_parent_import_crash_recovery_finalizes_only_surviving_budget_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    specs = contract.expand(stage="parent-import")
    spec = specs[0]
    raw_root = tmp_path / "raw"
    receipt_path = raw_root / "parent-import" / "stage_receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text("{}\n", encoding="utf-8")
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    run_ledger.finalize(
        spec.run_id,
        status="complete",
        artifact=str(raw_root / "parent-import/summaries/terminal.json"),
    )
    budget_ledger = PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=continuation.parent_budget_debit_for_v2116(contract),
    )
    projection = orchestrator._v2116_parent_import_projection(spec)
    budget_ledger.reserve(projection)
    verified = {"status": "complete", "go": True, "fixture": "reverified"}
    monkeypatch.setattr(
        orchestrator,
        "_verify_v2_stage_receipt",
        lambda *_args, **_kwargs: verified,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("crash recovery attempted source or provider work")

    monkeypatch.setattr(
        orchestrator, "build_v2116_parent_import_receipt", forbidden
    )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    monkeypatch.setattr(orchestrator, "MultiModelLLM", forbidden)
    monkeypatch.setattr(orchestrator, "validate_live_provider_catalog", forbidden)

    result = orchestrator._execute_v2116_parent_import_stage(
        contract,
        specs,
        raw_root=raw_root,
        repo_root=tmp_path,
        parent_repo_root=tmp_path / "unused-parent",
        paid=SimpleNamespace(),
        run_ledger=run_ledger,
        budget_ledger=budget_ledger,
    )

    assert result == verified
    row = budget_ledger.snapshot()["runs"][spec.run_id]
    assert row["status"] == "complete"
    assert row["actual"]["cost_usd"] == 0.0
    assert row["actual"]["completions"] == 0
    assert 0 < row["actual"]["storage_bytes"] <= 5_000_000


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            "reserved-parent",
            "parent-import budget row differs from its exact zero-provider projection",
        ),
        ("science-reservation", "contains a scientific reservation"),
    ),
)
def test_acceptance_rejects_reserved_parent_or_any_science_budget_prefix(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    contract = _contract(monkeypatch)
    run_snapshot = _run_snapshot_before_science(contract)
    budget_snapshot = _budget_snapshot_before_science(
        contract,
        parent_status="reserved" if mutation == "reserved-parent" else "complete",
    )
    if mutation == "science-reservation":
        science = contract.expand(stage="experiment-b")[0]
        budget_snapshot["runs"][science.run_id] = {
            "status": "reserved",
            "reservation": {
                "run_id": science.run_id,
                "stage_bucket": science.budget_bucket,
            },
        }

    with pytest.raises(continuation.PilotV2116ContinuationError, match=message):
        continuation._audit_acceptance_denominator(
            contract,
            run_snapshot,
            budget_snapshot,
        )


@pytest.mark.parametrize("kind", ("unexpected", "symlink"))
def test_acceptance_rejects_non_exact_pre_science_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    contract = _contract(monkeypatch)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    if kind == "unexpected":
        path = raw_root / "experiment-d" / "preplanted.json"
        path.parent.mkdir()
        path.write_text("{}\n", encoding="utf-8")
        message = "unexpected paths"
    else:
        target = tmp_path / "outside.json"
        target.write_text("{}\n", encoding="utf-8")
        (raw_root / "linked.json").symlink_to(target)
        message = "contains a symlink"

    with pytest.raises(continuation.PilotV2116ContinuationError, match=message):
        continuation._audit_pre_science_namespace(raw_root, contract)


def test_acceptance_rejects_exported_key_and_blocks_provider_constructors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in continuation._PROVIDER_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret-never-read")
    with pytest.raises(
        continuation.PilotV2116ContinuationError,
        match="before provider credentials are loaded",
    ):
        continuation.require_v2116_provider_keys_absent()
    monkeypatch.delenv("OPENAI_API_KEY")

    original_factory = orchestrator.create_llm_provider
    original_multi = orchestrator.MultiModelLLM
    original_catalog = orchestrator.validate_live_provider_catalog
    with continuation._acceptance_provider_sentinels():
        for call in (
            lambda: orchestrator.create_llm_provider(),
            lambda: orchestrator.MultiModelLLM(),
            lambda: orchestrator.validate_live_provider_catalog(),
        ):
            with pytest.raises(
                continuation.PilotV2116ContinuationError,
                match="forbidden during acceptance",
            ):
                call()
    assert orchestrator.create_llm_provider is original_factory
    assert orchestrator.MultiModelLLM is original_multi
    assert orchestrator.validate_live_provider_catalog is original_catalog


def test_acceptance_rejects_resealed_but_tampered_ledger_marker() -> None:
    receipt = continuation._seal(
        {
            "schema_version": continuation.V2116_ACCEPTANCE_SCHEMA_VERSION,
            "ledger_prefixes": {
                "run_ledger": {
                    "event_count": 1,
                    "event_chain_head": "1" * 64,
                },
                "budget_ledger": {
                    "event_count": 1,
                    "event_chain_head": "2" * 64,
                },
            },
        }
    )
    prefix = {
        "event_count": 1,
        "event_chain_head": "1" * 64,
        "ledger_sha256": "3" * 64,
        "runs_sha256": "4" * 64,
    }
    snapshot = {
        "events": [
            {"event_sha256": "1" * 64},
            {
                "event_type": "acceptance_receipt_bound",
                "payload": {"tampered": True},
            },
        ]
    }

    with pytest.raises(
        continuation.PilotV2116ContinuationError,
        match="acceptance ledger marker drifted",
    ):
        continuation._verify_acceptance_marker(
            snapshot,
            prefix=prefix,
            receipt=receipt,
            receipt_path=(
                "experiment_results/pilot-v2.11.6/raw/"
                "scientific_dispatch_acceptance.json"
            ),
            budget=False,
        )


def test_v2116_has_no_capability_or_preflight_dispatch_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)

    assert contract.stage_ids == (
        "parent-import",
        "experiment-d",
        "experiment-b",
        "cross-model",
    )
    assert orchestrator._scientific_stage_ids(contract) == (
        "experiment-d",
        "experiment-b",
        "cross-model",
    )
    assert all(
        not orchestrator._is_capability_stage(contract, stage_id)
        for stage_id in contract.stage_ids
    )
    for model_id in contract.model_ids:
        with pytest.raises(
            orchestrator.PilotOrchestrationError,
            match="lacks one exact projection-producing preflight",
        ):
            orchestrator._preflight_stage_for_model(contract, model_id)


def test_v2116_d_b_and_cross_projection_routes_preserve_exact_call_denominators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    projection_path = tmp_path / "projection.json"
    payload = {
        "projection": {
            "fixture::action": {
                "reserved_p95": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "cost_usd": 0.001,
                }
            },
            "fixture::semantic": {
                "reserved_p95": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "cost_usd": 0.001,
                }
            },
        }
    }
    projection_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_projection",
        lambda *_args, **_kwargs: (payload, projection_path),
    )

    b = tuple(
        orchestrator.projection_from_preflight(
            contract, spec, raw_root=tmp_path
        )
        for spec in contract.expand(stage="experiment-b")
    )
    cross = tuple(
        orchestrator.projection_from_preflight(
            contract, spec, raw_root=tmp_path
        )
        for spec in contract.expand(stage="cross-model")
    )
    d = []
    for seed in MAIN_SEEDS:
        group = tuple(
            spec
            for spec in contract.expand(stage="experiment-d")
            if spec.environment_seed == seed
        )
        representative = next(spec for spec in group if spec.arm_id == "matched-a")
        d.append(
            orchestrator._d_group_projection(
                contract,
                representative,
                raw_root=tmp_path,
            )
        )

    assert sum(row.completions for row in d) == 1480
    assert sum(row.completions for row in b) == 1440
    assert sum(row.completions for row in cross) == 336
    assert all(row.stage_bucket == "hosted_v2116" for row in (*d, *b, *cross))
    assert {tuple(sorted(row.basis["calls_by_kind"])) for row in d} == {
        ("action", "semantic")
    }


def test_v2116_d_mid_stage_reservation_rejection_stops_remaining_itt_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    raw_root = tmp_path / "raw"
    reserve_attempts: list[int] = []
    provider_dispatches: list[int] = []
    d_executor_attempts: list[int] = []
    receipt_calls: list[str] = []

    class RejectSecondReservationBudgetLedger:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.attempts = 0

        def reserve(self, _projection: Any) -> None:
            self.attempts += 1
            reserve_attempts.append(self.attempts)
            if self.attempts == 2:
                raise PilotBudgetError("injected second D group reserve rejection")

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        orchestrator, "_persist_release_attestation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        orchestrator, "_parent_budget_debit", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        orchestrator, "PilotBudgetLedger", RejectSecondReservationBudgetLedger
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v2116_scientific_dispatch_acceptance",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator, "_assert_prerequisites", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        orchestrator, "_remaining_core_projections", lambda *_args, **_kwargs: ()
    )
    monkeypatch.setattr(
        orchestrator, "_assert_projection_matrix_fits", lambda *_args, **_kwargs: None
    )

    def catalog(
        _contract_value: Any, *, model_ids: Any
    ) -> dict[str, Any]:
        return {"rows": [{"profile_id": tuple(model_ids)[0]}]}

    monkeypatch.setattr(orchestrator, "validate_live_provider_catalog", catalog)
    monkeypatch.setattr(
        orchestrator,
        "verify_provider_catalog_receipt",
        lambda receipt, **_kwargs: receipt,
    )

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("provider construction escaped the D test boundary")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden_provider)
    monkeypatch.setattr(orchestrator, "MultiModelLLM", forbidden_provider)

    def execute_d_group(
        _contract_value: Any,
        specs: Any,
        *,
        budget_ledger: RejectSecondReservationBudgetLedger,
        run_ledger: Any,
        **_kwargs: Any,
    ) -> None:
        group = tuple(specs)
        if all(run_ledger.is_terminal(spec.run_id) for spec in group):
            return
        seed = group[0].environment_seed
        d_executor_attempts.append(seed)
        budget_ledger.reserve(SimpleNamespace(run_id=f"d-s{seed}"))
        provider_dispatches.append(seed)
        run_ledger.finalize_many(
            [
                {
                    "run_id": spec.run_id,
                    "status": "complete",
                    "artifact": None,
                    "failure": None,
                }
                for spec in group
            ]
        )

    monkeypatch.setattr(orchestrator, "_execute_d_seed", execute_d_group)

    def write_stage_receipt(
        _contract_value: Any,
        stage_id: str,
        *,
        raw_root: Path,
        ledger: Any,
        status: str,
        artifacts: Any = None,
        failure: Any = None,
        **_kwargs: Any,
    ) -> Path:
        stage_specs = contract.expand(stage=stage_id)
        rows = ledger.snapshot()["runs"]
        counts: dict[str, int] = {}
        for spec in stage_specs:
            row_status = rows[spec.run_id]["status"]
            counts[row_status] = counts.get(row_status, 0) + 1
        hard_stops = counts.get("budget-stopped", 0) + counts.get(
            "integrity-stopped", 0
        )
        payload = {
            "stage_id": stage_id,
            "status": status,
            "status_counts": counts,
            "terminal": all(
                rows[spec.run_id]["status"] in orchestrator.TERMINAL_RUN_STATUSES
                for spec in stage_specs
            ),
            "hard_stop_cell_count": hard_stops,
            "execution_progression_go": hard_stops == 0,
            "artifacts": dict(artifacts or {}),
            "failure": dict(failure) if failure else None,
        }
        path = raw_root / stage_id / "stage_receipt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == payload
        else:
            path.write_text(
                json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        receipt_calls.append(status)
        return path

    monkeypatch.setattr(orchestrator, "_write_stage_receipt", write_stage_receipt)

    first = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="experiment-d",
        resume=False,
        raw_root=raw_root,
        repo_root=ROOT,
    )

    assert reserve_attempts == [1, 2]
    assert len(d_executor_attempts) == 2
    assert provider_dispatches == [d_executor_attempts[0]]
    assert first["status"] == "complete-with-no-go"
    assert first["status_counts"] == {"budget-stopped": 44, "complete": 11}
    assert first["terminal"] is True
    assert first["hard_stop_cell_count"] == 44
    assert first["execution_progression_go"] is False
    assert receipt_calls == ["complete-with-no-go"]

    ledger_path = raw_root / "run_ledger.json"
    before_resume = ledger_path.read_bytes()
    snapshot = json.loads(before_resume)["runs"]
    d_rows = [
        snapshot[spec.run_id] for spec in contract.expand(stage="experiment-d")
    ]
    assert [row["status"] for row in d_rows].count("complete") == 11
    assert [row["status"] for row in d_rows].count("budget-stopped") == 44
    stopped_failures = {
        canonical_sha256(row["failure"])
        for row in d_rows
        if row["status"] == "budget-stopped"
    }
    assert len(stopped_failures) == 1
    stopped_failure = next(
        row["failure"] for row in d_rows if row["status"] == "budget-stopped"
    )
    assert stopped_failure == {
        "error_type": "PilotBudgetError",
        "message": "injected second D group reserve rejection",
        "model_id": "gpt52_main",
        "environment_seed": d_executor_attempts[1],
        "provider_dispatch_started": False,
        "projection_scope": "current-and-remaining-experiment-d-stage",
        "stop_origin": "experiment-d-group-pre-dispatch-budget-rejection",
    }
    for stage_id, expected in (("experiment-b", 25), ("cross-model", 6)):
        descendant_rows = [
            snapshot[spec.run_id] for spec in contract.expand(stage=stage_id)
        ]
        assert len(descendant_rows) == expected
        assert {row["status"] for row in descendant_rows} == {"budget-stopped"}
        assert all(
            row["failure"]["source_stage"] == "experiment-d"
            for row in descendant_rows
        )

    resumed = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="experiment-d",
        resume=True,
        raw_root=raw_root,
        repo_root=ROOT,
    )

    assert resumed == first
    assert reserve_attempts == [1, 2]
    assert len(d_executor_attempts) == 2
    assert provider_dispatches == [d_executor_attempts[0]]
    assert receipt_calls == ["complete-with-no-go", "complete-with-no-go"]
    assert ledger_path.read_bytes() == before_resume


def test_v2116_fake_matrix_selects_only_remaining_first_seed_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    observed: dict[str, Any] = {}

    class FakeRunLedger:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.rows: dict[str, str] = {}

        def register(self, specs: Any) -> None:
            observed["registered"] = tuple(specs)
            self.rows.update({spec.run_id: "scheduled" for spec in specs})

        def status(self, run_id: str) -> str:
            return self.rows[run_id]

        def is_terminal(self, run_id: str) -> bool:
            return self.rows[run_id] == "complete"

    class FakeBudgetLedger:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "_bootstrap_development_utility",
        lambda *_args, **_kwargs: {"fixture": "bootstrap"},
    )
    monkeypatch.setattr(orchestrator, "PilotRunLedger", FakeRunLedger)
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", FakeBudgetLedger)

    def recover(_budget: Any, ledger: FakeRunLedger, spec: Any) -> bool:
        ledger.rows[spec.run_id] = "complete"
        return True

    def execute_d(
        _contract: Any,
        specs: Any,
        *,
        run_ledger: FakeRunLedger,
        **_kwargs: Any,
    ) -> None:
        for spec in specs:
            run_ledger.rows[spec.run_id] = "complete"

    monkeypatch.setattr(orchestrator, "_recover_or_stop_interrupted_reservation", recover)
    monkeypatch.setattr(orchestrator, "_execute_d_seed", execute_d)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("fake selection test reached a provider")

    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    monkeypatch.setattr(orchestrator, "MultiModelLLM", forbidden)

    result = orchestrator.run_development_fake_matrix(
        contract_path=CONTRACT_PATH,
        resume=False,
        raw_root=tmp_path,
    )

    registered = observed["registered"]
    assert {spec.stage_id for spec in registered} == {
        "experiment-d",
        "experiment-b",
        "cross-model",
    }
    assert {spec.environment_seed for spec in registered} == {MAIN_SEEDS[0]}
    assert len(registered) == 18
    assert result["registered_cells"] == 18
    assert result["status"] == "pass"
    assert result["diagnostic_only"] is True
    assert result["scientific_evidence"] is False


def _cli_contract(*, status: str = "frozen") -> SimpleNamespace:
    return SimpleNamespace(
        contract_id=pilot_contract_module.PILOT_CONTRACT_ID_V2_11_6,
        status=status,
    )


def _cli_args(*values: str) -> Any:
    return run_pilot.build_parser().parse_args(
        ["--contract", str(CONTRACT_PATH), *values]
    )


def test_v2116_cli_parent_import_requires_only_the_exact_parent_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _cli_contract())
    monkeypatch.setattr(
        orchestrator, "load_pilot_contract", lambda _path: contract
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("CLI crossed its parent-root validation boundary")

    monkeypatch.setattr(run_pilot, "execute_stage", forbidden)
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.6 parent-import requires --parent-repo-root",
    ):
        run_pilot.execute(_cli_args("--stage", "parent-import"))

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="--authority-repo-root is accepted only by",
    ):
        run_pilot.execute(
            _cli_args(
                "--stage",
                "parent-import",
                "--parent-repo-root",
                str(tmp_path / "v2115"),
                "--authority-repo-root",
                str(tmp_path / "legacy-authority"),
            )
        )


@pytest.mark.parametrize("stage_id", ("capability-gate", "long-context-preflight"))
def test_v2116_cli_cannot_route_into_capability_or_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
) -> None:
    contract = _contract(monkeypatch)
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _cli_contract())
    monkeypatch.setattr(
        orchestrator, "load_pilot_contract", lambda _path: contract
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=f"unknown frozen stage: {stage_id}",
    ):
        run_pilot.execute(
            _cli_args(
                "--stage",
                stage_id,
                "--raw-root",
                str(tmp_path / "raw"),
            )
        )


def test_v2116_cli_real_stage_rejects_draft_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: _cli_contract(status="draft"),
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("draft contract reached real dispatch")

    monkeypatch.setattr(run_pilot, "execute_stage", forbidden)
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.6 real stages require a frozen contract",
    ):
        run_pilot.execute(
            _cli_args(
                "--stage",
                "experiment-d",
                "--raw-root",
                str(tmp_path / "raw"),
            )
        )


def test_v2116_cli_routes_only_acceptance_flags_to_zero_provider_acceptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    launch = raw_root / "scientific_launch_input.json"
    output = raw_root / "scientific_dispatch_acceptance.json"
    observed: dict[str, Any] = {}
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _cli_contract())

    def accept(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {"status": "go", "go": True, "scientific_evidence": False}

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("acceptance CLI reached a provider/stage executor")

    monkeypatch.setattr(run_pilot, "accept_v2116_scientific_dispatch", accept)
    monkeypatch.setattr(run_pilot, "execute_stage", forbidden)
    monkeypatch.setattr(run_pilot, "run_development_fake_matrix", forbidden)

    result = run_pilot.execute(
        _cli_args(
            "--accept-scientific-dispatch",
            "--raw-root",
            str(raw_root),
            "--scientific-launch-input",
            str(launch),
            "--acceptance-output",
            str(output),
        )
    )

    assert result == {"status": "go", "go": True, "scientific_evidence": False}
    assert observed == {
        "contract_path": CONTRACT_PATH,
        "repo_root": run_pilot.ROOT,
        "raw_root": raw_root,
        "scientific_launch_input_path": launch,
        "receipt_path": output,
    }


def test_v2116_cli_rejects_acceptance_artifact_flags_on_science_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _cli_contract())

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid CLI flags reached dispatch")

    monkeypatch.setattr(run_pilot, "execute_stage", forbidden)
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="require --accept-scientific-dispatch",
    ):
        run_pilot.execute(
            _cli_args(
                "--stage",
                "experiment-d",
                "--raw-root",
                str(tmp_path / "raw"),
                "--acceptance-output",
                str(tmp_path / "receipt.json"),
            )
        )
