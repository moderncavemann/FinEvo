from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v2113_acceptance as acceptance
from verified_memory.pilot_budget import (
    ParentBudgetDebit,
    PilotBudgetCaps,
    PilotBudgetLedger,
    RunProjection,
)
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_3.yaml"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _clear_hosted_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in acceptance._PROVIDER_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_acceptance_receipt_seal_detects_mutation() -> None:
    receipt = acceptance._seal(
        {
            "schema_version": "fixture-v1",
            "scientific_evidence": False,
            "value": {"cells": 131},
        }
    )

    acceptance._verify_seal(receipt)
    receipt["value"]["cells"] = 130

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="self-hash mismatch",
    ):
        acceptance._verify_seal(receipt)


def test_acceptance_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"go":true,"go":false}\n', encoding="utf-8")

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="duplicate JSON key",
    ):
        acceptance._strict_json(path, name="duplicate fixture")


def test_acceptance_rejects_symlink_source(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    link = tmp_path / "receipt.json"
    target.write_text("{}\n", encoding="utf-8")
    link.symlink_to(target)

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="symlink",
    ):
        acceptance._strict_json(link, name="symlink fixture")


def test_acceptance_rejects_symlinked_raw_parent(tmp_path: Path) -> None:
    repo_root = tmp_path / "release"
    outside = tmp_path / "outside"
    repo_root.mkdir()
    (outside / "pilot-v2.11.3" / "raw").mkdir(parents=True)
    (repo_root / "experiment_results").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="raw namespace contains a symlink",
    ):
        acceptance._exact_roots(
            repo_root,
            repo_root / "experiment_results" / "pilot-v2.11.3" / "raw",
        )


@pytest.mark.parametrize(
    "relative",
    (
        "experiment-a/copied-result.json",
        (
            "unexpected/finevo-pilot-v2.11.2--experiment-c--gpt52_main--"
            "full--registered-rate-shock--stage0-selected--s1099057501/"
            "decoded.json"
        ),
    ),
)
def test_acceptance_rejects_any_pre_science_or_legacy_scientific_path(
    tmp_path: Path,
    relative: str,
) -> None:
    raw_root = tmp_path / "experiment_results/pilot-v2.11.3/raw"
    path = raw_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="pre-science raw namespace contains scientific artifacts",
    ):
        acceptance._audit_pre_science_namespace(raw_root)


def test_provider_boundary_blocks_canonical_factories_and_restores_identity() -> None:
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
                acceptance.PilotV2113AcceptanceError,
                match="zero-provider acceptance attempted",
            ):
                call()

    assert acceptance.canonical_llm_providers.create_llm_provider is provider_factory
    assert acceptance.canonical_llm_providers.MultiModelLLM is multi_model
    assert (
        acceptance.canonical_provider_catalog.validate_live_provider_catalog
        is catalog_validator
    )


def test_real_contract_forward_matrix_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = replace(load_pilot_contract(CONTRACT_PATH), status="frozen")
    monkeypatch.setattr(acceptance, "load_pilot_contract", lambda _path: contract)

    acceptance._require_contract(contract, ROOT)


def test_stage_budget_rows_are_limited_to_exact_registered_projections() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    runs: dict[str, Any] = {}
    for stage_id in acceptance.V2113_OPERATIONAL_STAGE_IDS:
        for spec in contract.expand(stage=stage_id):
            projection = (
                orchestrator._v2113_parent_import_projection(spec)
                if spec.execution_mode == "parent_authority_import"
                else orchestrator._v2113_operational_import_projection(spec)
            ).to_dict()
            runs[spec.run_id] = {
                "stage_bucket": spec.budget_bucket,
                "reservation": projection,
            }
    science = RunProjection(
        run_id="fixture-scientific-projection",
        stage_bucket="hosted_v2113",
        cost_usd=1.0,
        completions=48,
        storage_bytes=20_000_000,
        basis={"method": "fixture"},
    ).to_dict()
    runs[science["run_id"]] = {
        "stage_bucket": science["stage_bucket"],
        "reservation": science,
    }
    projection_receipt = {
        "projection_sha256_by_run": {
            science["run_id"]: acceptance.canonical_sha256(science)
        }
    }

    acceptance._verify_current_budget_rows(contract, {"runs": runs}, projection_receipt)
    runs[science["run_id"]]["stage_bucket"] = "parent_v2112"

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="stage bucket drifted",
    ):
        acceptance._verify_current_budget_rows(
            contract, {"runs": runs}, projection_receipt
        )


def test_projection_audit_reads_real_forward_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    def normal_projection(_contract: Any, spec: Any, **_kwargs: Any) -> RunProjection:
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
        _contract: Any, representative: Any, **_kwargs: Any
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
    for stage_id in acceptance.V2113_OPERATIONAL_STAGE_IDS:
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
    boundary = contract.v2113_forward_boundary
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
        orchestrator, "_assert_projection_matrix_fits", lambda *_a, **_k: None
    )

    result = acceptance._audit_projections(
        contract,
        repo_root=ROOT,
        raw_root=ROOT / "experiment_results" / "pilot-v2.11.3" / "raw",
        paid=SimpleNamespace(),
        run_ledger=SimpleNamespace(),
        budget_ledger=budget,
    )

    assert result["fresh_calls_by_model"] == boundary["matrix"]["fresh_calls_by_model"]
    assert result["fresh_provider_calls"] == 5_816


def test_acceptance_refuses_loaded_hosted_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_hosted_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret-never-read")

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="before provider credentials are loaded; present=OPENAI_API_KEY",
    ):
        acceptance._require_provider_keys_absent()


def test_acceptance_cli_go_receipt_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        run_pilot,
        "execute",
        lambda _args: {"status": "go", "go": True, "scientific_evidence": False},
    )

    assert run_pilot.main(["--accept-scientific-dispatch"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "go"


class _SnapshotLedger:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.value = snapshot

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.value))


def test_acceptance_audit_happy_path_is_deterministic_and_zero_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_hosted_keys(monkeypatch)
    repo_root = tmp_path / "release"
    raw_root = repo_root / "experiment_results" / "pilot-v2.11.3" / "raw"
    raw_root.mkdir(parents=True)
    launch_path = raw_root / "scientific_launch_input.json"
    release_path = raw_root / "release_attestation.json"
    run_path = raw_root / "run_ledger.json"
    budget_path = raw_root / "budget_ledger.json"
    _write_json(launch_path, {"launch": "bound"})
    release_value = {"schema_version": "fixture-release-v1", "status": "pass"}
    _write_json(release_path, release_value)
    _write_json(run_path, {"fixture": "run"})
    _write_json(budget_path, {"fixture": "budget"})

    contract = SimpleNamespace(
        contract_id=acceptance.V2113_CONTRACT_ID,
        canonical_hash="a" * 64,
    )
    paid = orchestrator.GitProvenance(
        git_tag="pilot-v2.11.3-science",
        head_commit="b" * 40,
        tag_commit="b" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={"contract": "fixture"},
        release_attestation=release_value,
    )
    run_snapshot = {
        "events": [
            {"event_sha256": f"{index:064x}"}
            for index in range(1, acceptance.V2113_EXPECTED_ACCEPTED_RUN_EVENTS + 1)
        ],
        "ledger_sha256": "d" * 64,
    }
    budget_snapshot = {
        "events": [
            {"event_sha256": f"{index + 100:064x}"}
            for index in range(1, acceptance.V2113_EXPECTED_ACCEPTED_BUDGET_EVENTS + 1)
        ],
        "ledger_sha256": "f" * 64,
    }
    provider_calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        provider_calls.append("called")
        raise AssertionError("acceptance reached a provider boundary")

    monkeypatch.setattr(acceptance, "_require_contract", lambda *_a, **_k: None)
    monkeypatch.setattr(
        acceptance,
        "_open_ledgers",
        lambda *_a, **_k: (
            _SnapshotLedger(run_snapshot),
            _SnapshotLedger(budget_snapshot),
            run_path,
            budget_path,
        ),
    )
    monkeypatch.setattr(orchestrator, "verify_paid_provenance", lambda *_a, **_k: paid)
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    monkeypatch.setattr(orchestrator, "validate_live_provider_catalog", forbidden)
    monkeypatch.setattr(orchestrator, "MultiModelLLM", forbidden)
    monkeypatch.setattr(
        acceptance,
        "_audit_denominator",
        lambda *_a, **_k: {
            "ledger_cells": 136,
            "operational_cells": 5,
            "scientific_cells": 131,
            "provider_backed_scientific_cells": 126,
            "offline_scientific_cells": 5,
        },
    )
    monkeypatch.setattr(
        acceptance,
        "_audit_operational_receipts",
        lambda *_a, **_k: ({"all": "complete"}, {}),
    )
    monkeypatch.setattr(
        acceptance,
        "_audit_authorities",
        lambda *_a, **_k: ({"all": "verified"}, {}),
    )
    monkeypatch.setattr(
        acceptance,
        "_audit_configs_and_d_groups",
        lambda *_a, **_k: (
            {"provider_config_count": 126},
            {"group_count": 5, "cells_per_group": 11},
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "_audit_projections",
        lambda *_a, **_k: {
            "projection_unit_count": 81,
            "fresh_provider_calls": 5816,
            "fresh_calls_by_kind": {"action": 4848, "semantic": 968},
            "full_matrix_fits": True,
        },
    )

    first = acceptance.audit_v2113_scientific_dispatch(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        scientific_launch_input_path=launch_path,
    )
    second = acceptance.audit_v2113_scientific_dispatch(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        scientific_launch_input_path=launch_path,
    )

    assert first == second
    acceptance._verify_seal(first)
    assert first["scientific_evidence"] is False
    assert first["pre_science_namespace"] == acceptance._PRE_SCIENCE_NAMESPACE
    assert first["provider_boundary"]["zero_provider_acceptance"] is True
    assert provider_calls == []
    assert not any(
        "timestamp" in key or key.endswith("_at")
        for value in (first,)
        for key in _recursive_keys(value)
    )


def _recursive_keys(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            result.append(str(key))
            result.extend(_recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_recursive_keys(item))
    return result


@dataclass(frozen=True)
class _Spec:
    run_id: str = "fixture-science-cell"

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "stage_id": "experiment-c"}


def _acceptance_transaction_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Build real append-only ledgers while stubbing unrelated release audits."""

    _clear_hosted_keys(monkeypatch)
    repo_root = tmp_path / "release"
    raw_root = repo_root / "experiment_results" / "pilot-v2.11.3" / "raw"
    raw_root.mkdir(parents=True)
    contract_path = repo_root / "experiments" / "pilot_v2_11_3.yaml"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text("fixture: true\n", encoding="utf-8")
    launch_path = raw_root / "scientific_launch_input.json"
    _write_json(launch_path, {"fixture": "launch"})

    specs = tuple(_Spec(f"operational-{index}") for index in range(5))
    contract = SimpleNamespace(
        contract_id=acceptance.V2113_CONTRACT_ID,
        canonical_hash="a" * 64,
        expand=lambda stage=None: specs,
    )
    run_path = raw_root / "run_ledger.json"
    budget_path = raw_root / "budget_ledger.json"
    run_ledger = orchestrator.PilotRunLedger(
        run_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(specs)
    for spec in specs:
        run_ledger.finalize(
            spec.run_id,
            status="complete",
            artifact=f"fixture/{spec.run_id}.json",
        )

    caps = PilotBudgetCaps(
        total_usd=500.0,
        max_completions=10_000,
        max_storage_bytes=5_000_000_000,
        stage_usd_caps={
            "operational": 0.0,
            "hosted_v2113": 499.0,
            "manual_reserve": 1.0,
        },
        automatic_reserve_usd=1.0,
    )
    parent = ParentBudgetDebit(
        parent_contract_sha256="b" * 64,
        parent_run_ledger_sha256="c" * 64,
        parent_budget_ledger_sha256="d" * 64,
        stage_bucket="operational",
        cost_usd=0.0,
        hosted_completions=0,
        storage_bytes=0,
    )
    budget_ledger = PilotBudgetLedger(
        budget_path,
        contract_hash=contract.canonical_hash,
        caps=caps,
        tamper_evident=True,
        parent_debit=parent,
    )
    for spec in specs:
        projection = RunProjection(
            run_id=spec.run_id,
            stage_bucket="operational",
            cost_usd=0.0,
            completions=0,
            storage_bytes=0,
            basis={"fixture": "zero-call"},
        )
        budget_ledger.reserve(projection)
        budget_ledger.finalize(
            spec.run_id,
            status="complete",
            cost_usd=0.0,
            completions=0,
            storage_bytes=0,
        )
    assert len(run_ledger.snapshot()["events"]) == 7
    assert len(budget_ledger.snapshot()["events"]) == 12

    def open_ledgers(_contract: Any, _raw_root: Path) -> tuple[Any, Any, Path, Path]:
        return (
            orchestrator.PilotRunLedger(
                run_path,
                contract_hash=contract.canonical_hash,
                tamper_evident=True,
            ),
            PilotBudgetLedger(
                budget_path,
                contract_hash=contract.canonical_hash,
                caps=caps,
                tamper_evident=True,
                parent_debit=parent,
            ),
            run_path,
            budget_path,
        )

    initial_run, initial_budget, _run_path, _budget_path = open_ledgers(
        contract, raw_root
    )
    receipt = acceptance._seal(
        {
            "schema_version": (
                acceptance.V2113_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION
            ),
            "ledger_prefixes": {
                "run_ledger": acceptance._ledger_prefix(
                    initial_run.snapshot(), run_path
                ),
                "budget_ledger": acceptance._ledger_prefix(
                    initial_budget.snapshot(), budget_path
                ),
            },
        }
    )
    receipt_path = raw_root / acceptance.V2113_SCIENTIFIC_DISPATCH_ACCEPTANCE_FILENAME
    paid = SimpleNamespace(git_tag="fixture-tag", head_commit="e" * 40)
    audit_calls: list[str] = []

    def fake_audit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        audit_calls.append("audit")
        return acceptance._json_copy(receipt)

    def fake_core(
        path: str | Path,
        *,
        contract: Any,
        run_ledger: Any,
        budget_ledger: Any,
        require_markers: bool,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        value = acceptance._strict_json(Path(path), name="transaction receipt")
        acceptance._verify_seal(value)
        run_snapshot = run_ledger.snapshot()
        budget_snapshot = budget_ledger.snapshot()
        acceptance._verify_ledger_prefix(
            value["ledger_prefixes"]["run_ledger"],
            run_snapshot,
            name="run ledger",
        )
        acceptance._verify_ledger_prefix(
            value["ledger_prefixes"]["budget_ledger"],
            budget_snapshot,
            name="budget ledger",
        )
        if require_markers:
            acceptance._verify_acceptance_event_binding(
                value,
                run_snapshot,
                budget_snapshot,
                contract=contract,
                receipt_path=receipt_path.relative_to(repo_root).as_posix(),
            )
        return acceptance._json_copy(value)

    def fake_public_verify(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        return fake_core(path, require_markers=True, **kwargs)

    monkeypatch.setattr(acceptance, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(acceptance, "_require_contract", lambda *_a, **_k: None)
    monkeypatch.setattr(acceptance, "_open_ledgers", open_ledgers)
    monkeypatch.setattr(acceptance, "_audit_denominator", lambda *_a, **_k: {})
    monkeypatch.setattr(acceptance, "audit_v2113_scientific_dispatch", fake_audit)
    monkeypatch.setattr(
        acceptance,
        "_verify_v2113_scientific_dispatch_acceptance_core",
        fake_core,
    )
    monkeypatch.setattr(
        acceptance,
        "verify_v2113_scientific_dispatch_acceptance",
        fake_public_verify,
    )
    monkeypatch.setattr(
        acceptance.orch,
        "verify_paid_provenance",
        lambda *_a, **_k: paid,
    )

    def accept() -> dict[str, Any]:
        return acceptance.accept_v2113_scientific_dispatch(
            contract_path=contract_path,
            repo_root=repo_root,
            raw_root=raw_root,
            scientific_launch_input_path=launch_path,
        )

    return {
        "accept": accept,
        "audit_calls": audit_calls,
        "receipt_path": receipt_path,
        "run_path": run_path,
        "budget_path": budget_path,
        "open_ledgers": open_ledgers,
        "contract": contract,
        "raw_root": raw_root,
        "fake_public_verify": fake_public_verify,
    }


def test_atomic_receipt_write_failure_cleans_up_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "scientific_dispatch_acceptance.json"
    receipt = acceptance._seal({"fixture": "atomic-no-replace"})
    original = acceptance._write_all

    def partial_write_then_fail(descriptor: int, payload: bytes) -> None:
        acceptance.os.write(descriptor, payload[:7])
        raise OSError("injected receipt write failure")

    monkeypatch.setattr(acceptance, "_write_all", partial_write_then_fail)
    with pytest.raises(OSError, match="injected receipt write failure"):
        acceptance._persist_exact_receipt(path, receipt)

    assert not path.exists()
    assert not tuple(tmp_path.glob(f".{path.name}.*.tmp"))

    monkeypatch.setattr(acceptance, "_write_all", original)
    acceptance._persist_exact_receipt(path, receipt)
    assert acceptance._strict_json(path, name="retried receipt") == receipt


def test_acceptance_second_call_is_byte_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _acceptance_transaction_case(tmp_path, monkeypatch)
    first = case["accept"]()
    before = {
        name: case[f"{name}_path"].read_bytes() for name in ("receipt", "run", "budget")
    }

    second = case["accept"]()
    after = {
        name: case[f"{name}_path"].read_bytes() for name in ("receipt", "run", "budget")
    }

    assert first == second
    assert before == after
    assert case["audit_calls"] == ["audit"]
    run, budget, _run_path, _budget_path = case["open_ledgers"](
        case["contract"], case["raw_root"]
    )
    assert len(run.snapshot()["events"]) == 8
    assert len(budget.snapshot()["events"]) == 13


def test_acceptance_recovers_after_receipt_before_run_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _acceptance_transaction_case(tmp_path, monkeypatch)
    original_write = orchestrator.PilotRunLedger._write
    failed = False

    def fail_first_marker_write(self: Any) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected run-marker write failure")
        original_write(self)

    monkeypatch.setattr(orchestrator.PilotRunLedger, "_write", fail_first_marker_write)
    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="failed to bind acceptance receipt into the run ledger",
    ):
        case["accept"]()

    assert case["receipt_path"].is_file()
    assert len(json.loads(case["run_path"].read_text())["events"]) == 7
    assert len(json.loads(case["budget_path"].read_text())["events"]) == 12

    monkeypatch.setattr(orchestrator.PilotRunLedger, "_write", original_write)
    case["accept"]()
    assert len(json.loads(case["run_path"].read_text())["events"]) == 8
    assert len(json.loads(case["budget_path"].read_text())["events"]) == 13


def test_acceptance_recovers_after_run_marker_before_budget_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _acceptance_transaction_case(tmp_path, monkeypatch)
    original_write = PilotBudgetLedger._write
    failed = False

    def fail_first_marker_write(self: Any) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected budget-marker write failure")
        original_write(self)

    monkeypatch.setattr(PilotBudgetLedger, "_write", fail_first_marker_write)
    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="failed to bind acceptance receipt into the budget ledger",
    ):
        case["accept"]()

    assert len(json.loads(case["run_path"].read_text())["events"]) == 8
    assert len(json.loads(case["budget_path"].read_text())["events"]) == 12

    monkeypatch.setattr(PilotBudgetLedger, "_write", original_write)
    case["accept"]()
    assert len(json.loads(case["run_path"].read_text())["events"]) == 8
    assert len(json.loads(case["budget_path"].read_text())["events"]) == 13


def test_acceptance_recovers_after_both_markers_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _acceptance_transaction_case(tmp_path, monkeypatch)
    final_verify = case["fake_public_verify"]
    failed = False

    def fail_after_both_markers(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        nonlocal failed
        value = final_verify(path, **kwargs)
        if not failed:
            failed = True
            raise RuntimeError("injected post-marker crash")
        return value

    monkeypatch.setattr(
        acceptance,
        "verify_v2113_scientific_dispatch_acceptance",
        fail_after_both_markers,
    )
    with pytest.raises(RuntimeError, match="injected post-marker crash"):
        case["accept"]()

    before = {
        name: case[f"{name}_path"].read_bytes() for name in ("receipt", "run", "budget")
    }
    assert len(json.loads(before["run"])["events"]) == 8
    assert len(json.loads(before["budget"])["events"]) == 13

    case["accept"]()
    after = {
        name: case[f"{name}_path"].read_bytes() for name in ("receipt", "run", "budget")
    }
    assert before == after


def _verification_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    repo_root = tmp_path / "release"
    raw_root = repo_root / "experiment_results" / "pilot-v2.11.3" / "raw"
    raw_root.mkdir(parents=True)
    source_path = raw_root / "accepted-source.json"
    _write_json(source_path, {"source": "immutable"})
    spec = _Spec()
    contract = SimpleNamespace(
        contract_id=acceptance.V2113_CONTRACT_ID,
        canonical_hash="1" * 64,
        expand=lambda stage=None: (spec,),
    )
    paid = SimpleNamespace(
        git_tag="pilot-v2.11.3-science",
        head_commit="2" * 40,
    )
    run = _SnapshotLedger(
        {
            "events": [
                {"event_sha256": f"{index:064x}"}
                for index in range(1, acceptance.V2113_EXPECTED_ACCEPTED_RUN_EVENTS + 1)
            ],
            "runs": {spec.run_id: {"spec": spec.to_dict(), "status": "complete"}},
        }
    )
    caps = {"total_usd": 500.0}
    budget = _SnapshotLedger(
        {
            "events": [
                {"event_sha256": f"{index + 100:064x}"}
                for index in range(
                    1, acceptance.V2113_EXPECTED_ACCEPTED_BUDGET_EVENTS + 1
                )
            ],
            "caps": caps,
            "runs": {spec.run_id: {"fixture": "accepted-operational-row"}},
        }
    )
    run.value["events"][-1]["payload"] = {
        "runs_sha256": acceptance.canonical_sha256(run.value["runs"])
    }
    denominator = {
        "ledger_cells": 136,
        "operational_cells": 5,
        "scientific_cells": 131,
        "provider_backed_scientific_cells": 126,
        "offline_scientific_cells": 5,
        "stage_cell_counts": dict(acceptance.V2113_EXPECTED_STAGE_CELL_COUNTS),
        "operational_status": "complete",
        "scientific_status": "scheduled",
        "scientific_run_ids_sha256": "3" * 64,
    }
    receipt = acceptance._seal(
        {
            "schema_version": (
                acceptance.V2113_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION
            ),
            "status": "go",
            "go": True,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "release": {
                "git_tag": paid.git_tag,
                "git_commit": paid.head_commit,
                "tag_object_type": "tag",
                "worktree_clean": True,
            },
            "raw_namespace": acceptance.V2113_RAW_ROOT.as_posix(),
            "pre_science_namespace": dict(acceptance._PRE_SCIENCE_NAMESPACE),
            "denominator": denominator,
            "operational_gates": {"fixture": "verified"},
            "authorities": {"fixture": "verified"},
            "runner_configs": {"provider_config_count": 126},
            "experiment_d": {"group_count": 5, "cells_per_group": 11},
            "budget_projection": {
                "projection_unit_count": 81,
                "fresh_provider_calls": 5816,
                "fresh_calls_by_kind": {"action": 4848, "semantic": 968},
                "full_matrix_fits": True,
                "hard_caps": caps,
            },
            "ledger_prefixes": {
                "run_ledger": {
                    "file_sha256": "4" * 64,
                    "ledger_sha256": "5" * 64,
                    "event_count": acceptance.V2113_EXPECTED_ACCEPTED_RUN_EVENTS,
                    "event_chain_head": run.value["events"][
                        acceptance.V2113_EXPECTED_ACCEPTED_RUN_EVENTS - 1
                    ]["event_sha256"],
                },
                "budget_ledger": {
                    "file_sha256": "6" * 64,
                    "ledger_sha256": "7" * 64,
                    "event_count": acceptance.V2113_EXPECTED_ACCEPTED_BUDGET_EVENTS,
                    "event_chain_head": budget.value["events"][
                        acceptance.V2113_EXPECTED_ACCEPTED_BUDGET_EVENTS - 1
                    ]["event_sha256"],
                },
            },
            "bound_source_file_sha256": {
                source_path.relative_to(repo_root).as_posix(): acceptance._file_sha256(
                    source_path, name="fixture source"
                )
            },
            "provider_boundary": acceptance._expected_provider_boundary(),
            "scientific_evidence": False,
            "claim_boundary": acceptance._CLAIM_BOUNDARY,
        }
    )
    receipt_path = raw_root / acceptance.V2113_SCIENTIFIC_DISPATCH_ACCEPTANCE_FILENAME
    _write_json(receipt_path, receipt)
    relative_receipt_path = receipt_path.relative_to(repo_root).as_posix()
    common_marker_payload = {
        "receipt_schema_version": (
            acceptance.V2113_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION
        ),
        "receipt_path": relative_receipt_path,
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "accepted_run_event_count": acceptance.V2113_EXPECTED_ACCEPTED_RUN_EVENTS,
        "accepted_run_event_chain_head": receipt["ledger_prefixes"]["run_ledger"][
            "event_chain_head"
        ],
        "accepted_budget_event_count": (
            acceptance.V2113_EXPECTED_ACCEPTED_BUDGET_EVENTS
        ),
        "accepted_budget_event_chain_head": receipt["ledger_prefixes"]["budget_ledger"][
            "event_chain_head"
        ],
    }
    run.value["events"].append(
        {
            "event_index": acceptance.V2113_EXPECTED_ACCEPTED_RUN_EVENTS,
            "event_type": acceptance.V2113_ACCEPTANCE_LEDGER_EVENT_TYPE,
            "previous_event_sha256": common_marker_payload[
                "accepted_run_event_chain_head"
            ],
            "payload": {
                **common_marker_payload,
                "runs_sha256": acceptance.canonical_sha256(run.value["runs"]),
            },
            "event_sha256": "8" * 64,
        }
    )
    budget.value["events"].append(
        {
            "event_index": acceptance.V2113_EXPECTED_ACCEPTED_BUDGET_EVENTS,
            "event_type": acceptance.V2113_ACCEPTANCE_LEDGER_EVENT_TYPE,
            "previous_event_sha256": common_marker_payload[
                "accepted_budget_event_chain_head"
            ],
            "payload": {
                **common_marker_payload,
                "budget_runs_sha256": acceptance.canonical_sha256(budget.value["runs"]),
            },
            "event_sha256": "9" * 64,
        }
    )
    monkeypatch.setattr(acceptance, "_require_contract", lambda *_a, **_k: None)
    monkeypatch.setattr(
        acceptance, "_expected_denominator", lambda _contract: denominator
    )
    monkeypatch.setattr(
        acceptance,
        "_static_acceptance_material",
        lambda *_a, **_k: {
            field: receipt[field]
            for field in (
                "release",
                "operational_gates",
                "authorities",
                "runner_configs",
                "experiment_d",
                "budget_projection",
                "bound_source_file_sha256",
            )
        },
    )
    monkeypatch.setattr(
        acceptance, "_verify_current_budget_rows", lambda *_a, **_k: None
    )
    return {
        "repo_root": repo_root,
        "raw_root": raw_root,
        "source_path": source_path,
        "contract": contract,
        "paid": paid,
        "run": run,
        "budget": budget,
        "receipt_path": receipt_path,
    }


def _verify_case(case: dict[str, Any]) -> dict[str, Any]:
    return acceptance.verify_v2113_scientific_dispatch_acceptance(
        case["receipt_path"],
        contract=case["contract"],
        repo_root=case["repo_root"],
        raw_root=case["raw_root"],
        paid=case["paid"],
        run_ledger=case["run"],
        budget_ledger=case["budget"],
    )


def test_acceptance_verifier_allows_ledger_appends_after_accepted_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _verification_case(tmp_path, monkeypatch)

    verified = _verify_case(case)

    assert verified["go"] is True
    assert len(case["run"].snapshot()["events"]) == (
        acceptance.V2113_EXPECTED_ACCEPTED_RUN_EVENTS + 1
    )
    assert len(case["budget"].snapshot()["events"]) == (
        acceptance.V2113_EXPECTED_ACCEPTED_BUDGET_EVENTS + 1
    )


def test_acceptance_verifier_rejects_prefix_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _verification_case(tmp_path, monkeypatch)
    case["run"].value["events"][acceptance.V2113_EXPECTED_ACCEPTED_RUN_EVENTS - 1][
        "event_sha256"
    ] = ("9" * 64)

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="run ledger accepted event prefix drifted",
    ):
        _verify_case(case)


@pytest.mark.parametrize("mutation", ("file", "self-hash"))
def test_acceptance_pre_marker_requires_exact_file_and_self_hash(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "run_ledger.json"
    _write_json(path, {"fixture": "pre-marker"})
    snapshot = {
        "events": [{"event_sha256": "1" * 64}],
        "ledger_sha256": "2" * 64,
    }
    accepted = acceptance._ledger_prefix(snapshot, path)
    ledger = SimpleNamespace(path=path)

    if mutation == "file":
        _write_json(path, {"fixture": "drifted"})
    else:
        snapshot["ledger_sha256"] = "3" * 64

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="pre-marker file or self-hash differs",
    ):
        acceptance._verify_unmarked_ledger_identity(
            accepted,
            snapshot,
            ledger,
            name="run ledger",
        )


def test_acceptance_verifier_rejects_resealed_static_field_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _verification_case(tmp_path, monkeypatch)
    forged = acceptance._strict_json(case["receipt_path"], name="acceptance fixture")
    forged["runner_configs"]["config_set_sha256"] = "9" * 64
    forged = acceptance._seal(forged)
    _write_json(case["receipt_path"], forged)
    for ledger in (case["run"], case["budget"]):
        ledger.value["events"][-1]["payload"]["receipt_content_sha256"] = forged[
            "integrity"
        ]["content_sha256"]

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="runner_configs.*source recomputation",
    ):
        _verify_case(case)


def test_acceptance_verifier_rejects_bound_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _verification_case(tmp_path, monkeypatch)
    _write_json(case["source_path"], {"source": "mutated"})

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="accepted source file drifted",
    ):
        _verify_case(case)


@pytest.mark.parametrize(
    "mutation",
    (
        "top-level-extra",
        "provider-extra",
        "denominator-extra",
        "pre-science-drift",
        "claim-drift",
    ),
)
def test_acceptance_verifier_rejects_exact_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    case = _verification_case(tmp_path, monkeypatch)
    forged = acceptance._strict_json(case["receipt_path"], name="acceptance fixture")
    if mutation == "top-level-extra":
        forged["unregistered_field"] = True
    elif mutation == "provider-extra":
        forged["provider_boundary"]["unregistered_field"] = 0
    elif mutation == "denominator-extra":
        forged["denominator"]["unregistered_field"] = 0
    elif mutation == "pre-science-drift":
        forged["pre_science_namespace"]["scientific_stage_paths_present"] = 1
    else:
        forged["claim_boundary"] = "effectiveness established"
    _write_json(case["receipt_path"], acceptance._seal(forged))

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match="identity or denominator drifted",
    ):
        _verify_case(case)


@pytest.mark.parametrize("ledger_name", ("run", "budget"))
def test_acceptance_verifier_rejects_missing_ledger_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ledger_name: str,
) -> None:
    case = _verification_case(tmp_path, monkeypatch)
    case[ledger_name].value["events"].pop()

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match=f"not bound after its {ledger_name}-ledger prefix",
    ):
        _verify_case(case)


@pytest.mark.parametrize(
    ("ledger_name", "field"),
    (
        ("run", "receipt_content_sha256"),
        ("run", "runs_sha256"),
        (
            "budget",
            "budget_runs_sha256",
        ),
    ),
)
def test_acceptance_verifier_rejects_marker_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ledger_name: str,
    field: str,
) -> None:
    case = _verification_case(tmp_path, monkeypatch)
    case[ledger_name].value["events"][-1]["payload"][field] = "f" * 64

    with pytest.raises(
        acceptance.PilotV2113AcceptanceError,
        match=f"{ledger_name}-ledger marker differs",
    ):
        _verify_case(case)


def test_public_science_verifier_cannot_disable_marker_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _verification_case(tmp_path, monkeypatch)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        acceptance.verify_v2113_scientific_dispatch_acceptance(
            case["receipt_path"],
            contract=case["contract"],
            repo_root=case["repo_root"],
            raw_root=case["raw_root"],
            paid=case["paid"],
            run_ledger=case["run"],
            budget_ledger=case["budget"],
            _require_markers=False,
        )


def test_v2113_science_without_acceptance_fails_before_provider_and_stops_itt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    repo_root = tmp_path / "release"
    raw_root = repo_root / "experiment_results" / "pilot-v2.11.3" / "raw"
    raw_root.mkdir(parents=True)
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    boundary = contract.v2113_forward_boundary
    assert boundary is not None
    PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=ParentBudgetDebit.from_dict(boundary["parent_budget_debit"]),
    )
    paid = orchestrator.GitProvenance(
        git_tag="pilot-v2.11.3-science",
        head_commit="7" * 40,
        tag_commit="7" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )
    provider_calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        provider_calls.append("called")
        raise AssertionError("missing acceptance reached provider construction")

    monkeypatch.setattr(orchestrator, "verify_paid_provenance", lambda *_a, **_k: paid)
    monkeypatch.setattr(
        orchestrator, "_persist_release_attestation", lambda *_a, **_k: None
    )
    monkeypatch.setattr(acceptance, "_require_contract", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    monkeypatch.setattr(orchestrator, "validate_live_provider_catalog", forbidden)
    monkeypatch.setattr(orchestrator, "MultiModelLLM", forbidden)
    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        lambda *_a, **_k: raw_root / "experiment-c" / "stage_receipt.json",
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="stage prerequisites failed",
    ):
        orchestrator._execute_stage_locked(
            contract_path=CONTRACT_PATH,
            stage_id="experiment-c",
            resume=True,
            raw_root=raw_root,
            repo_root=repo_root,
        )

    reloaded = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    scientific = tuple(
        spec
        for stage_id in acceptance.V2113_SCIENTIFIC_STAGE_IDS
        for spec in contract.expand(stage=stage_id)
    )
    assert len(scientific) == 131
    assert {reloaded.status(spec.run_id) for spec in scientific} == {
        "integrity-stopped"
    }
    assert provider_calls == []
    assert not (raw_root / "experiment-c" / "provider_catalog").exists()
    budget = PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=ParentBudgetDebit.from_dict(boundary["parent_budget_debit"]),
    )
    assert budget.snapshot()["runs"] == {}
