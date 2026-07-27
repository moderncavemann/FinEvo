from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import run_pilot
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10_1.yaml"
V210_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10.yaml"


def _contract():
    return load_pilot_contract(CONTRACT_PATH)


def _paid() -> orchestrator.GitProvenance:
    return orchestrator.GitProvenance(
        git_tag="pilot-v2.10.1-science",
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )


def _projection_and_flat_binding(contract, model_id: str):
    profile = contract.provider_profiles[model_id]
    runtime_model = orchestrator._runtime_model_for_profile(profile)
    reservation = {
        "sample_count": 1,
        "raw_p95": {
            "prompt_tokens": 10.0,
            "completion_tokens": 5.0,
            "total_tokens": 15.0,
            "cost_usd": 0.0,
        },
        "reserved_p95": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
            "cost_usd": 0.0,
        },
        "reserve_multiplier": 1.25,
    }
    projection = {
        f"{profile.served_model}::{kind}": deepcopy(reservation)
        for kind in ("action", "semantic")
    }
    reservations = {
        runtime_model: {
            kind: {
                "authority": {
                    "pilot_contract_hash": contract.canonical_hash,
                    "pilot_tag": "pilot-v2.10.1-science",
                },
                "reservation": deepcopy(reservation),
            }
            for kind in ("action", "semantic")
        }
    }
    return (
        {
            "served_model": profile.served_model,
            "projection": projection,
        },
        {
            "receipt_path": (
                "experiment_results/pilot-v2.10.1/raw/parent-import/"
                f"observed_p95/{model_id}/observed_p95_authority_receipt.json"
            ),
            "receipt_file_sha256": "b" * 64,
            "receipt_content_sha256": "c" * 64,
            "git_commit": "a" * 40,
            "reservations": reservations,
        },
    )


def test_v2101_denominator_and_cli_use_the_fresh_contract() -> None:
    contract = _contract()
    counts = Counter(spec.stage_id for spec in contract.expand())

    assert contract.contract_id == orchestrator.V2101_CONTRACT_ID
    assert len(contract.expand()) == 211
    assert counts["parent-import"] == 1
    assert counts["q-ref-resolution"] == 1
    assert counts["stage0-calibration"] == 14
    assert (
        sum(
            count
            for stage_id, count in counts.items()
            if stage_id.startswith(("experiment-", "local-experiment-"))
        )
        == 195
    )
    assert run_pilot._raw_root_for_contract(CONTRACT_PATH) == (
        ROOT / "experiment_results" / "pilot-v2.10.1" / "raw"
    )


def test_v2101_parent_qref_stage0_transactions_are_zero_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    paid = _paid()
    raw_root = tmp_path / "experiment_results" / "pilot-v2.10.1" / "raw"
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    budget = PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("V2.10.1 prerequisite transaction constructed a provider")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    monkeypatch.setattr(
        orchestrator,
        "_v2_control_gate_ok",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        type(contract),
        "validate_provenance",
        lambda self, git_commit, git_tag: {
            "git_tag": git_tag,
            "resolved_git_commit": git_commit,
            "commit_resolution": self.implementation["commit_resolution"],
            "p0_base_commit": self.implementation["p0_base_commit"],
            "contract_id": self.contract_id,
            "contract_sha256": self.canonical_hash,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_v2101_parent_import_for_orchestrator",
        lambda **_kwargs: {
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_v2101_importer_p95_profiles",
        lambda *_args, **_kwargs: {"verified": True},
    )

    parent_specs = contract.expand(stage="parent-import")
    orchestrator._execute_v24_parent_import_stage(
        contract,
        parent_specs,
        raw_root=raw_root,
        repo_root=tmp_path,
        parent_repo_root=tmp_path,
        paid=paid,
        run_ledger=ledger,
        budget_ledger=budget,
    )
    assert Counter(row["status"] for row in ledger.snapshot()["runs"].values()) == {
        "complete": 1,
        "scheduled": 210,
    }

    monkeypatch.setattr(
        orchestrator,
        "_assert_prerequisites",
        lambda *_args, **_kwargs: True,
    )
    qref = orchestrator._seal_bound_payload(
        {
            "schema_version": (orchestrator.PILOT_V2101_IMPORTED_QREF_SCHEMA_VERSION),
            "status": "pass",
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "q_ref": 63.50397933257746,
            "row_count": 48,
            "source_import": {
                "source_run_id": "immutable-v2.9-qref",
                "source_artifacts": {
                    "q_ref_resolution": {
                        "snapshot_path": "/immutable/v2.9/q_ref_resolution.json",
                        "file_sha256": "d" * 64,
                    }
                },
            },
            "claim_boundary": "zero-provider q-ref fixture",
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": paid.git_tag,
                "git_commit": paid.head_commit,
            },
        }
    )
    monkeypatch.setattr(
        orchestrator,
        "_expected_v2101_q_ref_resolution",
        lambda *_args, **_kwargs: deepcopy(qref),
    )
    orchestrator._execute_v2101_q_ref_import_stage(
        contract,
        contract.expand(stage="q-ref-resolution"),
        raw_root=raw_root,
        paid=paid,
        run_ledger=ledger,
        budget_ledger=budget,
    )
    assert Counter(row["status"] for row in ledger.snapshot()["runs"].values()) == {
        "complete": 2,
        "scheduled": 209,
    }

    def stage0_fixture(*_args: Any, **_kwargs: Any):
        cells = []
        for spec in contract.expand(stage="stage0-calibration"):
            run_dir = raw_root / "stage0-calibration" / "runs" / spec.run_id
            envelope = run_dir / "imported_run_envelope.json"
            terminal = (
                raw_root / "stage0-calibration" / "summaries" / f"{spec.run_id}.json"
            )
            orchestrator._persist_exact_json(
                envelope,
                orchestrator._seal_bound_payload(
                    {
                        "schema_version": (
                            orchestrator.PILOT_V2101_IMPORTED_RUN_ENVELOPE_SCHEMA_VERSION
                        ),
                        "contract_id": contract.contract_id,
                        "contract_sha256": contract.canonical_hash,
                        "run_spec": spec.to_dict(),
                        "bindings": {
                            "contract_sha256": contract.canonical_hash,
                            "git_tag": paid.git_tag,
                            "git_commit": paid.head_commit,
                        },
                    }
                ),
            )
            orchestrator._persist_exact_json(
                terminal,
                {"run_id": spec.run_id, "provider_calls": 0},
            )
            cells.append((spec, envelope, terminal, {}))
        selection = orchestrator._seal_bound_payload(
            {
                "schema_version": "finevo-stage0-selection-v1",
                "selected_profile_id": "nu-0.5",
                "contract_sha256": contract.canonical_hash,
                "bindings": {
                    "contract_sha256": contract.canonical_hash,
                    "git_tag": paid.git_tag,
                    "git_commit": paid.head_commit,
                },
            }
        )
        return selection, tuple(cells)

    monkeypatch.setattr(
        orchestrator,
        "_expected_v27_stage0_selection",
        stage0_fixture,
    )
    monkeypatch.setattr(
        orchestrator,
        "_remaining_core_projections",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_projection_matrix_fits",
        lambda *_args, **_kwargs: None,
    )
    orchestrator._execute_imported_stage0_stage(
        contract,
        contract.expand(stage="stage0-calibration"),
        raw_root=raw_root,
        paid=paid,
        run_ledger=ledger,
        budget_ledger=budget,
    )
    assert Counter(row["status"] for row in ledger.snapshot()["runs"].values()) == {
        "complete": 16,
        "scheduled": 195,
    }


@pytest.mark.parametrize(
    "field",
    ("receipt_path", "git_commit", "receipt_content_sha256"),
)
def test_v2101_p95_tamper_stops_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    contract = _contract()
    model_id = "gpt52_main"
    projection, binding = _projection_and_flat_binding(contract, model_id)
    if field == "git_commit":
        binding[field] = "f" * 40
        expected = "exact current-release flat contract"
    elif field == "receipt_content_sha256":
        binding[field] = "e" * 64
        expected = "differs from the sealed projection"
    else:
        binding.pop(field)
        expected = "exact current-release flat contract"
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_projection",
        lambda *_args, **_kwargs: (deepcopy(projection), tmp_path / "p95.json"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_verified_observed_p95_binding",
        lambda *_args, **_kwargs: deepcopy(binding),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        lambda *_args, **_kwargs: pytest.fail(
            "provider constructed after malformed V2.10.1 p95 binding"
        ),
    )
    if field == "receipt_content_sha256":
        # A content-hash drift is caught when the projection/receipt equality
        # check sees an authority path that no longer names the sealed source.
        projection["projection"][
            f"{contract.provider_profiles[model_id].served_model}::action"
        ]["reserved_p95"]["prompt_tokens"] += 1
    with pytest.raises(orchestrator.PilotOrchestrationError, match=expected):
        orchestrator._runner_p95_reservations(
            contract,
            model_id,
            raw_root=tmp_path,
            paid=_paid(),
        )


def test_v2101_development_fake_matrix_is_39_diagnostic_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("V2.10.1 fake matrix attempted a live provider")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    result = orchestrator.run_development_fake_matrix(
        contract_path=CONTRACT_PATH,
        resume=False,
        raw_root=tmp_path,
    )

    assert result["status"] == "pass"
    assert result["registered_cells"] == 39
    assert result["status_counts"] == {"complete": 39}
    assert result["diagnostic_only"] is True
    assert result["scientific_evidence"] is False


@pytest.mark.parametrize(
    ("contract_path", "git_tag"),
    (
        (CONTRACT_PATH, "pilot-v2.10.1-science"),
        (V210_CONTRACT_PATH, "pilot-v2.10-science"),
    ),
)
def test_imported_stage0_second_reservation_failure_is_fully_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_path: Path,
    git_tag: str,
) -> None:
    contract = load_pilot_contract(contract_path)
    paid = orchestrator.GitProvenance(
        git_tag=git_tag,
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )
    raw_root = tmp_path / contract.contract_id / "raw"
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    for stage_id in ("parent-import", "q-ref-resolution"):
        prerequisite = contract.expand(stage=stage_id)[0]
        artifact = raw_root / stage_id / f"{prerequisite.run_id}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
        ledger.finalize(
            prerequisite.run_id,
            status="complete",
            artifact=str(artifact),
            failure=None,
        )

    budget = PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    )
    real_reserve = budget.reserve
    reserve_attempts = 0

    def fail_second_reservation(projection: RunProjection) -> None:
        nonlocal reserve_attempts
        reserve_attempts += 1
        if reserve_attempts == 2:
            raise OSError("injected second Stage-0 reservation failure")
        real_reserve(projection)

    monkeypatch.setattr(budget, "reserve", fail_second_reservation)
    provider_constructions = 0

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        nonlocal provider_constructions
        provider_constructions += 1
        raise AssertionError("imported Stage-0 constructed a provider")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden_provider)

    specs = tuple(contract.expand(stage="stage0-calibration"))
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="Stage-0 import/replay failed",
    ):
        orchestrator._execute_imported_stage0_stage(
            contract,
            specs,
            raw_root=raw_root,
            paid=paid,
            run_ledger=ledger,
            budget_ledger=budget,
        )

    stage0_ids = {spec.run_id for spec in specs}
    first_budget = budget.snapshot()["runs"]
    assert set(first_budget) == stage0_ids
    for spec in specs:
        assert first_budget[spec.run_id]["reservation"] == RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=0.0,
            completions=0,
            storage_bytes=2_000_000,
            basis={
                "method": (
                    f"{orchestrator._imported_stage0_label(contract).lower()}-"
                    "imported-stage0-envelope"
                ),
                "provider_calls": 0,
                "hosted_completion_cap_counted": False,
            },
        ).to_dict()
    assert {row["status"] for row in first_budget.values()} == {
        "integrity-stopped"
    }
    assert all(
        row["actual"]["cost_usd"] == 0.0
        and row["actual"]["completions"] == 0
        and row["actual"]["storage_bytes"]
        <= row["reservation"]["storage_bytes"]
        for row in first_budget.values()
    )
    assert Counter(
        row["status"] for row in ledger.snapshot()["runs"].values()
    ) == {
        "complete": 2,
        "integrity-stopped": 209,
    }
    receipt_path = orchestrator._stage_receipt_path(
        raw_root,
        "stage0-calibration",
    )
    first_receipt_bytes = receipt_path.read_bytes()
    first_receipt = orchestrator._read_json(receipt_path)
    assert first_receipt["status"] == "integrity-stopped"
    assert first_receipt["status_counts"] == {"integrity-stopped": 14}
    assert first_receipt["failure"]["error_type"] == "OSError"
    assert first_receipt["failure"]["message_bytes"] > 0
    assert len(first_receipt["failure"]["message_sha256"]) == 64
    assert "traceback" not in first_receipt["failure"]
    assert provider_constructions == 0

    resumed = orchestrator._execute_imported_stage0_stage(
        contract,
        specs,
        raw_root=raw_root,
        paid=paid,
        run_ledger=ledger,
        budget_ledger=budget,
    )

    assert resumed == first_receipt
    assert receipt_path.read_bytes() == first_receipt_bytes
    assert budget.snapshot()["runs"] == first_budget
    assert provider_constructions == 0
