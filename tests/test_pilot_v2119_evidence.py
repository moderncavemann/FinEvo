from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidenceError
from verified_memory import pilot_orchestrator, pilot_v2119_evidence
from verified_memory.pilot_v2119_continuation import (
    PilotV2119ContinuationError,
    parent_budget_debit_for_v2119,
)
from verified_memory.pilot_v2119_evidence import (
    CURRENT_LEDGER_DENOMINATOR,
    LOGICAL_REGISTERED_DENOMINATOR,
    LOGICAL_SCIENTIFIC_DENOMINATOR,
    _audit_current_budget,
    _absent_owner_stop,
    _budget_owner_mapping,
    _expected_mapping,
    _failure_artifact_evidence,
    _observed_owner_linkage,
    _package_target,
    _resolve_current_paths,
    _validate_current_git,
    _verify_package_tree,
    _write_package,
    assemble_v2119_terminal_evidence,
    build_pilot_v2119_evidence_package,
    inherited_capability_by_model,
)


ROOT = Path(__file__).resolve().parents[1]


def _contracts():
    return (
        load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml"),
        load_pilot_contract(ROOT / "experiments/pilot_v2_11_5.yaml"),
    )


def _normalized_row(spec, *, status: str = "complete", value: float = 1.0):
    scientific = spec.stage_id in {
        "experiment-a",
        "experiment-c",
        "experiment-d",
        "experiment-b",
        "cross-model",
    }
    return {
        **spec.to_dict(),
        "status": status,
        "failure": (
            None
            if status == "complete"
            else {
                "error_type": "FixtureTerminalFailure",
                "message": "preregistered fixture failure",
            }
        ),
        "artifact_kind": (
            "verified-run" if status == "complete" else "failure-audit-artifact"
        ),
        "artifact_sha256": "f" * 64,
        "scientific_eligible": bool(status == "complete" and scientific),
        "metrics": {
            "utility": {"shock_recovery_discounted": value},
            "total_discounted_utility": value,
            "guardrails": {"provider_failure_count": 0},
        },
        "gate_evidence": {},
        "capability": {},
        "narrative": {},
    }


def _capability_summary(model_id: str):
    denominators = {
        "utility-ranking": 12,
        "rule-application": 12,
        "rule-proposal": 6,
    }
    return {
        "model_id": model_id,
        "runtime_model": (
            "openai/gpt-5.2-2025-12-11"
            if model_id == "gpt52_main"
            else "openai/gpt-5.6-sol"
        ),
        "requested_model": (
            "gpt-5.2-2025-12-11" if model_id == "gpt52_main" else "gpt-5.6-sol"
        ),
        "capability_pass": True,
        "interface_pass": True,
        "parse_failure_count": 0,
        "provider_failure_count": 0,
        "category_totals": {
            name: {
                "denominator": denominator,
                "registered_total": denominator,
                "correct": denominator,
                "required": 5 if name == "rule-proposal" else 10,
            }
            for name, denominator in denominators.items()
        },
        "source_wrapper_content_sha256": (
            "be8684bd1208bb5049be744910c10bdaf5f48e69ad6f13ae086ecef9ce42e32f"
            if model_id == "gpt52_main"
            else "f3a3025347327545e33d42149efe1ed0d29c3279429b3f379ecc88c6cdeab863"
        ),
    }


def _authority_fixture(parent, receipt):
    stable = {}
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        summary = receipt["capability_authority"][model_id]
        row = {
            "source_kind": "sealed-closed-loop-observed-p95",
            "source_model_id": model_id,
            "source_served_model": summary["requested_model"],
            "source_preflight_run_id": f"v2112-{model_id}-preflight",
            "source_preflight_run_spec_sha256": "1" * 64,
            "source_execution_artifact_sha256": "2" * 64,
            "source_provider_call_journal_sha256": "3" * 64,
        }
        stable[summary["runtime_model"]] = {
            "action": deepcopy(row),
            "semantic": deepcopy(row),
        }
    source_gate = {
        "path": "experiment_results/pilot-v2.11.5/raw/long-context-preflight/post_gate_authority.json",
        "file_sha256": "4" * 64,
        "content_sha256": "5" * 64,
    }
    receipt["dispatch_authority_source"] = {
        "source_gate": deepcopy(source_gate),
        "stable_source_authorities": deepcopy(stable),
    }
    return {
        "authority_release": {"source_gate": source_gate},
        "stable_source_authorities": stable,
    }


def _preflight_fixture(receipt):
    sources = {}
    for model_id, summary in receipt["capability_authority"].items():
        row = receipt["dispatch_authority_source"]["stable_source_authorities"][
            summary["runtime_model"]
        ]["action"]
        sources[model_id] = {
            "source_preflight": {
                "run_id": row["source_preflight_run_id"],
                "run_spec_sha256": row["source_preflight_run_spec_sha256"],
                "model_id": row["source_model_id"],
                "served_model": row["source_served_model"],
                "execution_artifact_sha256": row["source_execution_artifact_sha256"],
                "provider_call_journal_sha256": row[
                    "source_provider_call_journal_sha256"
                ],
            }
        }
    return {
        "available": True,
        "go": True,
        "path": (
            "/fixture/authority/"
            + receipt["dispatch_authority_source"]["source_gate"]["path"]
        ),
        "file_sha256": receipt["dispatch_authority_source"]["source_gate"][
            "file_sha256"
        ],
        "content_sha256": receipt["dispatch_authority_source"]["source_gate"][
            "content_sha256"
        ],
        "denominator": {"eligible_model_ids": ["gpt52_main", "gpt56_diagnostic"]},
        "authority_sources": sources,
        "operational_imports": {
            model_id: {
                "provider_calls_current_attempt": 0,
                "provider_construction_current_attempt": False,
            }
            for model_id in sources
        },
    }


def _parent_capability_fixture(parent_rows):
    return {
        row["model_id"]: {"capability": deepcopy(row["capability"])}
        for row in parent_rows
        if row["stage_id"] == "capability-gate"
    }


def _terminal_fixture():
    current, parent = _contracts()
    parent_rows = []
    failed_a = {spec.run_id for spec in parent.expand(stage="experiment-a")[-3:]}
    for stage_id in (
        "parent-import",
        "capability-gate",
        "long-context-preflight",
        "experiment-c",
        "experiment-a",
    ):
        for spec in parent.expand(stage=stage_id):
            status = "failed" if spec.run_id in failed_a else "complete"
            row = _normalized_row(spec, status=status, value=1.0)
            if stage_id == "capability-gate":
                row["capability"] = {"capability": _capability_summary(spec.model_id)}
            parent_rows.append(row)

    cross_seeds = tuple(current.seeds["sets"]["cross-model"])
    current_rows = []
    for spec in current.expand():
        value = 1.0
        if spec.stage_id == "experiment-b":
            value = 110.0 if spec.arm_id == "full" else 100.0
        elif spec.stage_id == "cross-model":
            value = 210.0 if spec.arm_id == "full" else 200.0
        # Keep non-cross-model B seeds deterministic too; the cross aggregator
        # must nevertheless use exactly the first three preregistered seeds.
        if spec.stage_id == "experiment-b" and spec.environment_seed not in cross_seeds:
            value += 0.5
        current_rows.append(_normalized_row(spec, value=value))

    receipt = {
        "canonical_remaining_cell_mapping": _expected_mapping(current, parent),
        "capability_authority": {
            model_id: _capability_summary(model_id)
            for model_id in ("gpt52_main", "gpt56_diagnostic")
        },
    }
    _authority_fixture(parent, receipt)
    gates = {
        "experiment_a": {
            "status": "no-go",
            "scientific_evidence_complete": False,
            "support_retrieval_effect": False,
            "claim_action": "retain route traceability only",
            "authority_binding": deepcopy(
                current.to_dict()["v2119_recovery_boundary"]["parent_stage_receipts"][
                    "experiment-a"
                ]
            ),
        },
        "experiment_c": {
            "status": "no-go",
            "scientific_evidence_complete": False,
            "support_rule_reliability": False,
            "claim_action": "withdraw or narrow the rule-reliability claim",
            "authority_binding": deepcopy(
                current.to_dict()["v2119_recovery_boundary"]["parent_stage_receipts"][
                    "experiment-c"
                ]
            ),
        },
    }
    return current, parent, parent_rows, current_rows, receipt, gates


def _assemble(fixture):
    current, parent, parent_rows, current_rows, receipt, gates = fixture
    return assemble_v2119_terminal_evidence(
        contract=current,
        parent_contract=parent,
        parent_terminal_rows=parent_rows,
        current_rows=current_rows,
        parent_import_receipt=receipt,
        parent_capability_by_model=_parent_capability_fixture(parent_rows),
        parent_preflight_authority=_preflight_fixture(receipt),
        current_authority=_authority_fixture(parent, deepcopy(receipt)),
        external_parent_gates=gates,
    )


def test_build_rejects_provider_key_before_paths_or_provider(monkeypatch, tmp_path):
    def forbidden_path_resolution(*args, **kwargs):  # pragma: no cover
        raise AssertionError("publication inspected paths before key guard")

    def forbidden_provider(*args, **kwargs):  # pragma: no cover
        raise AssertionError("publication constructed a provider")

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-present")
    monkeypatch.setattr(
        pilot_v2119_evidence,
        "_resolve_current_paths",
        forbidden_path_resolution,
    )
    monkeypatch.setattr(pilot_orchestrator, "_provider_for_profile", forbidden_provider)
    with pytest.raises(PilotV2119ContinuationError, match="credentials.*present"):
        build_pilot_v2119_evidence_package(
            contract_path=tmp_path / "missing-contract.yaml",
            run_ledger_path=tmp_path / "missing-run-ledger.json",
            raw_root=tmp_path / "missing-raw",
            build_root=tmp_path / "missing-build",
            source_repo_root=tmp_path / "missing-source",
            authority_repo_root=tmp_path / "missing-authority",
        )


def test_build_wraps_entire_consumer_in_provider_sentinels(monkeypatch, tmp_path):
    for name in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    def malicious_guarded_builder(**kwargs):  # pragma: no cover
        return pilot_orchestrator._provider_for_profile(None)

    monkeypatch.setattr(
        pilot_v2119_evidence,
        "_build_pilot_v2119_evidence_package_guarded",
        malicious_guarded_builder,
    )
    with pytest.raises(Exception, match="provider/catalog construction is forbidden"):
        build_pilot_v2119_evidence_package(
            contract_path=tmp_path / "unused-contract",
            run_ledger_path=tmp_path / "unused-ledger",
            raw_root=tmp_path / "unused-raw",
            build_root=tmp_path / "unused-build",
        )


def test_complete_87_to_136_terminal_fixture_is_provider_free(monkeypatch):
    def forbidden_provider(*args, **kwargs):  # pragma: no cover - called on regression
        raise AssertionError("publication constructed a provider")

    monkeypatch.setattr(pilot_orchestrator, "_provider_for_profile", forbidden_provider)
    aggregate = _assemble(_terminal_fixture())

    denominator = aggregate["denominator"]
    assert denominator["current_release"]["row_count"] == CURRENT_LEDGER_DENOMINATOR
    assert denominator["current_release"]["scientific_row_count"] == 86
    assert denominator["logical_v2115_matrix"]["row_count"] == (
        LOGICAL_REGISTERED_DENOMINATOR
    )
    assert denominator["logical_v2115_matrix"]["scientific_row_count"] == (
        LOGICAL_SCIENTIFIC_DENOMINATOR
    )
    assert denominator["logical_v2115_matrix"]["current_parent_meta_row_excluded"]
    assert denominator["logical_v2115_matrix"]["itt_failures_retained"] == 3
    assert aggregate["publication_status"] == {
        "terminal_evidence_complete": True,
        "all_preregistered_claims_supported": False,
        "classification": "terminal-complete-with-preregistered-no-go",
        "negative_results_retained": True,
    }
    assert aggregate["claim_gates"]["experiment_a"]["status"] == "no-go"
    assert aggregate["claim_gates"]["experiment_c"]["status"] == "no-go"
    assert aggregate["claim_gates"]["experiment_d"]["status"] == "no-go"
    assert aggregate["experiment_b"]["arms"]["full"]["registered_seeds"] == 5
    assert set(aggregate["cross_model"]) == {
        "gpt52_main",
        "gpt56_diagnostic",
    }
    for model in aggregate["cross_model"].values():
        assert model["usable_paired_seeds"] == [
            1099057501,
            1421875452,
            1769977770,
        ]
        assert model["direction"] == "positive"
        assert model["directional_micro_pilot_replication"] is True
        assert model["matched_a_a_null_registered"] is False
        assert model["repeatability_or_effect_size_claim_allowed"] is False
    assert all(
        item["provider_calls_current_attempt"] == 0
        for item in aggregate["inherited_capability"].values()
    )


def test_missing_current_cell_fails_closed():
    fixture = list(_terminal_fixture())
    fixture[3] = fixture[3][:-1]
    with pytest.raises(PilotEvidenceError, match="preregistered denominator"):
        _assemble(tuple(fixture))


def test_nonterminal_current_cell_stays_in_itt_and_blocks_publication():
    fixture = list(_terminal_fixture())
    fixture[3] = deepcopy(fixture[3])
    fixture[3][1]["status"] = "scheduled"
    fixture[3][1]["artifact_kind"] = None
    fixture[3][1]["scientific_eligible"] = False
    with pytest.raises(PilotEvidenceError, match="nonterminal ITT row"):
        _assemble(tuple(fixture))


def test_current_parent_row_cannot_replace_a_continuation_cell():
    fixture = list(_terminal_fixture())
    fixture[3] = deepcopy(fixture[3])
    parent_row = next(row for row in fixture[3] if row["stage_id"] == "parent-import")
    fixture[3][-1] = deepcopy(parent_row)
    with pytest.raises(PilotEvidenceError, match="missing/duplicate run id"):
        _assemble(tuple(fixture))


def test_inherited_capability_requires_exact_two_model_authority():
    fixture = _terminal_fixture()
    receipt = deepcopy(fixture[4])
    receipt["capability_authority"]["unregistered_model"] = _capability_summary(
        "gpt52_main"
    )
    with pytest.raises(PilotEvidenceError, match="model denominator drifted"):
        inherited_capability_by_model(
            receipt,
            parent_capability_by_model=_parent_capability_fixture(fixture[2]),
            parent_preflight_authority=_preflight_fixture(receipt),
            current_authority=_authority_fixture(fixture[1], deepcopy(receipt)),
        )


def test_inherited_capability_denominator_tamper_fails_closed():
    fixture = _terminal_fixture()
    receipt = deepcopy(fixture[4])
    receipt["capability_authority"]["gpt52_main"]["category_totals"]["rule-proposal"][
        "denominator"
    ] = 5
    with pytest.raises(PilotEvidenceError, match="capability drifted"):
        inherited_capability_by_model(
            receipt,
            parent_capability_by_model=_parent_capability_fixture(fixture[2]),
            parent_preflight_authority=_preflight_fixture(receipt),
            current_authority=_authority_fixture(fixture[1], deepcopy(receipt)),
        )


@pytest.mark.parametrize(
    ("gate_name", "support_key"),
    [
        ("experiment_a", "support_retrieval_effect"),
        ("experiment_c", "support_rule_reliability"),
    ],
)
def test_parent_no_go_cannot_be_promoted(gate_name, support_key):
    fixture = list(_terminal_fixture())
    fixture[5] = deepcopy(fixture[5])
    fixture[5][gate_name]["status"] = "supported"
    fixture[5][gate_name][support_key] = True
    with pytest.raises(PilotEvidenceError, match="must remain external no-go"):
        _assemble(tuple(fixture))


def test_mapping_tamper_fails_before_aggregation():
    fixture = list(_terminal_fixture())
    fixture[4] = deepcopy(fixture[4])
    fixture[4]["canonical_remaining_cell_mapping"]["rows"][0]["logical_cell_sha256"] = (
        "0" * 64
    )
    with pytest.raises(PilotEvidenceError, match="mapping drifted"):
        _assemble(tuple(fixture))


class _SnapshotLedger:
    def __init__(self, rows):
        self._snapshot = {
            "runs": {
                row["run_id"]: {
                    "spec": {
                        key: row[key]
                        for key in (
                            "run_id",
                            "contract_id",
                            "stage_id",
                            "model_id",
                            "requested_model",
                            "arm_id",
                            "narrative_id",
                            "environment_seed",
                            "decoding_seed",
                            "utility_profile_id",
                            "shock_id",
                            "budget_bucket",
                            "num_agents",
                            "episode_length",
                            "execution_mode",
                        )
                    },
                    "status": row["status"],
                    "failure": row["failure"],
                    "artifact": (
                        "fixture.json" if row["status"] == "complete" else None
                    ),
                }
                for row in rows
            }
        }

    def snapshot(self):
        return deepcopy(self._snapshot)


class _SnapshotBudget:
    def __init__(self, snapshot):
        self._snapshot = deepcopy(snapshot)

    def snapshot(self):
        return deepcopy(self._snapshot)


def _terminal_budget_fixture(tmp_path, current, current_rows):
    budget = PilotBudgetLedger(
        tmp_path / "budget_ledger.json",
        contract_hash=current.canonical_hash,
        caps=pilot_orchestrator._budget_caps(current),
        tamper_evident=True,
        parent_debit=parent_budget_debit_for_v2119(current),
    )
    owners = _budget_owner_mapping(current)
    parent_spec = tuple(current.expand(stage="parent-import"))[0]
    parent_projection = pilot_orchestrator._v2119_parent_import_projection(parent_spec)
    budget.reserve(parent_projection)
    budget.finalize(
        parent_projection.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=1,
    )
    prefix = budget.snapshot()
    budget.bind_acceptance_receipt(
        receipt_schema_version=(
            "finevo-pilot-v2.11.9-scientific-dispatch-acceptance-v1"
        ),
        receipt_path=(
            "experiment_results/pilot-v2.11.9/raw/"
            "scientific_dispatch_acceptance.json"
        ),
        receipt_content_sha256="c" * 64,
        accepted_run_event_count=2,
        accepted_run_event_chain_head="d" * 64,
        accepted_budget_event_count=len(prefix["events"]),
        accepted_budget_event_chain_head=prefix["events"][-1]["event_sha256"],
    )
    projections = {}
    for budget_id in owners:
        if budget_id == parent_projection.run_id:
            continue
        projection = RunProjection(
            run_id=budget_id,
            stage_bucket="hosted_v2119",
            cost_usd=0.0,
            completions=0,
            storage_bytes=1,
            basis={"fixture": "provider-free-terminal-budget"},
        )
        budget.reserve(projection)
        budget.finalize(
            budget_id,
            status="complete",
            cost_usd=0.0,
            completions=0,
            storage_bytes=1,
        )
        projections[budget_id] = canonical_sha256(projection.to_dict())
    acceptance = {"budget_projection": {"projection_sha256_by_run_id": projections}}
    return budget, acceptance


def test_terminal_budget_fixture_maps_37_units_to_all_87_itt_rows(tmp_path):
    current, _, _, current_rows, _, _ = _terminal_fixture()
    owners = _budget_owner_mapping(current)
    assert len(owners) == 37
    d_owners = {
        run_id: linked
        for run_id, linked in owners.items()
        if "--experiment-d--" in run_id
    }
    assert len(d_owners) == 5
    assert {len(linked) for linked in d_owners.values()} == {11}
    assert sum(len(linked) for linked in owners.values()) == 87

    budget, acceptance = _terminal_budget_fixture(tmp_path, current, current_rows)
    audit = _audit_current_budget(
        current,
        raw_root=tmp_path,
        budget=budget,
        current_rows=current_rows,
        acceptance=acceptance,
        run_ledger=_SnapshotLedger(current_rows),
    )
    assert audit["budget_owner_universe_count"] == 37
    assert audit["observed_budget_unit_count"] == 37
    assert audit["absent_unreserved_owner_count"] == 0
    assert audit["experiment_d_group_count"] == 5
    assert audit["linked_itt_row_count"] == 87
    assert len(audit["owner_rows"]) == 37


def test_terminal_budget_status_must_match_all_linked_d_rows(tmp_path):
    current, _, _, current_rows, _, _ = _terminal_fixture()
    budget, acceptance = _terminal_budget_fixture(tmp_path, current, current_rows)
    tampered = deepcopy(current_rows)
    d_row = next(row for row in tampered if row["stage_id"] == "experiment-d")
    d_row["status"] = "failed"
    d_row["failure"] = {"error_type": "Tampered", "message": "status mismatch"}
    d_row["scientific_eligible"] = False
    with pytest.raises(PilotEvidenceError, match="budget/ITT replay failed"):
        _audit_current_budget(
            current,
            raw_root=tmp_path,
            budget=budget,
            current_rows=tampered,
            acceptance=acceptance,
            run_ledger=_SnapshotLedger(tampered),
        )


def test_foreign_budget_event_type_is_rejected(tmp_path):
    current, _, _, current_rows, _, _ = _terminal_fixture()
    budget, acceptance = _terminal_budget_fixture(tmp_path, current, current_rows)
    snapshot = budget.snapshot()
    snapshot["events"].append(
        {
            "event_type": "foreign_reclassified_result",
            "payload": {"scientific_evidence": True},
            "event_index": len(snapshot["events"]),
            "previous_event_sha256": snapshot["events"][-1]["event_sha256"],
            "event_sha256": "e" * 64,
        }
    )
    with pytest.raises(PilotEvidenceError, match="event type/count inventory"):
        _audit_current_budget(
            current,
            raw_root=tmp_path,
            budget=_SnapshotBudget(snapshot),
            current_rows=current_rows,
            acceptance=acceptance,
            run_ledger=_SnapshotLedger(current_rows),
        )


def test_single_run_finalized_before_itt_recovery_is_publishable(tmp_path):
    current, _, _, current_rows, _, _ = _terminal_fixture()
    budget, acceptance = _terminal_budget_fixture(tmp_path, current, current_rows)
    recovered = deepcopy(current_rows)
    row = next(
        item
        for item in recovered
        if item["stage_id"] == "experiment-b" and item["arm_id"] == "no-memory"
    )
    row["status"] = "integrity-stopped"
    row["failure"] = {
        "error_type": "BudgetFinalizedBeforeITT",
        "message": (
            "a prior process created budget state without a terminal ITT cell; "
            "the cell is retained and is not redispatched"
        ),
    }
    row["scientific_eligible"] = False
    row["artifact_kind"] = "terminal-failure-without-artifact"
    row["artifact_sha256"] = None
    audit = _audit_current_budget(
        current,
        raw_root=tmp_path,
        budget=budget,
        current_rows=recovered,
        acceptance=acceptance,
        run_ledger=_SnapshotLedger(recovered),
    )
    assert audit["owner_rows"][row["run_id"]]["linkage"] == {
        "classification": "finalized-before-itt-recovery",
        "direct_terminal_count": 0,
        "exact_recovery_count": 1,
        "pre_reservation_original_terminal_count": 0,
        "original_terminal_after_budget_recovery_count": 0,
        "linked_status_counts": {"integrity-stopped": 1},
    }


def test_d_finalized_before_partial_itt_recovery_is_publishable(tmp_path):
    current, _, _, current_rows, _, _ = _terminal_fixture()
    budget, acceptance = _terminal_budget_fixture(tmp_path, current, current_rows)
    recovered = deepcopy(current_rows)
    owners = _budget_owner_mapping(current)
    group_id, linked = next(
        (run_id, values)
        for run_id, values in owners.items()
        if "--experiment-d--" in run_id
    )
    specs = {spec.run_id: spec for spec in current.expand(stage="experiment-d")}
    focal = specs[linked[0]]
    failure = {
        "error_type": "BudgetFinalizedBeforeITT",
        "message": (
            "a prior process created shared Experiment D budget state without an "
            "exact terminal ITT group; no redispatch is permitted"
        ),
        "model_id": focal.model_id,
        "environment_seed": focal.environment_seed,
        "provider_dispatch_started": False,
        "stop_origin": "pre-catalog-interrupted-reservation-recovery",
    }
    by_id = {row["run_id"]: row for row in recovered}
    for run_id in linked[1:]:
        by_id[run_id]["status"] = "integrity-stopped"
        by_id[run_id]["failure"] = deepcopy(failure)
        by_id[run_id]["scientific_eligible"] = False
        by_id[run_id]["artifact_kind"] = "terminal-failure-without-artifact"
        by_id[run_id]["artifact_sha256"] = None
    audit = _audit_current_budget(
        current,
        raw_root=tmp_path,
        budget=budget,
        current_rows=recovered,
        acceptance=acceptance,
        run_ledger=_SnapshotLedger(recovered),
    )
    linkage = audit["owner_rows"][group_id]["linkage"]
    assert linkage["classification"] == "finalized-before-itt-recovery"
    assert linkage["direct_terminal_count"] == 1
    assert linkage["exact_recovery_count"] == 10


def test_early_d_stop_allows_absent_unreserved_future_owners(tmp_path):
    current, _, _, current_rows, _, _ = _terminal_fixture()
    stopped = deepcopy(current_rows)
    for row in stopped:
        if row["stage_id"] == "parent-import":
            continue
        if row["stage_id"] == "experiment-d":
            failure = {
                "error_type": "PreDispatchIntegrityStop",
                "cause_type": "ReleaseBindingDrift",
                "message": "fixture pre-dispatch stop",
                "model_id": row["model_id"],
                "environment_seed": row["environment_seed"],
                "provider_dispatch_started": False,
                "stop_origin": "experiment-d-pre-provider-revalidation",
            }
        else:
            failure = {
                "error_type": "StageExecutionNoGo",
                "message": "Experiment D contains a budget or integrity hard stop",
                "source_stage": "experiment-d",
                "blocked_stage": row["stage_id"],
            }
        row["status"] = "integrity-stopped"
        row["failure"] = failure
        row["scientific_eligible"] = False
        row["artifact_kind"] = "terminal-failure-without-artifact"
        row["artifact_sha256"] = None

    budget = PilotBudgetLedger(
        tmp_path / "budget_ledger.json",
        contract_hash=current.canonical_hash,
        caps=pilot_orchestrator._budget_caps(current),
        tamper_evident=True,
        parent_debit=parent_budget_debit_for_v2119(current),
    )
    parent_spec = tuple(current.expand(stage="parent-import"))[0]
    parent_projection = pilot_orchestrator._v2119_parent_import_projection(parent_spec)
    budget.reserve(parent_projection)
    budget.finalize(
        parent_spec.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=1,
    )
    prefix = budget.snapshot()
    budget.bind_acceptance_receipt(
        receipt_schema_version=(
            "finevo-pilot-v2.11.9-scientific-dispatch-acceptance-v1"
        ),
        receipt_path=(
            "experiment_results/pilot-v2.11.9/raw/"
            "scientific_dispatch_acceptance.json"
        ),
        receipt_content_sha256="c" * 64,
        accepted_run_event_count=2,
        accepted_run_event_chain_head="d" * 64,
        accepted_budget_event_count=len(prefix["events"]),
        accepted_budget_event_chain_head=prefix["events"][-1]["event_sha256"],
    )
    science_ids = set(_budget_owner_mapping(current)) - {parent_spec.run_id}
    acceptance = {
        "budget_projection": {
            "projection_sha256_by_run_id": {run_id: "a" * 64 for run_id in science_ids}
        }
    }
    audit = _audit_current_budget(
        current,
        raw_root=tmp_path,
        budget=budget,
        current_rows=stopped,
        acceptance=acceptance,
        run_ledger=_SnapshotLedger(stopped),
    )
    assert audit["budget_owner_universe_count"] == 37
    assert audit["observed_budget_unit_count"] == 1
    assert audit["absent_unreserved_owner_count"] == 36
    assert len(audit["absent_owner_rows"]) == 36


def test_absent_budget_owner_requires_exact_undispatched_failure(tmp_path):
    current, _, _, current_rows, _, _ = _terminal_fixture()
    stopped = deepcopy(current_rows)
    for row in stopped:
        if row["stage_id"] == "parent-import":
            continue
        row["status"] = "integrity-stopped"
        row["failure"] = {"error_type": "UnregisteredStop", "message": "bad"}
        row["scientific_eligible"] = False
        row["artifact_kind"] = "terminal-failure-without-artifact"
        row["artifact_sha256"] = None
    budget = PilotBudgetLedger(
        tmp_path / "budget_ledger.json",
        contract_hash=current.canonical_hash,
        caps=pilot_orchestrator._budget_caps(current),
        tamper_evident=True,
        parent_debit=parent_budget_debit_for_v2119(current),
    )
    parent_spec = tuple(current.expand(stage="parent-import"))[0]
    projection = pilot_orchestrator._v2119_parent_import_projection(parent_spec)
    budget.reserve(projection)
    budget.finalize(
        parent_spec.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=1,
    )
    prefix = budget.snapshot()
    budget.bind_acceptance_receipt(
        receipt_schema_version=(
            "finevo-pilot-v2.11.9-scientific-dispatch-acceptance-v1"
        ),
        receipt_path=(
            "experiment_results/pilot-v2.11.9/raw/"
            "scientific_dispatch_acceptance.json"
        ),
        receipt_content_sha256="c" * 64,
        accepted_run_event_count=2,
        accepted_run_event_chain_head="d" * 64,
        accepted_budget_event_count=len(prefix["events"]),
        accepted_budget_event_chain_head=prefix["events"][-1]["event_sha256"],
    )
    science_ids = set(_budget_owner_mapping(current)) - {parent_spec.run_id}
    acceptance = {
        "budget_projection": {
            "projection_sha256_by_run_id": {run_id: "a" * 64 for run_id in science_ids}
        }
    }
    with pytest.raises(PilotEvidenceError, match="exact undispatched stop"):
        _audit_current_budget(
            current,
            raw_root=tmp_path,
            budget=budget,
            current_rows=stopped,
            acceptance=acceptance,
            run_ledger=_SnapshotLedger(stopped),
        )


def test_absent_owner_accepts_exact_prerequisite_and_b_propagation_only():
    current, _, _, current_rows, _, _ = _terminal_fixture()
    owners = _budget_owner_mapping(current)
    rows = {row["run_id"]: deepcopy(row) for row in current_rows}

    b_id, b_linked = next(
        (run_id, linked)
        for run_id, linked in owners.items()
        if "--experiment-b--" in run_id
    )
    b_row = rows[b_linked[0]]
    b_row.update(
        {
            "status": "integrity-stopped",
            "failure": {
                "error_type": "PrerequisiteNoGo",
                "cause_type": "PilotOrchestrationError",
                "message": "sealed prerequisite fixture no-go",
                "source_stage": "experiment-b",
            },
            "artifact_kind": "terminal-failure-without-artifact",
            "artifact_sha256": None,
        }
    )
    assert _absent_owner_stop(
        current,
        budget_id=b_id,
        linked_ids=b_linked,
        run_rows=rows,
    )["disposition_counts"] == {"prerequisite-no-go": 1}

    cross_id, cross_linked = next(
        (run_id, linked)
        for run_id, linked in owners.items()
        if "--cross-model--" in run_id
    )
    cross_row = rows[cross_linked[0]]
    cross_row.update(
        {
            "status": "integrity-stopped",
            "failure": {
                "error_type": "StageExecutionNoGo",
                "message": (
                    "experiment-b contains a budget/integrity hard stop or "
                    "lacks its mandatory pre-science selection"
                ),
                "source_stage": "experiment-b",
                "blocked_stage": "cross-model",
            },
            "artifact_kind": "terminal-failure-without-artifact",
            "artifact_sha256": None,
        }
    )
    assert _absent_owner_stop(
        current,
        budget_id=cross_id,
        linked_ids=cross_linked,
        run_rows=rows,
    )["disposition_counts"] == {"ancestor-stage-execution-no-go": 1}

    cross_row["failure"]["source_stage"] = "experiment-d"
    with pytest.raises(PilotEvidenceError, match="exact undispatched stop"):
        _absent_owner_stop(
            current,
            budget_id=cross_id,
            linked_ids=cross_linked,
            run_rows=rows,
        )
    cross_row["failure"] = {
        "error_type": "ProjectionNoGo",
        "message": "fabricated scope",
        "projection_scope": "anything",
    }
    with pytest.raises(PilotEvidenceError, match="exact undispatched stop"):
        _absent_owner_stop(
            current,
            budget_id=cross_id,
            linked_ids=cross_linked,
            run_rows=rows,
        )


def test_stage_stopped_requires_earlier_exact_same_stage_initiator():
    current, _, _, current_rows, _, _ = _terminal_fixture()
    owners = _budget_owner_mapping(current)
    rows = {row["run_id"]: deepcopy(row) for row in current_rows}
    b_specs = tuple(current.expand(stage="experiment-b"))
    initiator_id = b_specs[0].run_id
    tail_id = b_specs[1].run_id
    rows[initiator_id].update(
        {
            "status": "budget-stopped",
            "failure": {
                "error_type": "PilotBudgetError",
                "message": "reservation exceeds the frozen stage cap",
            },
            "artifact_kind": "terminal-failure-without-artifact",
            "artifact_sha256": None,
        }
    )
    rows[tail_id].update(
        {
            "status": "budget-stopped",
            "failure": {
                "error_type": "StageStopped",
                "message": "an earlier budget/integrity failure stopped this stage",
            },
            "artifact_kind": "terminal-failure-without-artifact",
            "artifact_sha256": None,
        }
    )
    assert _absent_owner_stop(
        current,
        budget_id=initiator_id,
        linked_ids=owners[initiator_id],
        run_rows=rows,
    )["disposition_counts"] == {"per-spec-pre-dispatch-budget-stop": 1}
    assert _absent_owner_stop(
        current,
        budget_id=tail_id,
        linked_ids=owners[tail_id],
        run_rows=rows,
    )["disposition_counts"] == {"own-stage-tail-stop": 1}

    rows[initiator_id] = deepcopy(
        next(row for row in current_rows if row["run_id"] == initiator_id)
    )
    with pytest.raises(PilotEvidenceError, match="exact undispatched stop"):
        _absent_owner_stop(
            current,
            budget_id=tail_id,
            linked_ids=owners[tail_id],
            run_rows=rows,
        )

    reserved_failure = {
        "error_type": "PilotBudgetError",
        "message": "reserved execution exceeded frozen budget",
    }
    rows[initiator_id].update(
        {
            "status": "budget-stopped",
            "failure": deepcopy(reserved_failure),
            "artifact_kind": "failure-audit-artifact",
            "artifact_sha256": "a" * 64,
        }
    )
    observed_budget_rows = {
        initiator_id: {
            "status": "budget-stopped",
            "failure": deepcopy(reserved_failure),
        }
    }
    assert _absent_owner_stop(
        current,
        budget_id=tail_id,
        linked_ids=owners[tail_id],
        run_rows=rows,
        observed_budget_rows=observed_budget_rows,
    )["disposition_counts"] == {"own-stage-tail-stop": 1}

    observed_budget_rows[initiator_id]["failure"]["message"] = "drifted"
    with pytest.raises(PilotEvidenceError, match="exact undispatched stop"):
        _absent_owner_stop(
            current,
            budget_id=tail_id,
            linked_ids=owners[tail_id],
            run_rows=rows,
            observed_budget_rows=observed_budget_rows,
        )


def test_terminal_failure_without_artifact_is_retained(tmp_path):
    assert _failure_artifact_evidence(tmp_path, None) == {
        "artifact_kind": "terminal-failure-without-artifact",
        "artifact_sha256": None,
    }


def test_partial_d_interrupted_reservation_linkage_is_retained():
    current, _, _, current_rows, _, _ = _terminal_fixture()
    owners = _budget_owner_mapping(current)
    group_id, linked = next(
        (run_id, values)
        for run_id, values in owners.items()
        if "--experiment-d--" in run_id
    )
    specs = {spec.run_id: spec for spec in current.expand(stage="experiment-d")}
    focal = specs[linked[0]]
    budget_failure = {
        "error_type": "InterruptedReservation",
        "message": (
            "a prior process created shared Experiment D budget state without an "
            "exact terminal ITT group; no redispatch is permitted"
        ),
        "model_id": focal.model_id,
        "environment_seed": focal.environment_seed,
        "provider_dispatch_started": False,
        "stop_origin": "pre-catalog-interrupted-reservation-recovery",
        "accounting_basis": "unreconciled-conservative-reservation",
    }
    raw_rows = _SnapshotLedger(current_rows).snapshot()["runs"]
    raw_rows[linked[0]]["status"] = "failed"
    raw_rows[linked[0]]["failure"] = {
        "error_type": "CrashFixture",
        "message": "one original pre-reservation terminal",
    }
    raw_rows[linked[0]]["artifact"] = None
    for run_id in linked[1:]:
        raw_rows[run_id]["status"] = "integrity-stopped"
        raw_rows[run_id]["failure"] = deepcopy(budget_failure)
        raw_rows[run_id]["artifact"] = None
    linkage = _observed_owner_linkage(
        current,
        budget_id=group_id,
        budget_row={"status": "integrity-stopped", "failure": budget_failure},
        linked_ids=linked,
        run_rows=raw_rows,
    )
    assert linkage["classification"] == "interrupted-reservation-recovery"
    assert linkage["direct_terminal_count"] == 10
    assert linkage["pre_reservation_original_terminal_count"] == 1

    tampered_failure = {**budget_failure, "accounting_basis": "changed"}
    with pytest.raises(PilotEvidenceError, match="terminal linkage drifted"):
        _observed_owner_linkage(
            current,
            budget_id=group_id,
            budget_row={
                "status": "integrity-stopped",
                "failure": tampered_failure,
            },
            linked_ids=linked,
            run_rows=raw_rows,
        )


def test_single_owner_after_itt_recovery_keeps_original_terminal():
    current, _, _, current_rows, _, _ = _terminal_fixture()
    owners = _budget_owner_mapping(current)
    budget_id, linked = next(
        (run_id, values)
        for run_id, values in owners.items()
        if "--experiment-b--" in run_id
    )
    rows = _SnapshotLedger(current_rows).snapshot()["runs"]
    original = rows[linked[0]]
    failure = {
        "error_type": "InterruptedReservationAfterITT",
        "message": (
            "a terminal ITT row retained an unreconciled reservation; the "
            "conservative reservation was charged before stopping"
        ),
        "accounting_basis": "unreconciled-conservative-reservation",
    }
    linkage = _observed_owner_linkage(
        current,
        budget_id=budget_id,
        budget_row={"status": "integrity-stopped", "failure": failure},
        linked_ids=linked,
        run_rows=rows,
    )
    assert original["status"] == "complete"
    assert linkage["classification"] == ("original-terminal-after-budget-recovery")
    assert linkage["original_terminal_after_budget_recovery_count"] == 1


def test_current_source_root_symlink_alias_is_rejected(tmp_path):
    real = tmp_path / "real"
    (real / "experiments").mkdir(parents=True)
    raw = real / "experiment_results/pilot-v2.11.9/raw"
    raw.mkdir(parents=True)
    (real / "experiments/pilot_v2_11_9.yaml").write_text("fixture\n")
    (raw / "run_ledger.json").write_text("{}\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(PilotEvidenceError, match="symlink"):
        _resolve_current_paths(
            source_repo_root=alias,
            contract_path=alias / "experiments/pilot_v2_11_9.yaml",
            raw_root=alias / "experiment_results/pilot-v2.11.9/raw",
            run_ledger_path=(
                alias / "experiment_results/pilot-v2.11.9/raw/run_ledger.json"
            ),
        )


def test_package_target_rejects_raw_but_allows_current_evidence(tmp_path):
    current_root = tmp_path / "current"
    current_raw = current_root / "experiment_results/pilot-v2.11.9/raw"
    authority = tmp_path / "authority"
    parent_raw = authority / "experiment_results/pilot-v2.11.5/raw"
    current_raw.mkdir(parents=True)
    parent_raw.mkdir(parents=True)
    allowed = _package_target(
        current_root / "evidence",
        source_roots=(current_raw, authority, parent_raw),
    )
    assert allowed == current_root / "evidence/current_v2/pilot-v2.11.9"
    with pytest.raises(PilotEvidenceError, match="immutable source/raw root"):
        _package_target(
            current_raw,
            source_roots=(current_raw, authority, parent_raw),
        )


def test_reviewer_deliverable_shape_is_provider_free(tmp_path, monkeypatch):
    def forbidden_provider(*args, **kwargs):  # pragma: no cover
        raise AssertionError("package writing constructed a provider")

    monkeypatch.setattr(pilot_orchestrator, "_provider_for_profile", forbidden_provider)
    current, parent, _, _, _, _ = _terminal_fixture()
    aggregate = _assemble(_terminal_fixture())
    target = tmp_path / "package"
    manifest_path, checksums_path = _write_package(
        target,
        aggregate=aggregate,
        controls={"pass": True, "provider_calls": 0},
        contract=current,
        contract_source=ROOT / "experiments/pilot_v2_11_9.yaml",
        authority_contract=parent,
        authority_contract_source=ROOT / "experiments/pilot_v2_11_5.yaml",
    )
    required = {
        "aggregate.csv",
        "aggregate.json",
        "claim_metric_artifact.json",
        "contract/pilot_v2_11_9.yaml",
        "contract/pilot_v2_11_5.yaml",
        "contract/pilot_v2_11_5_source_manifest.json",
        "contract/pilot_v2_11_9_source_manifest.json",
        "failure_ledger.json",
        "method_differences_scaffold.json",
        "model_capability_failures.json",
        "narrative_results.json",
        "release_controls.json",
        "reviewer_report.md",
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    assert required == set(manifest["published_files"])
    assert required <= {row["path"] for row in checksums["files"]}
    claims = json.loads(
        (target / "claim_metric_artifact.json").read_text(encoding="utf-8")
    )
    assert len(claims["claims"]) == 6
    failure_ledger = json.loads(
        (target / "failure_ledger.json").read_text(encoding="utf-8")
    )
    assert set(failure_ledger) == {
        "schema_version",
        "contract_sha256",
        "denominator",
        "rows",
    }
    # Metric-long CSV has more than one line per logical run when metrics exist.
    assert len((target / "aggregate.csv").read_text().splitlines()) > 137
    verified = _verify_package_tree(
        target,
        contract=current,
        authority_contract=parent,
    )
    assert verified["file_count"] == len(required) + 2
    (target / "aggregate.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(PilotEvidenceError, match="checksum mismatch"):
        _verify_package_tree(
            target,
            contract=current,
            authority_contract=parent,
        )


def test_current_git_rejects_untracked_source(tmp_path):
    current, _ = _contracts()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    (tmp_path / "tracked.txt").write_text("sealed\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-qm", "fixture")
    git("tag", "-a", "pilot-v2.11.9-science", "-m", "fixture tag")
    commit = git("rev-parse", "HEAD")
    release = {
        "local_tag": {
            "name": "pilot-v2.11.9-science",
            "object_id": git("rev-parse", "refs/tags/pilot-v2.11.9-science"),
            "peeled_commit": commit,
            "kind": "annotated",
        }
    }
    assert _validate_current_git(
        tmp_path.resolve(), current, commit, release_attestation=release
    )["tracked_worktree_clean"]

    tampered = deepcopy(release)
    tampered["local_tag"]["object_id"] = "0" * 40
    with pytest.raises(PilotEvidenceError, match="tracked-clean annotated tag"):
        _validate_current_git(
            tmp_path.resolve(), current, commit, release_attestation=tampered
        )

    (tmp_path / "untracked_source.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(PilotEvidenceError, match="tracked-clean annotated tag"):
        _validate_current_git(
            tmp_path.resolve(), current, commit, release_attestation=release
        )
