from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import pytest

from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory import pilot_orchestrator as orchestrator
import verified_memory.pilot_v21111_fresh_cohort as cohort
import verified_memory.pilot_v21111_release as release
from verified_memory.pilot_v2115_parent_import import (
    V2115_CALIBRATION_WRAPPER_SCHEMA_VERSION,
    V2115_CAPABILITY_WRAPPER_SCHEMA_VERSION,
    V2115_PREFLIGHT_WRAPPER_SCHEMA_VERSION,
    _seal as seal_v2115,
)


REPO = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO / "experiments" / "pilot_v2_11_11.yaml"
TEST_RUNTIME_PATH_SET_SHA256 = "6" * 64
TEST_RELEASE_LINEAGE = {"fixture": "sealed historical release lineage"}


def _lineage(contract):
    source = contract.v21111_fresh_cohort_boundary["v2115_scientific_authority"]
    return {
        "child_release": {
            "contract_id": source["contract_id"],
            "contract_sha256": source["contract_sha256"],
            "git_tag": source["science_tag"],
            "resolved_git_commit": source["science_commit"],
        },
        "parent_release": {
            "contract_id": "finevo-pilot-v2.11.2",
            "contract_sha256": "1" * 64,
            "git_tag": "pilot-v2.11.2-science",
            "git_tag_object": "2" * 40,
            "resolved_git_commit": "3" * 40,
        },
        "source_manifest": {
            "path": "experiments/pilot_v2_11_5_source_manifest.json",
            "file_sha256": "4" * 64,
            "content_sha256": "5" * 64,
        },
    }


def _wrapper_bundle(contract):
    lineage = _lineage(contract)
    calibration = seal_v2115(
        {
            "schema_version": V2115_CALIBRATION_WRAPPER_SCHEMA_VERSION,
            **deepcopy(lineage),
            "source_wrapper": {
                "schema_version": "source-calibration-v1",
                "content_sha256": "6" * 64,
            },
            "calibration": {},
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
            "hosted_provider_calls_during_import": 0,
            "hosted_cost_usd_during_import": 0.0,
            "imported_effect_cells": 0,
            "imported_scientific_run_summaries": 0,
            "imported_scientific_outcome_artifacts": [],
            "decoded_completion_reuse": False,
            "scientific_evidence": False,
            "evidence_use": "outcome-blind authority fixture",
        }
    )
    capabilities = {}
    preflights = {}
    expected_categories = {
        "utility-ranking": (12, 10),
        "rule-application": (12, 10),
        "rule-proposal": (6, 5),
    }
    for model_id in cohort.V21111_AUTHORITY_MODELS:
        profile = contract.provider_profiles[model_id]
        checks = {category: True for category in expected_categories}
        category_totals = {
            category: {
                "denominator": denominator,
                "registered_total": denominator,
                "registered_correct": denominator,
                "required": required,
                "interface_failure_count": 0,
            }
            for category, (denominator, required) in expected_categories.items()
        }
        capability = {
            "model_id": model_id,
            "requested_model": profile.requested_model,
            "served_model": profile.served_model,
            "runtime_model": f"{profile.transport}/{profile.requested_model}",
            "action_sample_count": 24,
            "semantic_sample_count": 6,
            "historical_source_calls": 30,
            "parse_failure_count": 0,
            "provider_failure_count": 0,
            "truncation_count": 0,
            "capability_pass": True,
            "interface_pass": True,
            "checks": checks,
            "capability_assessment": {
                "pass": True,
                "status": "pass",
                "checks": checks,
            },
            "interface_gate": {"pass": True, "failure_count": 0},
            "category_totals": category_totals,
            "taskset_sha256": "7" * 64,
            "stage_receipt_content_sha256": "8" * 64,
        }
        capabilities[model_id] = seal_v2115(
            {
                "schema_version": V2115_CAPABILITY_WRAPPER_SCHEMA_VERSION,
                **deepcopy(lineage),
                "source_wrapper": {
                    "schema_version": "source-capability-v1",
                    "content_sha256": "9" * 64,
                },
                "capability": capability,
                "provider_construction_current_attempt": False,
                "provider_calls_current_attempt": 0,
                "hosted_provider_calls_current_attempt": 0,
                "current_attempt_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                },
                "imported_effect_cells": 0,
                "imported_preflight_samples": 0,
                "scientific_evidence": False,
                "evidence_scope": "preregistered_task_capability_gate",
            }
        )
        reservations = {
            "action": {"reservation": {"sample_count": 24}},
            "semantic": {"reservation": {"sample_count": 8}},
        }
        preflights[model_id] = seal_v2115(
            {
                "schema_version": V2115_PREFLIGHT_WRAPPER_SCHEMA_VERSION,
                **deepcopy(lineage),
                "model_id": model_id,
                "runtime_model": f"{profile.transport}/{profile.requested_model}",
                "source_gate_receipt": {
                    "path": "long-context-preflight/post_gate.json",
                    "file_sha256": "a" * 64,
                    "content_sha256": "b" * 64,
                    "git_commit": "c" * 40,
                },
                "reservations": reservations,
                "source_reservation_sha256": canonical_sha256(reservations),
                "sample_counts": {"action": 24, "semantic": 8},
                "provider_construction_current_attempt": False,
                "provider_calls_current_attempt": 0,
                "hosted_provider_calls_current_attempt": 0,
                "historical_provider_calls": 32,
                "historical_calls_already_in_parent_debit": True,
                "imported_effect_cells": 0,
                "scientific_evidence": False,
            }
        )
    return calibration, capabilities, preflights


def _parent_receipt(contract):
    calibration, capabilities, preflights = _wrapper_bundle(contract)
    boundary = contract.v21111_fresh_cohort_boundary
    v2115 = boundary["v2115_scientific_authority"]
    v21110 = boundary["v21110_terminal_release"]
    denominator = cohort._validate_imported_authority_wrappers(
        contract,
        calibration=calibration,
        capabilities=capabilities,
        preflights=preflights,
    )
    source_manifest = cohort._json_copy(boundary["source_manifest"])
    source_replay = release._seal(
        {
            "schema_version": release.V21111_SOURCE_REPLAY_SCHEMA_VERSION,
            "contract_id": contract.contract_id,
            "performed": True,
            "recomputed_equal": True,
            "source_root_roles_pairwise_distinct": True,
            "source_manifest": source_manifest,
            "runtime_source_path_set_sha256": TEST_RUNTIME_PATH_SET_SHA256,
            "release_lineage_sha256": canonical_sha256(TEST_RELEASE_LINEAGE),
            "provider_construction": False,
            "provider_calls": 0,
            "scientific_evidence": False,
        }
    )
    return cohort._seal(
        {
            "schema_version": cohort.V21111_PARENT_IMPORT_SCHEMA,
            "status": "go",
            "go": True,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "v2115_authority": {
                "tag": v2115["science_tag"],
                "tag_object": v2115["science_tag_object"],
                "commit": v2115["science_commit"],
                "clean": True,
            },
            "v2115_parent_receipt": {
                "path": cohort.V21111_V2115_PARENT_RECEIPT_PATH,
                "file_sha256": "d" * 64,
                "content_sha256": "e" * 64,
            },
            "calibration_wrapper": calibration,
            "capability_wrappers": capabilities,
            "preflight_authority_wrappers": preflights,
            "authority_wrapper_denominator": denominator,
            "v21110_terminal": {
                "tag": v21110["science_tag"],
                "tag_object": v21110["science_tag_object"],
                "commit": v21110["science_commit"],
                "clean": True,
            },
            "source_manifest": source_manifest,
            "source_manifest_replay": source_replay,
            "parent_budget_debit": cohort.parent_budget_debit_for_v21111(
                contract
            ).to_dict(),
            "evidence_partition": cohort.stage_partition_from_contract(contract),
            "provider_boundary": {
                "provider_construction": False,
                "provider_calls": 0,
                "hosted_cost_usd": 0.0,
            },
            "scientific_evidence": False,
            "claim_boundary": cohort.V21111_PARENT_CLAIM_BOUNDARY,
        }
    )


def _write_receipt(raw: Path, receipt) -> Path:
    path = raw / "parent-import" / cohort.V21111_PARENT_RECEIPT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _source_bound_contract(repo: Path):
    contract = load_pilot_contract(CONTRACT_PATH)
    manifest = release._seal(
        {
            "schema_version": release.V21111_SOURCE_MANIFEST_SCHEMA_VERSION,
            "contract_id": contract.contract_id,
            "current_runtime_sources": {
                "release_python_source_path_set_sha256": (TEST_RUNTIME_PATH_SET_SHA256)
            },
            "release_lineage": TEST_RELEASE_LINEAGE,
        }
    )
    manifest_path = repo.joinpath(*release.V21111_SOURCE_MANIFEST_PATH.parts)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    boundary = cohort._json_copy(contract.v21111_fresh_cohort_boundary)
    boundary["source_manifest"] = {
        "path": release.V21111_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": release.V21111_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": cohort._file_sha256(manifest_path),
        "content_sha256": manifest["integrity"]["content_sha256"],
    }
    return (
        replace(contract, v21111_fresh_cohort_boundary=boundary),
        repo / "experiment_results/pilot-v2.11.11/raw",
    )


def _reseal_parent(receipt):
    value = deepcopy(receipt)
    value.pop("integrity", None)
    return cohort._seal(value)


def _reseal_replay_field(receipt, field: str, value) -> None:
    replay = deepcopy(receipt["source_manifest_replay"])
    replay.pop("integrity", None)
    replay[field] = value
    receipt["source_manifest_replay"] = release._seal(replay)


def _copy_contract(repo: Path) -> Path:
    target = repo / "experiments" / CONTRACT_PATH.name
    target.parent.mkdir(parents=True)
    shutil.copyfile(CONTRACT_PATH, target)
    return target


def _parent_complete_ledgers(contract, raw: Path) -> None:
    run = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run.register(contract.expand())
    (parent,) = contract.expand(stage="parent-import")
    run.finalize(parent.run_id, status="complete", artifact=None)
    budget = orchestrator.PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    )
    projection = orchestrator._v21111_parent_import_projection(parent)
    budget.reserve(projection)
    budget.finalize(
        parent.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=0,
    )


def test_child_parent_receipt_exposes_exact_two_model_capability_authority(
    tmp_path: Path,
) -> None:
    contract, raw = _source_bound_contract(tmp_path / "repo")
    receipt = _parent_receipt(contract)
    path = _write_receipt(raw, receipt)

    verified = cohort.verify_parent_import_receipt(path, contract=contract)
    assert verified["authority_wrapper_denominator"] == {
        "model_ids": ["gpt52_main", "gpt56_diagnostic"],
        "capability_wrapper_count": 2,
        "preflight_wrapper_count": 2,
        "calibration_wrapper_content_sha256": verified["calibration_wrapper"][
            "integrity"
        ]["content_sha256"],
        "capability_wrapper_content_sha256": {
            model_id: verified["capability_wrappers"][model_id]["integrity"][
                "content_sha256"
            ]
            for model_id in cohort.V21111_AUTHORITY_MODELS
        },
        "preflight_wrapper_content_sha256": {
            model_id: verified["preflight_authority_wrappers"][model_id]["integrity"][
                "content_sha256"
            ]
            for model_id in cohort.V21111_AUTHORITY_MODELS
        },
    }
    wrapper = cohort.verified_capability_wrapper_for_v21111(
        contract,
        "gpt56_diagnostic",
        raw_root=raw,
    )
    assert wrapper["capability"]["model_id"] == "gpt56_diagnostic"
    assert wrapper["capability"]["capability_pass"] is True


@pytest.mark.parametrize(
    "tamper, message",
    [
        (
            lambda value: value.update({"unexpected": True}),
            "top-level shape",
        ),
        (
            lambda value: value["v2115_parent_receipt"].update(
                {"path": "elsewhere/parent_import_receipt.json"}
            ),
            "receipt drifted",
        ),
        (
            lambda value: value["authority_wrapper_denominator"].update(
                {"capability_wrapper_count": 1}
            ),
            "receipt drifted",
        ),
        (
            lambda value: value["source_manifest_replay"].update({"provider_calls": 1}),
            "replay integrity drifted",
        ),
        (
            lambda value: _reseal_replay_field(
                value,
                "runtime_source_path_set_sha256",
                "8" * 64,
            ),
            "semantic binding drifted",
        ),
        (
            lambda value: _reseal_replay_field(
                value,
                "release_lineage_sha256",
                "9" * 64,
            ),
            "semantic binding drifted",
        ),
    ],
)
def test_child_parent_receipt_rejects_resealed_shape_path_and_denominator_tamper(
    tmp_path: Path,
    tamper,
    message: str,
) -> None:
    contract, raw = _source_bound_contract(tmp_path / "repo")
    receipt = _parent_receipt(contract)
    tamper(receipt)
    path = _write_receipt(raw, _reseal_parent(receipt))
    with pytest.raises(cohort.PilotV21111FreshCohortError, match=message):
        cohort.verify_parent_import_receipt(path, contract=contract)


def test_child_parent_receipt_rejects_resealed_capability_model_swap(
    tmp_path: Path,
) -> None:
    contract, raw = _source_bound_contract(tmp_path / "repo")
    receipt = _parent_receipt(contract)
    capabilities = receipt["capability_wrappers"]
    capabilities["gpt52_main"], capabilities["gpt56_diagnostic"] = (
        capabilities["gpt56_diagnostic"],
        capabilities["gpt52_main"],
    )
    receipt["authority_wrapper_denominator"]["capability_wrapper_content_sha256"] = {
        model_id: capabilities[model_id]["integrity"]["content_sha256"]
        for model_id in cohort.V21111_AUTHORITY_MODELS
    }
    path = _write_receipt(raw, _reseal_parent(receipt))
    with pytest.raises(
        cohort.PilotV21111FreshCohortError,
        match="capability authority drifted",
    ):
        cohort.verify_parent_import_receipt(path, contract=contract)


def test_child_parent_receipt_rejects_symlink_path(
    tmp_path: Path,
) -> None:
    contract, raw = _source_bound_contract(tmp_path / "repo")
    target = tmp_path / "repo" / "source.json"
    target.write_text(
        json.dumps(_parent_receipt(contract), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path = raw / "parent-import" / cohort.V21111_PARENT_RECEIPT_FILENAME
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    with pytest.raises(cohort.PilotV21111FreshCohortError, match="path is not exact"):
        cohort.verify_parent_import_receipt(path, contract=contract)


def test_parent_cost_uses_exact_1e8_quantum_and_rejects_one_quantum_tamper() -> None:
    actuals = [
        (1.0554670000000002, 48, 831898),
        (1.7282860000000004, 76, 1065105),
        (4.048673999999999, 204, 2742160),
        (0.950432, 40, 923804),
        (4.439952999999998, 224, 2837381),
        (2.9812842499999967, 160, 1410961),
        (0.0, 0, 140446),
    ]
    rows = {
        str(index): {
            "actual": {
                "cost_usd": cost,
                "completions": completions,
                "storage_bytes": storage,
            }
        }
        for index, (cost, completions, storage) in enumerate(actuals)
    }
    expected = {
        "cost_usd": 15.20409625,
        "hosted_completions": 752,
        "storage_bytes": 9951755,
    }
    verified = cohort._verified_parent_current_actual(rows, expected)
    assert str(verified["cost_usd"]) == "15.20409625"

    tampered = deepcopy(rows)
    tampered["0"]["actual"]["cost_usd"] += 0.00000001
    with pytest.raises(
        cohort.PilotV21111FreshCohortError,
        match="current actual debit drifted",
    ):
        cohort._verified_parent_current_actual(tampered, expected)


def test_real_v21110_v2115_parent_fixture_round_trips_when_available() -> None:
    terminal = Path(
        os.environ.get(
            "FINEVO_V21110_SCIENCE_RELEASE_ROOT",
            REPO.parent / "finevo-pilot-v2-11-10-science",
        )
    )
    authority = Path(
        os.environ.get(
            "FINEVO_V2115_SCIENCE_RELEASE_ROOT",
            REPO.parent / "finevo-pilot-v2-11-5-science",
        )
    )
    if (
        not (
            terminal / "experiment_results/pilot-v2.11.10/raw/budget_ledger.json"
        ).is_file()
        or not (
            authority
            / "experiment_results/pilot-v2.11.5/raw/parent-import/parent_import_receipt.json"
        ).is_file()
    ):
        pytest.skip("exact ignored V2.11.10/V2.11.5 science fixtures are absent")
    contract = load_pilot_contract(CONTRACT_PATH)
    receipt = cohort.verify_parent_sources(
        contract,
        repo_root=REPO,
        v21110_repo_root=terminal,
        v2115_repo_root=authority,
    )
    assert receipt["authority_wrapper_denominator"]["model_ids"] == [
        "gpt52_main",
        "gpt56_diagnostic",
    ]
    assert receipt["authority_wrapper_denominator"][
        "capability_wrapper_content_sha256"
    ] == {
        "gpt52_main": (
            "be8684bd1208bb5049be744910c10bdaf5f48e69ad6f13ae086ecef9ce42e32f"
        ),
        "gpt56_diagnostic": (
            "f3a3025347327545e33d42149efe1ed0d29c3279429b3f379ecc88c6cdeab863"
        ),
    }
    assert receipt["parent_budget_debit"]["cost_usd"] == 78.3237413125
    assert receipt["provider_boundary"]["provider_calls"] == 0


@pytest.mark.parametrize(
    ("stage_id", "expected_model_id"),
    [
        pytest.param("experiment-b", "gpt52_main", id="experiment-b"),
        pytest.param("cross-model", "gpt56_diagnostic", id="cross-model"),
    ],
)
def test_science_rejects_child_capability_drift_before_projection_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
    expected_model_id: str,
) -> None:
    contract = replace(load_pilot_contract(CONTRACT_PATH), status="frozen")
    repo = tmp_path / f"repo-{stage_id}"
    contract_path = _copy_contract(repo)
    raw = repo / cohort.V21111_RAW_NAMESPACE
    raw.mkdir(parents=True)
    _parent_complete_ledgers(contract, raw)
    paid = orchestrator.GitProvenance(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v21111_dispatch_refresh_go",
        lambda **_kwargs: {"status": "go", "go": True},
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_prerequisites",
        lambda *_args, **_kwargs: {},
    )

    def reject_capability(_contract, model_id: str, *, raw_root: Path):
        assert raw_root == raw.resolve()
        calls.append(("capability", model_id))
        raise cohort.PilotV21111FreshCohortError(
            "fixture child capability wrapper drift"
        )

    monkeypatch.setattr(
        orchestrator,
        "verified_capability_wrapper_for_v21111",
        reject_capability,
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("projection, provider, or execution was reached after drift")

    for name in (
        "validate_live_provider_catalog",
        "_remaining_core_projections",
        "projection_from_preflight",
        "_provider_for_profile",
        "create_llm_provider",
        "_execute_actor_run",
    ):
        monkeypatch.setattr(orchestrator, name, forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="imported capability authority failed.*wrapper drift",
    ):
        orchestrator._execute_stage_locked(
            contract_path=contract_path,
            stage_id=stage_id,
            resume=False,
            raw_root=raw,
            repo_root=repo,
        )

    assert calls == [("capability", expected_model_id)]
    ledger = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    assert {ledger.status(spec.run_id) for spec in contract.expand(stage=stage_id)} == {
        "scheduled"
    }
