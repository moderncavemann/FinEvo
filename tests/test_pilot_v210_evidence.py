from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import pilot_v24_evidence as evidence
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidenceError


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10_overlay.yaml"
PREREQUISITE_STAGES = {
    "parent-import",
    "q-ref-resolution",
    "stage0-calibration",
}


def _rows(contract, *, failed_effects: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in contract.expand():
        prerequisite = spec.stage_id in PREREQUISITE_STAGES
        status = "failed" if failed_effects and not prerequisite else "complete"
        rows.append(
            {
                **spec.to_dict(),
                "status": status,
                "failure": (
                    None
                    if status == "complete"
                    else {"kind": "fixture-terminal-provider-failure"}
                ),
                "artifact_kind": (
                    (
                        "terminal-summary"
                        if prerequisite
                        else "verified-run-manifest"
                    )
                    if status == "complete"
                    else None
                ),
                "artifact_sha256": (
                    "a" * 64 if status == "complete" else None
                ),
                "scientific_eligible": bool(
                    status == "complete"
                    and (
                        spec.stage_id == "stage0-calibration"
                        if prerequisite
                        else True
                    )
                ),
                "metrics": {},
                "gate_evidence": {},
                "capability": {},
                "narrative": {},
            }
        )
    return rows


def _denominator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "expected_count": 211,
        "observed_ledger_count": 211,
        "all_rows_present": True,
        "all_rows_terminal": True,
        "status_counts": dict(sorted(statuses.items())),
        "all_completed_artifacts_validated": True,
        "pass": True,
    }


def _release_controls(
    *,
    sensitivity_pass: bool = True,
) -> dict[str, Any]:
    sensitivity_controls = {}
    for lane_id in evidence._V210_C_SENSITIVITY_FILES:
        definition = evidence._v210_sensitivity_lane_definition(lane_id)
        sensitivity_controls[lane_id] = {
            **definition,
            "pass": sensitivity_pass,
            "available": sensitivity_pass,
            "provider_calls": 0,
            "descriptive_only": True,
            "effectiveness_gate": False,
            **(
                {
                    "path": f"/fixture/{definition['stage_id']}/rule_sensitivity.json",
                    "file_sha256": "b" * 64,
                    "content_sha256": "c" * 64,
                    "source_run_count": 5,
                    "grid_cell_count": 9,
                }
                if sensitivity_pass
                else {"reason": "fixture lane is incomplete"}
            ),
        }
    return {
        "pass": True,
        "experiment_c_rule_sensitivities": sensitivity_controls,
        "budget_ledger": {
            "pass": True,
            "raw_root_storage_bytes": 0,
            "checks": {"parent_debit_exact": True},
            "actual_totals": {
                "cost_usd": 3.212770875,
                "completions": 184,
                "storage_bytes": 50_425_235,
            },
            "actual_stage_cost_usd": {
                "parent_v23": 3.212770875,
            },
        },
    }


def _install_supported_gate_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> list[set[str]]:
    observed_stage_sets: list[set[str]] = []

    def record(rows) -> None:
        observed_stage_sets.append({str(row["stage_id"]) for row in rows})

    def c_gate(_contract, rows, *, stage_id, model_id):
        record(rows)
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "same_direction_counts": {"false_activation": 5},
            "claim_action": f"retain {model_id}/{stage_id}",
        }

    def a_gate(_contract, rows, *, stage_id, model_id):
        record(rows)
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "primary_contrast": {
                "raw_paired_deltas": {
                    str(seed): 1.0
                    for seed in _contract.seeds["sets"]["main"]
                }
            },
            "threshold_gate": {"same_direction_count": 5},
            "claim_action": f"retain {model_id}/{stage_id}",
        }

    def d_gate(_contract, rows, *, stage_id, model_id, arms):
        record(rows)
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "supported_treatments": ["no-memory"],
            "treatment_gates": {
                "no-memory": {
                    "six_step_discounted_utility_gate": {
                        "treatment_deltas": {
                            str(seed): 1.0
                            for seed in _contract.seeds["sets"]["main"]
                        }
                    }
                }
            },
            "claim_action": f"retain {model_id}/{stage_id}/{tuple(arms)!r}",
        }

    def b_summary(rows, *, stage_id, model_id, arms):
        record(rows)
        return {
            "comparison_type": "descriptive_preregistered_architecture_arms",
            "selection_rule": "do not select a winner",
            "arms": {arm: {} for arm in arms},
            "binding": f"{model_id}/{stage_id}",
        }

    monkeypatch.setattr(evidence, "_experiment_c_gate", c_gate)
    monkeypatch.setattr(evidence, "_experiment_a_gate", a_gate)
    monkeypatch.setattr(evidence, "_experiment_d_gate", d_gate)
    monkeypatch.setattr(evidence, "_experiment_b_summary", b_summary)
    return observed_stage_sets


def _sensitivity_payload(
    contract,
    rows: list[dict[str, Any]],
    *,
    lane_id: str,
    commit: str,
) -> dict[str, Any]:
    definition = evidence._v210_sensitivity_lane_definition(lane_id)
    stage_id = definition["stage_id"]
    model_id = definition["model_id"]
    source_rows = sorted(
        (
            row
            for row in rows
            if row["stage_id"] == stage_id
            and row["model_id"] == model_id
            and row["arm_id"] == "full"
        ),
        key=lambda row: row["run_id"],
    )
    sensitivity_contract = contract.stop_go["experiment_c"][
        "zero_api_sensitivity"
    ]
    cells = [
        {
            "alternative_success_weight": weight,
            "outcome_definition": outcome,
            "source_run_count": 5,
            "natural_rule_count": 0,
            "ever_active_count": 0,
            "retired_count": 0,
            "active_exposure_steps": 0,
        }
        for weight in sensitivity_contract["alternative_success_weights"]
        for outcome in sensitivity_contract["outcome_definitions"]
    ]
    return {
        "schema_version": "finevo-experiment-c-sensitivity-v1",
        "status": "pass",
        "terminal": True,
        "control_kind": "zero-api-offline-rule-sensitivity",
        "provider_calls": 0,
        "descriptive_only": True,
        "effectiveness_gate": False,
        "scientific_evidence": True,
        "absolute_flow_utility_threshold": {"value": 0.0},
        "verifier_config": {},
        "alternative_success_weights": list(
            sensitivity_contract["alternative_success_weights"]
        ),
        "outcome_definitions": list(
            sensitivity_contract["outcome_definitions"]
        ),
        "source_run_count": len(source_rows),
        "per_run": [],
        "aggregate_cells": cells,
        "bindings": {
            "contract_sha256": contract.canonical_hash,
            "git_tag": contract.implementation["required_git_tag"],
            "git_commit": commit,
            "stage0_selection": "stage0-calibration/stage0_selection.json",
            "stage0_selection_content_sha256": "d" * 64,
            "stage0_selection_file_sha256": "e" * 64,
            "source_stage": stage_id,
            "source_arm": "full",
            "source_manifests": [
                {
                    "run_id": row["run_id"],
                    "environment_seed": row["environment_seed"],
                    "manifest": f"{stage_id}/runs/{row['run_id']}/manifest.json",
                    "manifest_sha256": row["artifact_sha256"],
                }
                for row in source_rows
            ],
            "source_matrix_sha256": "f" * 64,
        },
        "claim_boundary": "fixture descriptive sensitivity only",
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
            "content_sha256": "c" * 64,
        },
    }


def _write_sensitivity_fixtures(
    source_root: Path,
    contract,
    rows: list[dict[str, Any]],
    *,
    commit: str,
) -> Path:
    raw = source_root / "raw"
    for lane_id in evidence._V210_C_SENSITIVITY_FILES:
        definition = evidence._v210_sensitivity_lane_definition(lane_id)
        path = raw / definition["stage_id"] / "rule_sensitivity.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                _sensitivity_payload(
                    contract,
                    rows,
                    lane_id=lane_id,
                    commit=commit,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return raw


def _install_sensitivity_loader_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_root: Path,
) -> list[tuple[str, str]]:
    observed: list[tuple[str, str]] = []

    def loader(
        _contract,
        *,
        raw_root,
        paid,
        stage_id,
        model_id,
        authority_repo_root,
    ):
        assert paid is None
        assert authority_repo_root == source_root.resolve()
        assert Path.cwd() == source_root.resolve()
        observed.append((stage_id, model_id))
        return json.loads(
            (raw_root / stage_id / "rule_sensitivity.json").read_text(
                encoding="utf-8"
            )
        )

    monkeypatch.setattr(
        orchestrator,
        "_load_verified_experiment_c_sensitivity",
        loader,
    )
    return observed


def test_v210_complete_fixture_excludes_16_and_uses_195_fresh_a_d_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    observed_stage_sets = _install_supported_gate_fixtures(monkeypatch)

    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
    )

    assert aggregate["schema_version"] == (
        "finevo-pilot-v2.10-evidence-package-v1"
    )
    assert aggregate["evidence_namespace"] == "current_v2/pilot-v2.10"
    assert aggregate["publication_status"] == "complete"
    assert aggregate["scientific_complete"] is True
    assert aggregate["itt_row_preservation"] == {
        "registered_rows": 211,
        "retained_rows": 211,
        "prerequisite_rows": 16,
        "fresh_a_d_rows": 195,
        "imported_a_d_rows": 0,
        "failed_or_stopped_rows": 0,
        "status_counts": {"complete": 211},
        "all_registered_rows_retained": True,
        "failures_retained": True,
    }
    prerequisites = aggregate["prerequisites"]
    assert prerequisites["registered_cells"] == 16
    assert prerequisites["all_prerequisites_complete"] is True
    assert prerequisites["fresh_a_d_cells_required"] == 195
    assert prerequisites["imported_a_d_effect_cells"] == 0
    assert prerequisites["import_provider_accounting"]["provider_calls"] == 0
    assert (
        aggregate["effect_aggregation_scope"][
            "fresh_v2_10_a_d_cells_only"
        ]
        is True
    )
    assert set(
        aggregate["effect_aggregation_scope"][
            "prerequisite_stage_ids_excluded"
        ]
    ) == PREREQUISITE_STAGES
    assert observed_stage_sets
    assert all(
        not (stage_ids & PREREQUISITE_STAGES)
        for stage_ids in observed_stage_sets
    )
    lineage = aggregate["parent_evidence_lineage"]
    assert lineage["source_contract_id"] == "finevo-pilot-v2.9"
    assert lineage["parent_status_counts"] == {"complete": 26, "failed": 185}
    assert lineage["parent_offline_candidate_admission_cells_generated"] == 10
    assert lineage["parent_actor_treatment_effect_outcomes_generated"] is False
    assert lineage["parent_rows_imported_into_v2_10_effect_aggregate"] == 0
    budget = aggregate["inherited_budget_boundary"]
    assert budget["total_cap_usd"] == 500.0
    assert budget["expected_cumulative_prior"]["cost_usd"] == 3.212770875
    assert budget["expected_cumulative_prior"]["hosted_completions"] == 184
    assert budget["expected_cumulative_prior"]["storage_bytes"] == 50_425_235
    assert budget["automatically_dispatchable_usd_before_v2_10"] == pytest.approx(
        495.787229125
    )


def test_v210_terminal_failures_are_retained_as_complete_with_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract, failed_effects=True)
    _install_supported_gate_fixtures(monkeypatch)

    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(sensitivity_pass=False),
    )

    assert aggregate["denominator"]["status_counts"] == {
        "complete": 16,
        "failed": 195,
    }
    assert aggregate["itt_row_preservation"]["failed_or_stopped_rows"] == 195
    assert aggregate["itt_row_preservation"]["fresh_a_d_rows"] == 195
    assert aggregate["scientific_matrix_complete"] is False
    assert aggregate["scientific_complete"] is False
    assert aggregate["publication_status"] == "complete-with-no-go"
    evidence._require_publishable_terminal_denominator(aggregate)


def test_v210_parent_reference_revalidates_v29_without_rewriting() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    parent_root = ROOT / "evidence" / "current_v2" / "pilot-v2.9"
    before = {
        path.relative_to(parent_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in parent_root.rglob("*")
        if path.is_file()
    }

    reference = evidence._validated_v210_parent_evidence_reference(
        contract,
        contract_path=CONTRACT_PATH,
    )

    after = {
        path.relative_to(parent_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in parent_root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert reference is not None
    assert reference["source_package_path"] == "evidence/current_v2/pilot-v2.9"
    assert reference["source_package_copied"] is False
    assert reference["implementation_no_go_verified"] is True
    assert reference["offline_candidate_disclosure_verified"] is True
    assert reference["parent_status_counts"] == {"complete": 26, "failed": 185}


def test_v210_lineage_and_budget_tamper_fail_closed() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    amendment = contract.to_dict()["p95_runner_binding_retry_amendment"]
    amendment["failure_classification"][
        "offline_candidate_admission_cells_generated"
    ] = 0
    tampered = SimpleNamespace(
        contract_id=evidence.PILOT_V210_CONTRACT_ID,
        status="draft",
        p95_runner_binding_retry_amendment=amendment,
    )
    with pytest.raises(PilotEvidenceError, match="implementation no-go lineage"):
        evidence._v210_parent_evidence_lineage(tampered)

    rows = _rows(contract)
    release = _release_controls()
    release["budget_ledger"]["actual_totals"]["storage_bytes"] = 1
    with pytest.raises(PilotEvidenceError, match="inherited debit/denominator"):
        evidence._v210_inherited_budget_boundary(
            contract,
            denominator=_denominator(rows),
            release_controls=release,
        )


def test_v210_copies_complete_manifest_chain_and_skips_v29_summarizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    chain = evidence._v210_source_manifest_amendment_chain(contract)

    assert [name for _, name in chain] == [
        "pilot_v2_10_source_manifest.json",
        "pilot_v2_9_source_manifest.json",
        "pilot_v2_8_source_manifest.json",
        "pilot_v2_7_source_manifest.json",
        "pilot_v2_6_source_manifest.json",
        "pilot_v2_5_source_manifest.json",
    ]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("V2.9 implementation summarizer was invoked")

    monkeypatch.setattr(evidence, "_v29_implementation_failure_summary", forbidden)
    assert (
        evidence._implementation_failure_summary_for_contract(
            {"contract_id": evidence.PILOT_V210_CONTRACT_ID},
            [],
            resolved_git_commit="a" * 40,
        )
        is None
    )


def test_v210_complete_c_lane_missing_sensitivity_fails_closed(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    source_root = tmp_path / "source"
    raw = source_root / "raw"
    raw.mkdir(parents=True)

    with pytest.raises(PilotEvidenceError, match="local.*missing or unsafe"):
        evidence._validated_v210_experiment_c_sensitivities(
            contract,
            raw_root=raw,
            rows=rows,
            common_commit="1" * 40,
            source_repo_root=source_root,
        )


def test_v210_tampered_c_lane_sensitivity_fails_closed_and_restores_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    source_root = tmp_path / "source"
    raw = _write_sensitivity_fixtures(
        source_root,
        contract,
        rows,
        commit="2" * 40,
    )
    local_path = raw / "local-experiment-c" / "rule_sensitivity.json"
    tampered = json.loads(local_path.read_text(encoding="utf-8"))
    tampered["provider_calls"] = 1
    local_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _install_sensitivity_loader_fixture(
        monkeypatch,
        source_root=source_root,
    )
    previous = Path.cwd()

    with pytest.raises(
        PilotEvidenceError,
        match="local.*bindings, grid, or source manifests drifted",
    ):
        evidence._validated_v210_experiment_c_sensitivities(
            contract,
            raw_root=raw,
            rows=rows,
            common_commit="2" * 40,
            source_repo_root=source_root,
        )
    assert Path.cwd() == previous


def test_v210_complete_publication_copies_both_sensitivities_from_external_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    commit = "3" * 40
    source_root = tmp_path / "external-source"
    raw = _write_sensitivity_fixtures(
        source_root,
        contract,
        rows,
        commit=commit,
    )
    (raw / "run_ledger.json").write_text("{}\n", encoding="utf-8")
    observed = _install_sensitivity_loader_fixture(
        monkeypatch,
        source_root=source_root,
    )
    _install_supported_gate_fixtures(monkeypatch)
    monkeypatch.setattr(
        evidence,
        "_normalize_ledger",
        lambda *_args, **_kwargs: (
            rows,
            _denominator(rows),
            commit,
        ),
    )
    monkeypatch.setattr(
        evidence,
        "_validated_release_controls",
        lambda *_args, **_kwargs: _release_controls(),
    )
    previous = Path.cwd()

    receipt = evidence.build_pilot_v24_evidence_package(
        contract_path=CONTRACT_PATH,
        run_ledger_path=raw / "run_ledger.json",
        raw_root=raw,
        build_root=tmp_path / "build",
        source_repo_root=source_root,
    )

    assert receipt.scientific_complete is True
    assert observed == [
        ("local-experiment-c", "llama33_local_controlled"),
        ("experiment-c", "gpt52_main"),
    ]
    expected_files = {
        "local": "local_experiment_c_rule_sensitivity.json",
        "gpt52": "experiment_c_rule_sensitivity.json",
    }
    manifest = json.loads(receipt.manifest_path.read_text(encoding="utf-8"))
    checksums = json.loads(receipt.checksums_path.read_text(encoding="utf-8"))
    checksum_paths = {row["path"] for row in checksums["files"]}
    for lane_id, package_path in expected_files.items():
        published = receipt.package_dir / package_path
        source_stage = evidence._v210_sensitivity_lane_definition(lane_id)[
            "stage_id"
        ]
        source = raw / source_stage / "rule_sensitivity.json"
        assert published.read_bytes() == source.read_bytes()
        assert package_path in manifest["published_files"]
        assert package_path in checksum_paths
        assert (
            manifest["experiment_c_rule_sensitivities"][lane_id][
                "package_path"
            ]
            == package_path
        )
        assert (
            manifest["experiment_c_rule_sensitivities"][lane_id]["published"]
            is True
        )
    claims = json.loads(
        (receipt.package_dir / "claim_metric_artifact.json").read_text(
            encoding="utf-8"
        )
    )["claims"]
    assert {
        row["artifact"]
        for row in claims
        if "rule sensitivity" in row["claim"]
    } == set(expected_files.values())
    assert set(receipt.claim_gates["experiment_c_rule_sensitivities"]) == {
        "local",
        "gpt52",
    }
    assert Path.cwd() == previous


def test_v210_orchestrator_sensitivity_loader_forwards_authority_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    authority_root = (tmp_path / "external-authority").resolve()
    authority_root.mkdir()
    value = {
        "bindings": {
            "git_tag": contract.implementation["required_git_tag"],
            "git_commit": "4" * 40,
        },
        "integrity": {"content_sha256": "5" * 64},
    }
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        orchestrator,
        "_read_json",
        lambda _path: json.loads(json.dumps(value)),
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_bound_payload",
        lambda *_args, **_kwargs: None,
    )

    def build(
        _contract,
        *,
        raw_root,
        git_tag,
        git_commit,
        stage_id,
        model_id,
        authority_repo_root,
    ):
        observed.update(
            {
                "raw_root": raw_root,
                "git_tag": git_tag,
                "git_commit": git_commit,
                "stage_id": stage_id,
                "model_id": model_id,
                "authority_repo_root": authority_repo_root,
            }
        )
        return {"bindings": value["bindings"]}

    monkeypatch.setattr(
        orchestrator,
        "_build_experiment_c_sensitivity",
        build,
    )
    raw = tmp_path / "raw"

    assert (
        orchestrator._load_verified_experiment_c_sensitivity(
            contract,
            raw_root=raw,
            paid=None,
            stage_id="local-experiment-c",
            model_id="llama33_local_controlled",
            authority_repo_root=authority_root,
        )
        == value
    )
    assert observed == {
        "raw_root": raw,
        "git_tag": contract.implementation["required_git_tag"],
        "git_commit": "4" * 40,
        "stage_id": "local-experiment-c",
        "model_id": "llama33_local_controlled",
        "authority_repo_root": authority_root,
    }
