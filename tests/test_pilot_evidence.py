from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from llm_providers import StructuredCompletion
from verified_memory.budget import UsageRecord
from verified_memory.pilot_contract import (
    PilotContractError,
    canonical_sha256,
    load_pilot_contract,
)
from verified_memory.pilot_evidence import (
    CURRENT_SCIENTIFIC_SCOPE,
    EVIDENCE_NAMESPACE,
    HISTORICAL_SCOPE,
    PilotEvidenceError,
    _experiment_a_gate,
    _experiment_c_gate,
    _cross_model_summary,
    _experiment_d_gate,
    _narrative_gate,
    _scientific_completion_status,
    _validate_provider_usage_rows,
    _validate_standard_run_contract,
    _validate_terminal_payload_marker,
    _validated_experiment_c_sensitivity,
    _validated_release_controls,
    build_pilot_evidence_package,
    write_terminal_summary,
)
from verified_memory.pilot_continuation import _completion_usage_row


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v1.yaml"


def _ledger(contract, *, overrides=None) -> dict:
    overrides = overrides or {}
    runs = {}
    for spec in contract.expand():
        row = {
            "spec": spec.to_dict(),
            "status": "failed",
            "artifact": None,
            "failure": {
                "error_type": "FixtureFailure",
                "message": "test fixture preserves the registered denominator",
            },
            "registered_at": "fixture",
            "terminal_at": "fixture",
        }
        row.update(overrides.get(spec.run_id, {}))
        runs[spec.run_id] = row
    return {
        "schema_version": "finevo-pilot-run-ledger-v1",
        "contract_hash": contract.canonical_hash,
        "created_at": "fixture",
        "updated_at": "fixture",
        "runs": runs,
    }


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _first_spec(contract, *, stage: str, arm: str):
    return contract.expand(stage=stage, arm=arm)[0]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provider_usage_rows(contract, spec, *, count: int = 1) -> list[dict]:
    profile = contract.provider_profiles[spec.model_id]
    provider = {
        "openai": "openai",
        "openrouter": "thirdparty",
        "ollama": "ollama",
    }[profile.transport]
    response_provider = {
        "openai": "OpenAI-direct",
        "openrouter": profile.provider_pin[0],
        "ollama": "local-ollama",
    }[profile.transport]
    response_route = {
        "openai": "direct",
        "openrouter": dict(profile.artifact_identity)["served_snapshot"],
        "ollama": "local",
    }[profile.transport]
    if profile.transport == "openai":
        reasoning_model = profile.requested_model.startswith(("gpt-5", "o1", "o3"))
        request_parameters = {
            "model",
            "messages",
            "top_p",
            *profile.openai_request_options().keys(),
            "max_completion_tokens" if reasoning_model else "max_tokens",
        }
        if not reasoning_model:
            request_parameters.add("temperature")
        sdk_name = "openai-python"
        sdk_version = "2.46.0"
        route_attestation_code = None
        temperature_dispatch = (
            "omitted_unsupported" if reasoning_model else "explicit"
        )
    elif profile.transport == "openrouter":
        request_parameters = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            *profile.openrouter_request_options().keys(),
        }
        sdk_name = "openai-python"
        sdk_version = "2.46.0"
        route_attestation_code = "OR_RA_PASS"
        temperature_dispatch = "explicit"
    else:
        request_parameters = {"model", "messages", "stream", "options"}
        if profile.json_mode == "json_object":
            request_parameters.add("format")
        sdk_name = "requests"
        sdk_version = "2.34.2"
        route_attestation_code = None
        temperature_dispatch = "explicit"
    if spec.decoding_seed is not None and profile.transport != "ollama":
        request_parameters.add("seed")
    return [
        {
            "schema_version": "verified-simulation-runner-v3",
            "provider": provider,
            "model": profile.requested_model,
            "response_model": profile.served_model,
            "attempts": 1,
            "error_type": None,
            "request_seed": spec.decoding_seed,
            "request_profile_id": profile.profile_id,
            "request_provider_pin": list(profile.provider_pin),
            "request_artifact_identity": dict(profile.artifact_identity),
            "request_price_snapshot_source": profile.price_snapshot.source,
            "request_price_snapshot_captured_at": (
                profile.price_snapshot.captured_at
            ),
            "response_provider": response_provider,
            "response_route": response_route,
            "finish_reason": "stop",
            "response_completed": True,
            "provider_error_details": None,
            "output_disposition": "accepted",
            "provider_sdk_name": sdk_name,
            "provider_sdk_version": sdk_version,
            "route_attestation_code": route_attestation_code,
            "temperature_dispatch": temperature_dispatch,
            "request_parameters": sorted(request_parameters),
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost_usd": 0.01,
            },
        }
        for _ in range(count)
    ]


def _rng_binding(seed: int) -> tuple[list[str], dict]:
    hashes = [_digest(f"rng:{seed}:{offset}") for offset in range(6)]
    return hashes, {
        "schema_version": "finevo-pilot-d-rng-schedule-v1",
        "derivation": "checkpoint-bound-domain-separated-sha256",
        "generated_before_provider_calls": True,
        "source_hash": _digest(f"rng-source:{seed}"),
        "schedule_hash": _digest(f"rng-schedule:{seed}"),
        "horizon": 6,
    }


def _continuation_causal(contract, *, seed: int, arm: str) -> dict:
    hashes, binding = _rng_binding(seed)
    treatment = {
        "matched-a": "matched-a",
        "matched-b": "matched-b",
        "no-memory": "no-memory",
        "shuffled-episodic": "shuffled-episodic",
        "wrong-context": "wrong-context",
        "error-verified": "erroneous-verified",
        "error-unverified": "erroneous-unverified",
    }[arm]
    common_start = _digest(f"error-common:{seed}")
    return {
        "kind": "continuation",
        "checkpoint_hash": _digest(f"checkpoint:{seed}"),
        "prefix_hash": _digest(f"prefix:{seed}"),
        "shock_schedule_hash": _digest("shock-schedule"),
        "pre_generated_rng_hashes": hashes,
        "rng_schedule_binding": binding,
        "shared_result_hash": _digest(f"continuation-result:{seed}"),
        "matched_replay_equal": True,
        "branch_treatment": treatment,
        "branch_rng_pre_step_hashes": hashes,
        "branch_action_completions": 24,
        "proposal_counters_before": {"0": 2, "1": 2, "2": 2, "3": 2},
        "proposal_counters_after": {"0": 2, "1": 2, "2": 2, "3": 2},
        "proposals_frozen": True,
        "focal_agent_id": 0,
        "wrong_context_source_agent_id": 1,
        "action_grid": dict(contract.stop_go["experiment_d"]["action_grid"]),
        "error_common_start_equal": True,
        "error_common_start_hash": common_start,
        "branch_forced_active_start_hash": (
            common_start
            if arm in {"error-verified", "error-unverified"}
            else None
        ),
    }


def _narrative_causal(contract, *, seed: int, narrative: str) -> dict:
    hashes, binding = _rng_binding(seed)
    return {
        "kind": "narrative",
        "checkpoint_hash": _digest(f"checkpoint:{seed}"),
        "prefix_hash": _digest(f"prefix:{seed}"),
        "pre_generated_rng_hashes": hashes,
        "rng_schedule_binding": binding,
        "shared_result_hash": _digest(f"narrative-result:{seed}"),
        "fixture_hash": contract.stop_go["experiment_d"][
            "narrative_fixture_hash"
        ],
        "branch_narrative_id": narrative,
        "branch_text_hash": _digest(f"narrative-text:{narrative}"),
        "branch_rng_pre_step_hashes": hashes,
        "branch_action_completions": 24,
        "proposal_counters_before": {"0": 2, "1": 2, "2": 2, "3": 2},
        "proposal_counters_after": {"0": 2, "1": 2, "2": 2, "3": 2},
        "proposals_frozen": True,
        "focal_agent_id": 0,
        "action_grid": dict(contract.stop_go["experiment_d"]["action_grid"]),
    }


def test_incomplete_package_preserves_every_itt_cell_and_narrows_claims(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    raw.mkdir()
    capability_spec = contract.expand(stage="capability-preflight")[0]
    ledger_path = _write_json(
        tmp_path / "ledger.json",
        _ledger(
            contract,
            overrides={
                capability_spec.run_id: {
                    "status": "capability-no-go",
                    "failure": {
                        "error_type": "CapabilityNoGo",
                        "message": "fixed capability threshold not met",
                    },
                }
            },
        ),
    )

    receipt = build_pilot_evidence_package(
        contract_path=CONTRACT_PATH,
        run_ledger_path=ledger_path,
        raw_root=raw,
        build_root=tmp_path / "evidence",
    )

    assert receipt.package_dir.relative_to(tmp_path / "evidence").as_posix() == (
        EVIDENCE_NAMESPACE
    )
    assert receipt.scientific_complete is False
    aggregate = json.loads((receipt.package_dir / "aggregate.json").read_text())
    assert aggregate["contract_sha256"] == contract.canonical_hash
    assert len(aggregate["rows"]) == len(contract.expand()) == 172
    assert aggregate["denominator"]["status_counts"] == {
        "capability-no-go": 1,
        "failed": 171,
    }
    assert aggregate["claim_gates"]["experiment_a"]["status"] == "no-go"
    assert aggregate["claim_gates"]["experiment_c"]["status"] == "no-go"
    assert aggregate["claim_gates"]["experiment_d"]["status"] == "no-go"

    failures = json.loads(
        (receipt.package_dir / "failure_ledger.json").read_text()
    )
    assert len(failures["rows"]) == 172
    assert any(row["status"] == "capability-no-go" for row in failures["rows"])
    assert {row["run_id"] for row in failures["rows"]} == {
        spec.run_id for spec in contract.expand()
    }
    report = (receipt.package_dir / "reviewer_report.md").read_text()
    for heading in (
        "Release, Stage-0, and budget",
        "Experiment A",
        "Experiment B",
        "Experiment C",
        "Experiment D",
        "Model capability",
        "Narrative content",
        "EconAgent / EconAI",
        "Claim narrowing",
    ):
        assert heading in report
    assert aggregate["release_controls"]["pass"] is False
    assert aggregate["experiment_c_rule_sensitivity"]["pass"] is False
    assert not (
        receipt.package_dir / "experiment_c_rule_sensitivity.json"
    ).exists()
    assert "backbone-independent" in report
    comparison = json.loads(
        (receipt.package_dir / "method_differences_scaffold.json").read_text()
    )
    assert comparison["status"] == "primary-source-backed method comparison"
    assert comparison["sources"]["econagent"]["url"].endswith("2310.10436")
    assert comparison["sources"]["econai"]["url"].endswith("2605.13762")
    assert "TODO" not in json.dumps(comparison)

    checksums = json.loads((receipt.package_dir / "checksums.json").read_text())
    declared = {row["path"]: row for row in checksums["files"]}
    assert "package_manifest.json" in declared
    assert f"contract/{CONTRACT_PATH.name}" in declared
    for relative, row in declared.items():
        data = (receipt.package_dir / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        assert len(data) == row["byte_size"]


def test_experiment_c_sensitivity_is_exact_grid_and_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    sensitivity_path = raw / "experiment-c" / "rule_sensitivity.json"
    sensitivity_path.parent.mkdir(parents=True)
    sensitivity_path.write_text("{}\n", encoding="utf-8")
    commit = "d" * 40

    rows = [
        {
            "run_id": spec.run_id,
            "stage_id": spec.stage_id,
            "model_id": spec.model_id,
            "arm_id": spec.arm_id,
            "status": "complete",
            "scientific_eligible": True,
            "artifact_sha256": _digest(spec.run_id),
        }
        for spec in contract.expand()
        if (
            spec.stage_id == "experiment-c"
            or (
                spec.stage_id == "experiment-b"
                and spec.model_id == "gpt52_main"
                and spec.arm_id == "full"
            )
        )
    ]
    sources = [
        {
            "run_id": row["run_id"],
            "manifest_sha256": row["artifact_sha256"],
        }
        for row in rows
        if row["stage_id"] == "experiment-b"
    ]
    weights = list(
        contract.stop_go["experiment_c"]["zero_api_sensitivity"][
            "alternative_success_weights"
        ]
    )
    outcomes = list(
        contract.stop_go["experiment_c"]["zero_api_sensitivity"][
            "outcome_definitions"
        ]
    )
    value = {
        "schema_version": "finevo-experiment-c-sensitivity-v1",
        "status": "pass",
        "terminal": True,
        "control_kind": "zero-api-offline-rule-sensitivity",
        "provider_calls": 0,
        "descriptive_only": True,
        "effectiveness_gate": False,
        "alternative_success_weights": weights,
        "outcome_definitions": outcomes,
        "source_run_count": 5,
        "aggregate_cells": [
            {
                "alternative_success_weight": weight,
                "outcome_definition": outcome,
            }
            for weight in weights
            for outcome in outcomes
        ],
        "bindings": {
            "contract_sha256": contract.canonical_hash,
            "git_tag": contract.implementation["required_git_tag"],
            "git_commit": commit,
            "source_manifests": sources,
        },
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
            "content_sha256": _digest("sensitivity"),
        },
    }

    import verified_memory.pilot_orchestrator as orchestrator

    monkeypatch.setattr(
        orchestrator,
        "_load_verified_experiment_c_sensitivity",
        lambda *args, **kwargs: value,
    )
    artifact, control = _validated_experiment_c_sensitivity(
        contract,
        raw_root=raw,
        rows=rows,
        common_commit=commit,
    )
    assert artifact == value
    assert control["pass"] is True
    assert control["provider_calls"] == 0
    assert control["grid_cell_count"] == 9

    value["aggregate_cells"][-1] = dict(value["aggregate_cells"][0])
    with pytest.raises(PilotEvidenceError, match="missing or duplicate"):
        _validated_experiment_c_sensitivity(
            contract,
            raw_root=raw,
            rows=rows,
            common_commit=commit,
        )


def test_standard_run_contract_rebuilds_config_and_provider_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = _first_spec(contract, stage="experiment-a", arm="full")
    profile = contract.provider_profiles[spec.model_id]
    expected_config = {
        "run_id": spec.run_id,
        "seed": spec.environment_seed,
        "scientific_scope": "preregistered_mechanism_micro_pilot",
    }

    class ExpectedConfig:
        def to_dict(self) -> dict:
            return dict(expected_config)

    import verified_memory.pilot_orchestrator as orchestrator
    import verified_memory.runner as runner_module

    monkeypatch.setattr(
        orchestrator,
        "_runner_p95_reservations",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "config_for_spec",
        lambda *args, **kwargs: ExpectedConfig(),
    )
    monkeypatch.setattr(
        runner_module,
        "build_sealed_run_config",
        lambda config, **kwargs: config.to_dict(),
    )
    provenance_git = {
        "git_tag": contract.implementation["required_git_tag"],
        "head_commit": "e" * 40,
        "tag_commit": "e" * 40,
        "tag_object_type": "tag",
        "worktree_clean": True,
        "contract_binding": contract.validate_provenance(
            "e" * 40,
            contract.implementation["required_git_tag"],
        ),
        "release_attestation": None,
    }
    records = {"api_usage": _provider_usage_rows(contract, spec)}
    _validate_standard_run_contract(
        contract,
        spec.to_dict(),
        config=expected_config,
        summary={"provider_model": f"openai/{profile.requested_model}"},
        records=records,
        provenance_git=provenance_git,
        raw_root=tmp_path,
    )

    with pytest.raises(PilotEvidenceError, match="fields=\\['seed'\\]"):
        _validate_standard_run_contract(
            contract,
            spec.to_dict(),
            config={**expected_config, "seed": spec.environment_seed + 1},
            summary={"provider_model": f"openai/{profile.requested_model}"},
            records=records,
            provenance_git=provenance_git,
            raw_root=tmp_path,
        )

    records["api_usage"][0]["request_profile_id"] = "wrong-profile"
    with pytest.raises(PilotEvidenceError, match="frozen request"):
        _validate_standard_run_contract(
            contract,
            spec.to_dict(),
            config=expected_config,
            summary={"provider_model": f"openai/{profile.requested_model}"},
            records=records,
            provenance_git=provenance_git,
            raw_root=tmp_path,
        )

    opus_spec = contract.expand(
        stage="cross-model-sentinels",
        model="opus48_sentinel",
        arm="full",
    )[0]
    opus_profile = contract.provider_profiles[opus_spec.model_id]
    expected_config.clear()
    expected_config.update(
        {
            "run_id": opus_spec.run_id,
            "seed": opus_spec.environment_seed,
            "scientific_scope": "preregistered_mechanism_micro_pilot",
        }
    )
    opus_records = {"api_usage": _provider_usage_rows(contract, opus_spec)}
    _validate_standard_run_contract(
        contract,
        opus_spec.to_dict(),
        config=expected_config,
        summary={
            "provider_model": f"thirdparty/{opus_profile.requested_model}"
        },
        records=opus_records,
        provenance_git=provenance_git,
        raw_root=tmp_path,
    )
    opus_records["api_usage"][0]["response_route"] = "wrong-snapshot"
    with pytest.raises(PilotEvidenceError, match="frozen request"):
        _validate_standard_run_contract(
            contract,
            opus_spec.to_dict(),
            config=expected_config,
            summary={
                "provider_model": f"thirdparty/{opus_profile.requested_model}"
            },
            records=opus_records,
            provenance_git=provenance_git,
            raw_root=tmp_path,
        )
    records["api_usage"][0]["request_profile_id"] = profile.profile_id
    records["api_usage"][0]["response_provider"] = "wrong-provider"
    records["api_usage"][0]["response_route"] = "wrong-route"
    with pytest.raises(PilotEvidenceError, match="frozen request"):
        _validate_standard_run_contract(
            contract,
            spec.to_dict(),
            config=expected_config,
            summary={"provider_model": f"openai/{profile.requested_model}"},
            records=records,
            provenance_git=provenance_git,
            raw_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "verified-simulation-runner-v2"),
        ("request_price_snapshot_source", "forged-price-snapshot"),
        ("finish_reason", "length"),
        ("response_completed", False),
        ("provider_error_details", {"error_type": "ForgedError"}),
        ("output_disposition", "discarded_due_to_contract_failure"),
        ("provider_sdk_version", "0.0.0"),
        ("temperature_dispatch", "explicit"),
        ("request_parameters", ["messages"]),
    ],
)
def test_provider_usage_rows_reject_current_attestation_tampering(
    field: str,
    value: object,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = _first_spec(contract, stage="experiment-d", arm="matched-a")
    rows = _provider_usage_rows(contract, spec, count=24)

    _validate_provider_usage_rows(contract, spec.to_dict(), rows)

    rows[0][field] = value
    with pytest.raises(PilotEvidenceError):
        _validate_provider_usage_rows(contract, spec.to_dict(), rows)


def test_provider_usage_rows_require_openrouter_route_attestation() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(
        stage="cross-model-sentinels",
        model="opus48_sentinel",
        arm="full",
    )[0]
    rows = _provider_usage_rows(contract, spec)
    _validate_provider_usage_rows(contract, spec.to_dict(), rows)

    rows[0]["route_attestation_code"] = None
    with pytest.raises(PilotEvidenceError, match="exact completion"):
        _validate_provider_usage_rows(contract, spec.to_dict(), rows)


def test_provider_usage_rows_keep_read_only_v1_v2_compatibility() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = _first_spec(contract, stage="experiment-d", arm="matched-a")
    strict_only = {
        "finish_reason",
        "response_completed",
        "provider_error_details",
        "output_disposition",
        "provider_sdk_name",
        "provider_sdk_version",
        "route_attestation_code",
        "temperature_dispatch",
        "request_parameters",
    }
    for schema in (
        "verified-simulation-runner-v1",
        "verified-simulation-runner-v2",
    ):
        row = _provider_usage_rows(contract, spec)[0]
        row["schema_version"] = schema
        for field in strict_only:
            row.pop(field)
        _validate_provider_usage_rows(
            contract,
            spec.to_dict(),
            [row],
            expected_runner_schema=schema,
            allow_legacy=True,
        )


def test_continuation_usage_row_is_current_and_evidence_valid() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = _first_spec(contract, stage="experiment-d", arm="matched-a")
    profile = contract.provider_profiles[spec.model_id]
    expected = _provider_usage_rows(contract, spec)[0]
    completion = StructuredCompletion(
        text='{"work": 0.5, "consumption": 0.5}',
        usage=UsageRecord(prompt_tokens=10, completion_tokens=5, cost_usd=0.01),
        model=profile.requested_model,
        provider="openai",
        attempts=1,
        latency_seconds=0.01,
        request_seed=spec.decoding_seed,
        response_model=profile.served_model,
        response_provider="OpenAI-direct",
        response_route="direct",
        request_profile_id=profile.profile_id,
        request_provider_pin=tuple(profile.provider_pin),
        request_artifact_identity=tuple(profile.artifact_identity),
        request_price_snapshot_source=profile.price_snapshot.source,
        request_price_snapshot_captured_at=profile.price_snapshot.captured_at,
        finish_reason="stop",
        native_finish_reason="stop",
        response_completed=True,
        provider_sdk_name=expected["provider_sdk_name"],
        provider_sdk_version=expected["provider_sdk_version"],
        route_attestation_code=None,
        request_parameters=tuple(expected["request_parameters"]),
        temperature_dispatch=expected["temperature_dispatch"],
        output_disposition="accepted",
    )
    row = _completion_usage_row(
        completion,
        treatment="matched-a",
        decision_t=6,
        agent_id=0,
    )

    assert row["schema_version"] == "verified-simulation-runner-v3"
    _validate_provider_usage_rows(contract, spec.to_dict(), [row])


def test_capability_terminal_rejects_unknown_schema_version(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="capability-preflight")[0]
    payload = {
        "metrics": {},
        "gate_evidence": {"go": True},
        "capability": {
            "schema_version": "finevo-capability-gate-v999",
            "pass": True,
            "preflight_go": True,
        },
    }

    with pytest.raises(PilotEvidenceError, match="unsupported capability schema"):
        _validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            payload,
            raw_root=tmp_path,
        )


def test_complete_matrix_with_failed_claim_gate_is_complete_with_no_go() -> None:
    rows = [
        {
            "stage_id": stage_id,
            "status": "complete",
            "scientific_eligible": True,
        }
        for stage_id in (
            "stage0-calibration",
            "experiment-a",
            "experiment-b",
            "experiment-c",
            "experiment-d",
            "cross-model-sentinels",
        )
    ]
    matrix, claims, complete = _scientific_completion_status(
        denominator={"pass": True},
        release_controls={"pass": True},
        gates={
            "experiment_a": {"status": "supported"},
            "experiment_c": {"status": "no-go"},
            "experiment_d": {"status": "supported"},
            "narrative": {"status": "supported"},
        },
        rows=rows,
    )
    assert matrix is True
    assert claims is False
    assert complete is False


def test_release_stage0_and_budget_controls_are_independently_revalidated(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    raw.mkdir()
    commit = "b" * 40
    release = {
        "schema_version": "finevo-pilot-release-attestation-v1",
        "status": "pass",
        "head_commit": commit,
        "local_tag": {
            "name": contract.implementation["required_git_tag"],
            "kind": "annotated",
            "peeled_commit": commit,
        },
        "remote": {
            "tag_kind": "annotated",
            "tag_peeled_commit": commit,
            "main_commit": commit,
        },
        "github_actions": {
            "workflow_file": "verified-memory-ci.yml",
            "run": {
                "head_sha": commit,
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
            },
            "required_jobs": [
                {
                    "name": "Python 3.12.7 / ubuntu-24.04",
                    "status": "completed",
                    "conclusion": "success",
                },
                {
                    "name": "Python 3.12.7 / macos-14",
                    "status": "completed",
                    "conclusion": "success",
                },
            ],
        },
    }
    release["attestation_sha256"] = canonical_sha256(release)
    _write_json(raw / "release_attestation.json", release)

    stage0_specs = contract.expand(stage="stage0-calibration")
    rows = []
    sources = []
    for spec in stage0_specs:
        manifest_hash = _digest(f"manifest:{spec.run_id}")
        rows.append(
            {
                **spec.to_dict(),
                "status": "complete",
                "artifact_kind": "verified-run-manifest",
                "artifact_sha256": manifest_hash,
            }
        )
        sources.append(
            {
                "run_id": spec.run_id,
                "utility_profile_id": spec.utility_profile_id,
                "environment_seed": spec.environment_seed,
                "manifest": f"raw/{spec.run_id}/manifest.json",
                "manifest_sha256": manifest_hash,
            }
        )
    selection = {
        "schema_version": "finevo-stage0-selection-v1",
        "contract_sha256": contract.canonical_hash,
        "selected_profile_id": "center",
        "selected_utility": {"rho": 1.0},
        "outcome_fields_used": [],
        "bindings": {
            "contract_sha256": contract.canonical_hash,
            "git_tag": contract.implementation["required_git_tag"],
            "git_commit": commit,
            "source_manifests": sources,
        },
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
        },
    }
    selection["integrity"]["content_sha256"] = canonical_sha256(selection)
    _write_json(
        raw / "stage0-calibration" / "stage0_selection.json",
        selection,
    )
    _write_json(
        raw / "stage0-calibration" / "stage_receipt.json",
        {
            "schema_version": "finevo-pilot-stage-receipt-v1",
            "contract_sha256": contract.canonical_hash,
            "stage_id": "stage0-calibration",
            "status": "complete",
            "terminal": True,
            "go": True,
            "registered_run_count": len(stage0_specs),
            "complete_cell_count": len(stage0_specs),
        },
    )

    budget_ids = {
        spec.run_id: spec.budget_bucket
        for spec in contract.expand()
        if spec.execution_mode != "checkpoint_continuation"
    }
    budget_ids.update(
        {
            (
                f"{contract.contract_id}--experiment-d--gpt52_main--"
                f"checkpoint-group--s{seed}"
            ): "core"
            for seed in contract.seeds["sets"]["main"]
        }
    )
    caps = {
        "total_usd": float(contract.budgets["total_usd"]),
        "max_completions": int(
            contract.budgets["max_provider_completions"]
        ),
        "completion_scope": str(contract.budgets["completion_scope"]),
        "max_storage_bytes": int(contract.budgets["max_storage_bytes"]),
        "stage_usd_caps": dict(contract.budgets["stage_usd_caps"]),
        "automatic_reserve_usd": float(
            contract.budgets["automatic_reserve_usd"]
        ),
        "dispatchable_usd": float(contract.budgets["total_usd"])
        - float(contract.budgets["automatic_reserve_usd"]),
    }
    budget = {
        "schema_version": "finevo-pilot-budget-ledger-v1",
        "contract_hash": contract.canonical_hash,
        "caps": caps,
        "runs": {
            run_id: {
                "stage_bucket": bucket,
                "status": "complete",
                "reservation": {
                    "run_id": run_id,
                    "stage_bucket": bucket,
                    "cost_usd": 0.0,
                    "completions": 0,
                    "storage_bytes": 0,
                    "basis": {"fixture": True},
                },
                "actual": {
                    "cost_usd": 0.0,
                    "completions": 0,
                    "storage_bytes": 0,
                },
            }
            for run_id, bucket in budget_ids.items()
        },
    }
    _write_json(raw / "budget_ledger.json", budget)

    controls = _validated_release_controls(
        contract,
        raw_root=raw,
        rows=rows,
        common_commit=commit,
    )
    assert controls["pass"] is True
    assert controls["budget_ledger"]["checks"][
        "valid_finalized_dispatch_units"
    ] is True
    assert controls["budget_ledger"]["checks"][
        "all_artifact_backed_dispatches_accounted"
    ] is True

    budget["runs"][next(iter(budget["runs"]))]["status"] = "reserved"
    _write_json(raw / "budget_ledger.json", budget)
    stopped = _validated_release_controls(
        contract,
        raw_root=raw,
        rows=rows,
        common_commit=commit,
    )
    assert stopped["pass"] is False
    assert stopped["budget_ledger"]["pass"] is False

    budget["runs"] = {
        run_id: {
            **row,
            "status": "complete",
        }
        for run_id, row in budget["runs"].items()
        if run_id in {spec.run_id for spec in stage0_specs}
    }
    _write_json(raw / "budget_ledger.json", budget)
    legitimate_no_dispatch_subset = _validated_release_controls(
        contract,
        raw_root=raw,
        rows=rows,
        common_commit=commit,
    )
    assert legitimate_no_dispatch_subset["pass"] is True
    assert legitimate_no_dispatch_subset["budget_ledger"]["checks"][
        "valid_finalized_dispatch_units"
    ] is True


def test_actor_terminal_is_rejected_and_capability_terminal_binds_provenance(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    raw.mkdir()
    actor_spec = _first_spec(contract, stage="experiment-a", arm="full")
    rejected = raw / "experiment-a" / "full-summary.json"
    write_terminal_summary(
        rejected,
        contract=contract,
        run_spec=actor_spec,
        resolved_git_commit="1" * 40,
        git_tag=str(contract.implementation["required_git_tag"]),
        payload={"metrics": {}},
        scientific_evidence=True,
        diagnostic_only=False,
        evidence_scope=CURRENT_SCIENTIFIC_SCOPE,
    )
    actor_ledger = _ledger(
        contract,
        overrides={
            actor_spec.run_id: {
                "status": "complete",
                "artifact": str(rejected.relative_to(raw)),
                "failure": None,
            }
        },
    )
    with pytest.raises(PilotEvidenceError, match="verified-run manifest"):
        build_pilot_evidence_package(
            contract_path=CONTRACT_PATH,
            run_ledger_path=_write_json(
                tmp_path / "actor-ledger.json", actor_ledger
            ),
            raw_root=raw,
            build_root=tmp_path / "actor-evidence",
        )

    spec = contract.expand(stage="capability-preflight")[0]
    artifact = raw / "capability-preflight" / "summary.json"
    capability_payload = {
        "metrics": {},
        "gate_evidence": {"go": True},
        "capability": {
            "schema_version": "finevo-capability-gate-v1",
            "pass": True,
            "preflight_go": True,
        },
    }
    with pytest.raises(PilotContractError, match="annotated tag"):
        write_terminal_summary(
            raw / "wrong-tag.json",
            contract=contract,
            run_spec=spec,
            resolved_git_commit="1" * 40,
            git_tag="not-pilot-v1",
            payload=capability_payload,
            scientific_evidence=False,
            diagnostic_only=False,
            evidence_scope="preregistered_capability_gate",
        )
    write_terminal_summary(
        artifact,
        contract=contract,
        run_spec=spec,
        resolved_git_commit="1" * 40,
        git_tag=str(contract.implementation["required_git_tag"]),
        payload=capability_payload,
        scientific_evidence=False,
        diagnostic_only=False,
        evidence_scope="preregistered_capability_gate",
    )
    ledger = _ledger(
        contract,
        overrides={
            spec.run_id: {
                "status": "complete",
                "artifact": str(artifact.relative_to(raw)),
                "failure": None,
            }
        },
    )
    receipt = build_pilot_evidence_package(
        contract_path=CONTRACT_PATH,
        run_ledger_path=_write_json(tmp_path / "ledger.json", ledger),
        raw_root=raw,
        build_root=tmp_path / "evidence",
    )
    aggregate = json.loads((receipt.package_dir / "aggregate.json").read_text())
    row = next(item for item in aggregate["rows"] if item["run_id"] == spec.run_id)
    assert row["scientific_eligible"] is False
    assert row["artifact_kind"] == "terminal-summary"
    assert aggregate["resolved_git_commit"] == "1" * 40
    assert aggregate["claim_gates"]["experiment_a"]["status"] == "no-go"
    with pytest.raises(PilotEvidenceError, match="overwrite"):
        build_pilot_evidence_package(
            contract_path=CONTRACT_PATH,
            run_ledger_path=tmp_path / "ledger.json",
            raw_root=raw,
            build_root=tmp_path / "evidence",
        )


def test_each_non_runner_execution_mode_requires_its_terminal_marker(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    raw.mkdir()
    capability = contract.expand(stage="capability-preflight")[0]
    qref = contract.expand(stage="q-ref-resolution")[0]
    offline = _first_spec(
        contract,
        stage="experiment-c",
        arm="verified-error-candidate",
    )
    continuation = _first_spec(
        contract,
        stage="experiment-d",
        arm="matched-a",
    )
    causal = _continuation_causal(
        contract,
        seed=continuation.environment_seed,
        arm="matched-a",
    )
    continuation_api_usage = _provider_usage_rows(
        contract,
        continuation,
        count=24,
    )
    shared_source = {
        "schema_version": "finevo-pilot-continuation-v1",
        "checkpoint_hash": causal["checkpoint_hash"],
        "prefix_hash": causal["prefix_hash"],
        "shock_schedule_hash": causal["shock_schedule_hash"],
        "pre_generated_rng_hashes": causal["pre_generated_rng_hashes"],
        "rng_schedule_binding": causal["rng_schedule_binding"],
        "matched_replay_equal": True,
        "focal_agent_id": 0,
        "wrong_context_source_agent_id": 1,
        "action_grid": causal["action_grid"],
        "erroneous_forced_active_common_start": {
            "equal": True,
            "forced_active_start_hash": causal[
                "error_common_start_hash"
            ],
        },
        "branches": {
            "matched-a": {
                "rng_pre_step_hashes": causal[
                    "branch_rng_pre_step_hashes"
                ],
                "proposal_counters_before": causal[
                    "proposal_counters_before"
                ],
                "proposal_counters_after": causal[
                    "proposal_counters_after"
                ],
                "freeze_proposals": True,
                "api_usage": continuation_api_usage,
                "api_usage_hash": canonical_sha256(continuation_api_usage),
                "intervention": {"forced_active_start_hash": None},
            }
        },
    }
    source_result_hash = canonical_sha256(shared_source)
    causal["shared_result_hash"] = source_result_hash
    shared_source["result_hash"] = source_result_hash
    shared_source.update(
        {
            "contract_sha256": contract.canonical_hash,
            "diagnostic_only": False,
            "scientific_evidence": True,
        }
    )
    shared_source_path = _write_json(
        raw / "experiment-d" / "checkpoints" / "s" / "continuations.json",
        shared_source,
    )
    continuation_metrics = _d_row(
        contract,
        seed=continuation.environment_seed,
        arm="matched-a",
        labor=80.0,
        utility=10.0,
    )["metrics"]
    reliability = {
        "unsupported_candidate_rejected": True,
        "false_rule_ever_active": False,
        "unverified_false_rule_ever_active": True,
        "same_candidate_content": True,
        "provider_calls": 0,
    }
    fixtures = (
        (
            capability,
            {
                "metrics": {},
                "gate_evidence": {"go": True},
                "capability": {
                    "schema_version": "finevo-capability-gate-v1",
                    "pass": True,
                    "preflight_go": True,
                },
            },
            False,
            False,
            "preregistered_capability_gate",
        ),
        (
            qref,
            {
                "metrics": {},
                "gate_evidence": {"pass": True},
                "q_ref_resolution": {
                    "q_ref": 2.0,
                    "row_count": 48,
                    "source_manifest": "raw/qref/manifest.json",
                    "source_manifest_sha256": _digest("qref-manifest"),
                    "resolution_artifact": "raw/q_ref_resolution.json",
                },
            },
            False,
            True,
            "deterministic_q_ref_resolution",
        ),
        (
            offline,
            {
                "metrics": {"rule_reliability": reliability},
                "gate_evidence": reliability,
                "offline_source": "raw/offline_candidate_admission.json",
            },
            True,
            False,
            CURRENT_SCIENTIFIC_SCOPE,
        ),
        (
            continuation,
            {
                "metrics": continuation_metrics,
                "gate_evidence": {
                    "matched_replay_equal": True,
                    "checkpoint_hash": causal["checkpoint_hash"],
                    "prefix_hash": causal["prefix_hash"],
                    "causal_bindings": causal,
                },
                "completion_receipt": {
                    "branch_action_completions": 24,
                    "observed_trajectory_action_rows": 24,
                    "shared_budget": {"status": "complete"},
                },
                "shared_source": str(shared_source_path),
                "shared_source_sha256": hashlib.sha256(
                    shared_source_path.read_bytes()
                ).hexdigest(),
            },
            True,
            False,
            CURRENT_SCIENTIFIC_SCOPE,
        ),
    )
    for index, (
        spec,
        payload,
        scientific,
        diagnostic,
        scope,
    ) in enumerate(fixtures):
        artifact = raw / f"mode-{index}.json"
        write_terminal_summary(
            artifact,
            contract=contract,
            run_spec=spec,
            resolved_git_commit="a" * 40,
            git_tag=str(contract.implementation["required_git_tag"]),
            payload=payload,
            scientific_evidence=scientific,
            diagnostic_only=diagnostic,
            evidence_scope=scope,
        )
        ledger = _ledger(
            contract,
            overrides={
                spec.run_id: {
                    "status": "complete",
                    "artifact": str(artifact.relative_to(raw)),
                    "failure": None,
                }
            },
        )
        receipt = build_pilot_evidence_package(
            contract_path=CONTRACT_PATH,
            run_ledger_path=_write_json(
                tmp_path / f"mode-{index}-ledger.json",
                ledger,
            ),
            raw_root=raw,
            build_root=tmp_path / f"mode-{index}-evidence",
        )
        aggregate = json.loads(
            (receipt.package_dir / "aggregate.json").read_text()
        )
        row = next(
            item for item in aggregate["rows"] if item["run_id"] == spec.run_id
        )
        assert row["artifact_kind"] == "terminal-summary"

    tampered_source = json.loads(shared_source_path.read_text())
    tampered_source["branches"]["matched-a"]["api_usage"].pop()
    _write_json(shared_source_path, tampered_source)
    continuation_ledger = _ledger(
        contract,
        overrides={
            continuation.run_id: {
                "status": "complete",
                "artifact": "mode-3.json",
                "failure": None,
            }
        },
    )
    with pytest.raises(PilotEvidenceError, match="shared source SHA-256"):
        build_pilot_evidence_package(
            contract_path=CONTRACT_PATH,
            run_ledger_path=_write_json(
                tmp_path / "tampered-shared-source-ledger.json",
                continuation_ledger,
            ),
            raw_root=raw,
            build_root=tmp_path / "tampered-shared-source-evidence",
        )


def test_diagnostic_and_historical_core_evidence_fail_closed(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    raw.mkdir()
    spec = contract.expand(stage="capability-preflight")[0]

    with pytest.raises(PilotEvidenceError, match="non-diagnostic"):
        write_terminal_summary(
            raw / "invalid.json",
            contract=contract,
            run_spec=spec,
            resolved_git_commit="2" * 40,
            git_tag=str(contract.implementation["required_git_tag"]),
            payload={"metrics": {}},
            scientific_evidence=True,
            diagnostic_only=True,
            evidence_scope=CURRENT_SCIENTIFIC_SCOPE,
        )

    historical_spec = _first_spec(
        contract,
        stage="experiment-c",
        arm="verified-error-candidate",
    )
    historical = raw / "historical.json"
    write_terminal_summary(
        historical,
        contract=contract,
        run_spec=historical_spec,
        resolved_git_commit="2" * 40,
        git_tag=str(contract.implementation["required_git_tag"]),
        payload={"metrics": {}},
        scientific_evidence=False,
        diagnostic_only=False,
        evidence_scope=HISTORICAL_SCOPE,
    )
    ledger = _ledger(
        contract,
        overrides={
                historical_spec.run_id: {
                "status": "complete",
                "artifact": str(historical.relative_to(raw)),
                "failure": None,
            }
        },
    )
    with pytest.raises(PilotEvidenceError, match="historical_pre_p0_v1"):
        build_pilot_evidence_package(
            contract_path=CONTRACT_PATH,
            run_ledger_path=_write_json(tmp_path / "ledger.json", ledger),
            raw_root=raw,
            build_root=tmp_path / "evidence",
        )


def test_terminal_checksum_and_standard_manifest_are_reverified(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    raw.mkdir()
    spec = contract.expand(stage="capability-preflight")[0]
    terminal = raw / "terminal.json"
    write_terminal_summary(
        terminal,
        contract=contract,
        run_spec=spec,
        resolved_git_commit="3" * 40,
        git_tag=str(contract.implementation["required_git_tag"]),
        payload={
            "metrics": {},
            "gate_evidence": {"go": True},
            "capability": {
                "schema_version": "finevo-capability-gate-v1",
                "pass": True,
                "preflight_go": True,
            },
        },
        scientific_evidence=False,
        diagnostic_only=False,
        evidence_scope="preregistered_capability_gate",
    )
    terminal.chmod(0o644)
    value = json.loads(terminal.read_text())
    value["payload"]["capability"]["preflight_go"] = False
    _write_json(terminal, value)
    tampered_ledger = _ledger(
        contract,
        overrides={
            spec.run_id: {
                "status": "complete",
                "artifact": str(terminal.relative_to(raw)),
                "failure": None,
            }
        },
    )
    with pytest.raises(PilotEvidenceError, match="checksum mismatch"):
        build_pilot_evidence_package(
            contract_path=CONTRACT_PATH,
            run_ledger_path=_write_json(tmp_path / "tampered-ledger.json", tampered_ledger),
            raw_root=raw,
            build_root=tmp_path / "evidence-tampered",
        )

    spec = _first_spec(contract, stage="experiment-a", arm="full")
    copied = raw / "manifest-run"
    shutil.copytree(ROOT / "artifacts" / "verified_runs" / "g0-local-s11", copied)
    action_stream = copied / "streams" / "actions.jsonl"
    action_stream.chmod(0o644)
    action_stream.write_text(action_stream.read_text() + "{}\n")
    manifest_ledger = _ledger(
        contract,
        overrides={
            spec.run_id: {
                "status": "complete",
                "artifact": str(copied.relative_to(raw)),
                "failure": None,
            }
        },
    )
    with pytest.raises(PilotEvidenceError, match="sealed run validation failed"):
        build_pilot_evidence_package(
            contract_path=CONTRACT_PATH,
            run_ledger_path=_write_json(tmp_path / "manifest-ledger.json", manifest_ledger),
            raw_root=raw,
            build_root=tmp_path / "evidence-manifest",
        )


def test_unsealed_complete_json_is_refused(tmp_path: Path) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    raw.mkdir()
    spec = _first_spec(contract, stage="experiment-d", arm="matched-a")
    unsealed = _write_json(
        raw / "plain-result.json",
        {
            "contract_sha256": contract.canonical_hash,
            "scientific_evidence": True,
            "metrics": {"focal_utility": 100.0},
        },
    )
    ledger = _ledger(
        contract,
        overrides={
            spec.run_id: {
                "status": "complete",
                "artifact": str(unsealed.relative_to(raw)),
                "failure": None,
            }
        },
    )
    with pytest.raises(PilotEvidenceError, match="not a sealed terminal summary"):
        build_pilot_evidence_package(
            contract_path=CONTRACT_PATH,
            run_ledger_path=_write_json(tmp_path / "ledger.json", ledger),
            raw_root=raw,
            build_root=tmp_path / "evidence",
        )


def _a_rows(contract) -> list[dict]:
    expected_rows = 48
    rows = []
    phases = ("pre-shock", "shock", "recovery")
    for seed in contract.seeds["sets"]["main"]:
        for arm, utility in (
            ("no-context", 100.0),
            ("prompt-only", 100.0),
            ("retrieval-only", 106.0),
            ("full", 110.0),
        ):
            retrieval = arm in {"retrieval-only", "full"}
            prompt = arm in {"prompt-only", "full"}
            trace = [
                {
                    "decision_t": decision_t,
                    "agent_id": agent_id,
                    "retrieved_episode_ids": (
                        [f"episode-{max(decision_t - 1, 0)}-{agent_id}"]
                        if retrieval and decision_t > 0
                        else []
                    ),
                }
                for decision_t in range(12)
                for agent_id in range(4)
            ]
            relevance = {
                "schema_version": "finevo-pilot-analysis-v1",
                "k": 5,
                "by_phase": {
                    phase: {
                        "relevant": 4,
                        "retrieved": 4,
                        "relevance": 1.0,
                    }
                    for phase in phases
                },
            }
            rows.append(
                {
                    "stage_id": "experiment-a",
                    "model_id": "gpt52_main",
                    "arm_id": arm,
                    "environment_seed": seed,
                    "status": "complete",
                    "scientific_eligible": True,
                    "metrics": {
                        "utility": {
                            "shock_recovery_discounted": utility,
                            "utility_deficit_auc": {
                                "full": 2.0,
                                "prompt-only": 5.0,
                                "retrieval-only": 3.0,
                                "no-context": 6.0,
                            }[arm],
                            "recovery_periods_to_within_10pct_for_two": {
                                "full": 1.0,
                                "prompt-only": 3.0,
                                "retrieval-only": 2.0,
                                "no-context": 4.0,
                            }[arm],
                        },
                        "memory": {
                            "context_to_prompt_count": (
                                expected_rows if prompt else 0
                            ),
                            "context_to_retrieval_count": (
                                expected_rows if retrieval else 0
                            ),
                            "route_trace_top5": trace,
                            "route_relevance_at_5": (
                                relevance
                                if retrieval
                                else {
                                    "schema_version": "finevo-pilot-analysis-v1",
                                    "k": 5,
                                    "by_phase": {},
                                }
                            ),
                        },
                        "actions": {
                            "labor_hours_counts": {"80": 48},
                            "consumption_rate_counts": {"0.3": 48},
                        },
                    },
                }
            )
    return rows


def test_experiment_a_requires_real_retrieval_phase_and_corroborating_coverage() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _a_rows(contract)
    gate = _experiment_a_gate(contract, rows)
    assert gate["status"] == "supported"
    assert gate["retrieval_trace_and_phase_coverage_pass"] is True
    assert gate["corroborating_contrast"]["pair_count"] == 5
    assert (
        gate["secondary_paired_metrics"]["utility_deficit_auc"][
            "paired_summary_full_minus_prompt_only"
        ]["median"]
        == -3.0
    )
    assert (
        gate["secondary_paired_metrics"][
            "recovery_periods_to_within_10pct_for_two"
        ]["paired_summary_full_minus_prompt_only"]["direction_count"]
        == 5
    )

    missing_trace = json.loads(json.dumps(rows))
    target = next(
        row
        for row in missing_trace
        if row["arm_id"] == "full"
        and row["environment_seed"] == contract.seeds["sets"]["main"][0]
    )
    target["metrics"]["memory"]["route_trace_top5"] = []
    assert _experiment_a_gate(contract, missing_trace)["status"] == "no-go"

    incomplete_phase = json.loads(json.dumps(rows))
    target = next(
        row
        for row in incomplete_phase
        if row["arm_id"] == "retrieval-only"
    )
    del target["metrics"]["memory"]["route_relevance_at_5"]["by_phase"][
        "recovery"
    ]
    assert _experiment_a_gate(contract, incomplete_phase)["status"] == "no-go"

    insufficient_corroboration = [
        row
        for row in rows
        if not (
            row["arm_id"] in {"retrieval-only", "no-context"}
            and row["environment_seed"]
            in contract.seeds["sets"]["main"][3:]
        )
    ]
    gate = _experiment_a_gate(contract, insufficient_corroboration)
    assert gate["status"] == "no-go"
    assert "corroborating" in " ".join(gate["reasons"])


def _d_row(
    contract,
    *,
    seed: int,
    arm: str,
    labor: float,
    utility: float,
) -> dict:
    causal = _continuation_causal(contract, seed=seed, arm=arm)
    return {
        "stage_id": "experiment-d",
        "model_id": "gpt52_main",
        "arm_id": arm,
        "narrative_id": "none",
        "environment_seed": seed,
        "status": "complete",
        "scientific_eligible": True,
        "gate_evidence": {
            "matched_replay_equal": True,
            "checkpoint_hash": causal["checkpoint_hash"],
            "prefix_hash": causal["prefix_hash"],
            "causal_bindings": causal,
        },
        "metrics": {
            "continuation": {
                "focal": {
                    "first_step": {
                        "labor_hours": labor,
                        "consumption_rate": 0.30,
                        "immediate_flow_utility": utility / 6.0,
                        "next_wealth": 100.0 + utility,
                    },
                    "discounted_flow_utility_sum": utility,
                    "final_wealth": 100.0 + utility,
                },
                "population": {
                    "first_step": {
                        "average_next_wealth": 100.0 + utility,
                        "gini_next_wealth": 0.1,
                        "low_labor_rate": 0.0,
                    },
                    "flow_utility_sum": 4.0 * utility,
                    "average_final_wealth": 100.0 + utility,
                    "gini_final_wealth": 0.1,
                    "mean_low_labor_rate": 0.0,
                },
            }
        },
    }


def test_experiment_d_gate_is_recomputed_from_sealed_branch_metrics() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    seeds = contract.seeds["sets"]["main"]
    rows = []
    for seed in seeds:
        rows.extend(
            (
                _d_row(contract, seed=seed, arm="matched-a", labor=80.0, utility=10.0),
                _d_row(contract, seed=seed, arm="matched-b", labor=80.0, utility=10.0),
                _d_row(contract, seed=seed, arm="no-memory", labor=96.0, utility=15.0),
                # Qualified action change without a downstream utility change
                # must remain prompt sensitivity only.
                _d_row(
                    contract,
                    seed=seed,
                    arm="shuffled-episodic",
                    labor=96.0,
                    utility=10.0,
                ),
                _d_row(contract, seed=seed, arm="wrong-context", labor=80.0, utility=10.0),
                _d_row(contract, seed=seed, arm="error-verified", labor=80.0, utility=10.0),
                _d_row(contract, seed=seed, arm="error-unverified", labor=80.0, utility=10.0),
            )
        )

    gate = _experiment_d_gate(contract, rows)

    assert gate["status"] == "supported"
    assert gate["supported_treatments"] == ["no-memory"]
    assert gate["prompt_sensitivity_only_treatments"] == [
        "shuffled-episodic"
    ]
    no_memory = gate["treatment_gates"]["no-memory"]
    assert no_memory["action_gates"]["labor_hours"]["passes"] is True
    assert no_memory["six_step_discounted_utility_gate"]["passes"] is True
    assert no_memory["classification"] == "closed-loop-continuation-effect"
    shuffled = gate["treatment_gates"]["shuffled-episodic"]
    assert shuffled["qualified_action_change"] is True
    assert shuffled["qualified_closed_loop_effect"] is False
    assert shuffled["classification"] == "prompt-sensitivity-only"

    broken = json.loads(json.dumps(rows))
    for seed in seeds[:2]:
        target = next(
            row
            for row in broken
            if row["environment_seed"] == seed
            and row["arm_id"] == "wrong-context"
        )
        target["gate_evidence"]["causal_bindings"][
            "branch_rng_pre_step_hashes"
        ][0] = _digest(f"polluted:{seed}")
    assert _experiment_d_gate(contract, broken)["status"] == "no-go"


def test_narrative_gate_requires_action_utility_and_paraphrase_equivalence() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    seeds = contract.seeds["sets"]["main"]
    rows = []
    for seed in seeds:
        rows.extend(
            (
                _d_row(contract, seed=seed, arm="matched-a", labor=80.0, utility=10.0),
                _d_row(contract, seed=seed, arm="matched-b", labor=80.0, utility=10.0),
            )
        )
        for narrative, labor, utility in (
            ("none", 80.0, 10.0),
            ("aligned", 96.0, 15.0),
            ("paraphrase", 96.0, 15.0),
            ("opposite", 80.0, 10.0),
        ):
            causal = _narrative_causal(
                contract,
                seed=seed,
                narrative=narrative,
            )
            rows.append(
                {
                    "stage_id": "experiment-d",
                    "model_id": "gpt52_main",
                    "arm_id": "narrative-content",
                    "narrative_id": narrative,
                    "environment_seed": seed,
                    "status": "complete",
                    "scientific_eligible": True,
                    "gate_evidence": {
                        "causal_bindings": causal,
                    },
                    "metrics": {
                        "narrative": {
                            "first_labor_hours": labor,
                            "first_consumption_rate": 0.30,
                            "immediate_flow_utility": utility / 6.0,
                            "six_step_discounted_flow_utility": utility,
                            "final_wealth": 100.0 + utility,
                        }
                    },
                }
            )

    gate = _narrative_gate(contract, rows)

    assert gate["status"] == "supported"
    assert gate["semantic_response"] is True
    assert (
        gate["aligned_vs_opposite_action_gates"]["labor_hours"]["passes"]
        is True
    )
    assert gate["aligned_vs_opposite_six_step_utility_gate"]["passes"] is True
    assert gate["paraphrase_equivalent_seed_count"] == 5
    assert "real-news understanding" in gate["claim_boundary"]

    broken = json.loads(json.dumps(rows))
    for seed in seeds[:2]:
        target = next(
            row
            for row in broken
            if row["environment_seed"] == seed
            and row["arm_id"] == "narrative-content"
            and row["narrative_id"] == "opposite"
        )
        target["gate_evidence"]["causal_bindings"]["fixture_hash"] = _digest(
            f"wrong-fixture:{seed}"
        )
    assert _narrative_gate(contract, broken)["status"] == "no-go"


def _rule_units(seed: int, *, active_steps: int) -> list[dict]:
    units = [
        {
            "unit_id": f"run:s{seed}:a{agent}:family:fixed",
            "run_id": "run",
            "seed": seed,
            "agent_id": agent,
            "rule_family_id": "fixed",
            "source": "injected",
            "injected": True,
            "natural": False,
            "ever_active": True,
            "active_exposure_steps": active_steps,
        }
        for agent in range(4)
    ]
    units.append(
        {
            "unit_id": f"run:s{seed}:a0:family:natural",
            "run_id": "run",
            "seed": seed,
            "agent_id": 0,
            "rule_family_id": "natural",
            "source": "natural",
            "injected": False,
            "natural": True,
            "ever_active": False,
            "active_exposure_steps": 0,
        }
    )
    return units


def _c_rows(contract, *, negative_loss: bool = False) -> list[dict]:
    rows = []
    for seed in contract.seeds["sets"]["main"]:
        rows.append(
            {
                "stage_id": "experiment-c",
                "model_id": "gpt52_main",
                "arm_id": "verified-error-candidate",
                "environment_seed": seed,
                "status": "complete",
                "scientific_eligible": True,
                "metrics": {
                    "rule_reliability": {
                        "false_rule_ever_active": False,
                        "unverified_false_rule_ever_active": True,
                    }
                },
            }
        )
        for stage, arm, total, exposure, units in (
            (
                "experiment-c",
                "verified-error-forced",
                110.0 if negative_loss else 98.0,
                1,
                _rule_units(seed, active_steps=1),
            ),
            (
                "experiment-c",
                "unverified-error-forced",
                105.0 if negative_loss else 90.0,
                5,
                _rule_units(seed, active_steps=5),
            ),
            ("experiment-b", "full", 100.0, 0, []),
            ("experiment-b", "unverified-dual", 100.0, 0, []),
        ):
            rows.append(
                {
                    "stage_id": stage,
                    "model_id": "gpt52_main",
                    "arm_id": arm,
                    "environment_seed": seed,
                    "status": "complete",
                    "scientific_eligible": True,
                    "metrics": {
                        "total_discounted_utility": total,
                        "rule_reliability": {
                            "active_exposure_steps": exposure,
                            "harmful_to_retirement_delay": (
                                1 if arm == "verified-error-forced" else None
                            ),
                            "by_agent_rule_family": units,
                        },
                    },
                }
            )
    return rows


def test_experiment_c_clamps_losses_and_requires_seed_agent_family_units() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    supported = _experiment_c_gate(contract, _c_rows(contract))
    assert supported["status"] == "supported"
    assert supported["positive_unverified_utility_loss_seed_count"] == 5
    assert len(supported["natural_proposal_descriptive_audit"]) == 10
    first = next(iter(supported["forced_active_pairs"].values()))
    assert first["verified_signed_control_minus_error"] == 2.0
    assert first["unverified_cumulative_utility_loss"] == 10.0

    negative = _experiment_c_gate(
        contract,
        _c_rows(contract, negative_loss=True),
    )
    assert negative["status"] == "no-go"
    assert negative["positive_unverified_utility_loss_seed_count"] == 0
    first = next(iter(negative["forced_active_pairs"].values()))
    assert first["verified_signed_control_minus_error"] == -10.0
    assert first["verified_cumulative_utility_loss"] == 0.0
    assert first["unverified_cumulative_utility_loss"] == 0.0

    missing_units = _c_rows(contract)
    missing_seeds = set(contract.seeds["sets"]["main"][:2])
    for row in missing_units:
        if (
            row["stage_id"] == "experiment-c"
            and row["arm_id"] == "verified-error-forced"
            and row["environment_seed"] in missing_seeds
        ):
            del row["metrics"]["rule_reliability"][
                "by_agent_rule_family"
            ]
    no_units = _experiment_c_gate(contract, missing_units)
    assert no_units["status"] == "no-go"
    assert "seed-agent-rule-family" in " ".join(no_units["reasons"])


def test_cross_model_summary_reuses_gpt52_b_and_separates_competence() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = []
    for seed in contract.seeds["sets"]["cross-model"]:
        for arm, utility in (("full", 12.0), ("no-memory", 10.0)):
            rows.append(
                {
                    "stage_id": "experiment-b",
                    "model_id": "gpt52_main",
                    "arm_id": arm,
                    "environment_seed": seed,
                    "status": "complete",
                    "failure": None,
                    "scientific_eligible": True,
                    "metrics": {
                        "utility": {
                            "shock_recovery_discounted": utility
                        },
                        "guardrails": {"provider_failure_count": 0},
                        "memory": {
                            "proposal_parse_status_counts": {"parsed": 4}
                        },
                    },
                }
            )
    capability_payload = {
        "pass": True,
        "preflight_go": True,
        "parse_failure_count": 0,
        "provider_failure_count": 0,
        "category_totals": {
            "utility-ranking": {
                "correct": 12,
                "denominator": 12,
                "required": 10,
            },
            "rule-application": {
                "correct": 11,
                "denominator": 12,
                "required": 10,
            },
            "rule-proposal": {
                "correct": 6,
                "denominator": 6,
                "required": 5,
            },
        },
    }

    result = _cross_model_summary(
        contract,
        rows,
        {
            "gpt52_main": {
                "ledger_status": "complete",
                "artifact_validated": True,
                "capability": capability_payload,
            }
        },
    )

    gpt = result["gpt52_main"]
    assert gpt["source"].startswith("reused experiment-b")
    assert gpt["directional_micro_pilot_replication"] is True
    assert gpt["usable_paired_seeds"] == list(
        contract.seeds["sets"]["cross-model"]
    )
    assert gpt["proposal_competence"]["correct"] == 6
    assert gpt["rule_application_competence"]["correct"] == 11
    assert "opus48_sentinel" in result

    invalid_capability = _cross_model_summary(
        contract,
        rows,
        {
            "gpt52_main": {
                "ledger_status": "complete",
                "artifact_validated": False,
                "capability": capability_payload,
            }
        },
    )
    assert (
        invalid_capability["gpt52_main"][
            "directional_micro_pilot_replication"
        ]
        is False
    )


def test_cross_negative_requires_three_strict_negatives_and_opus_clears_null() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    seeds = contract.seeds["sets"]["cross-model"]
    capability_payload = {
        "pass": True,
        "preflight_go": True,
        "category_totals": {},
    }
    capability = {
        "gpt52_main": {
            "ledger_status": "complete",
            "artifact_validated": True,
            "capability": capability_payload,
        },
        "opus48_sentinel": {
            "ledger_status": "complete",
            "artifact_validated": True,
            "capability": capability_payload,
        },
    }
    rows = []
    for index, seed in enumerate(seeds):
        for arm, utility in (
            ("full", 10.0 if index == 0 else 9.0),
            ("no-memory", 10.0),
        ):
            rows.append(
                {
                    "stage_id": "experiment-b",
                    "model_id": "gpt52_main",
                    "arm_id": arm,
                    "environment_seed": seed,
                    "status": "complete",
                    "failure": None,
                    "scientific_eligible": True,
                    "metrics": {
                        "utility": {
                            "shock_recovery_discounted": utility
                        }
                    },
                }
            )
        for arm, utility in (
            ("full", 11.0),
            ("no-memory", 10.0),
            ("matched-a", 10.0),
            ("matched-b", 12.0),
        ):
            rows.append(
                {
                    "stage_id": "cross-model-sentinels",
                    "model_id": "opus48_sentinel",
                    "arm_id": arm,
                    "environment_seed": seed,
                    "status": "complete",
                    "failure": None,
                    "scientific_eligible": True,
                    "metrics": {
                        "utility": {
                            "shock_recovery_discounted": utility
                        }
                    },
                }
            )
    result = _cross_model_summary(contract, rows, capability)
    assert result["gpt52_main"]["direction"] == "mixed-or-incomplete"
    assert (
        result["gpt52_main"]["directional_micro_pilot_replication"]
        is False
    )
    opus = result["opus48_sentinel"]
    assert opus["effect_exceeds_matched_a_a_null"] is False
    assert opus["matched_null_resolution"].startswith("unresolved")
    assert opus["directional_micro_pilot_replication"] is False

    for row in rows:
        if (
            row["stage_id"] == "cross-model-sentinels"
            and row["model_id"] == "opus48_sentinel"
            and row["arm_id"] == "matched-b"
        ):
            row["metrics"]["utility"]["shock_recovery_discounted"] = 10.2
    cleared = _cross_model_summary(contract, rows, capability)[
        "opus48_sentinel"
    ]
    assert cleared["effect_exceeds_matched_a_a_null"] is True
    assert cleared["directional_micro_pilot_replication"] is True
