from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath

import pytest

from verified_memory.pilot_budget import ParentBudgetDebit
from verified_memory import pilot_v2111_parent_import as parent_import


ROOT = Path(__file__).resolve().parents[1]
CHILD_CONTRACT_SHA256 = "a" * 64
CHILD_GIT_TAG = "pilot-v2.11.1-science"
CHILD_GIT_COMMIT = "b" * 40


def _usage_rows(model: str, cost: float) -> list[dict[str, object]]:
    rows = []
    for index in range(30):
        rows.append(
            {
                "response_model": model,
                "call_kind": "action" if index < 24 else "semantic",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": cost / 30,
                },
            }
        )
    return rows


def _samples() -> dict[str, list[dict[str, object]]]:
    sample = {
        "finish_reason": "stop",
        "response_completed": True,
        "output_disposition": "accepted",
        "error_type": None,
        "parse_success": True,
        "clipped": False,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "reasoning_tokens": 1,
        "visible_completion_tokens": 4,
    }
    return {
        "action": [dict(sample) for _ in range(24)],
        "semantic": [dict(sample) for _ in range(6)],
    }


def _fake_audit(repo_root: Path) -> dict[str, object]:
    manifest = json.loads(
        (ROOT / "experiments" / "pilot_v2_11_1_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities: dict[str, dict[str, object]] = {}
    profiles = {
        "gpt52_main": (
            "openai/gpt-5.2-2025-12-11",
            "gpt-5.2-2025-12-11",
            0.53580625,
        ),
        "gpt56_diagnostic": (
            "openai/gpt-5.6-sol",
            "gpt-5.6-sol",
            0.585795,
        ),
    }
    for model_id, (runtime, served, cost) in profiles.items():
        capabilities[model_id] = {
            "model_id": model_id,
            "run_id": manifest["capability_source"]["models"][model_id]["run_id"],
            "runtime_model": runtime,
            "requested_model": served,
            "served_model": served,
            "taskset_sha256": manifest["capability_source"]["taskset_sha256"],
            "historical_source_calls": 30,
            "action_sample_count": 24,
            "semantic_sample_count": 6,
            "category_totals": {
                "utility-ranking": {"registered_correct": 12},
                "rule-application": {"registered_correct": 12},
                "rule-proposal": {"registered_correct": 6},
            },
            "checks": {
                "utility-ranking": True,
                "rule-application": True,
                "rule-proposal": True,
            },
            "interface_gate": {"pass": True, "failure_count": 0},
            "capability_assessment": {
                "status": "pass",
                "pass": True,
                "checks": {
                    "utility-ranking": True,
                    "rule-application": True,
                    "rule-proposal": True,
                },
            },
            "prompt_tier_gate": {
                "upper_bound_method": "utf8-bytes-plus-256-v1",
                "ceiling_tokens": 200000,
                "maximum_upper_bound_tokens": 6292,
                "passed": True,
            },
            "actual_usage": {
                "prompt_tokens": 300,
                "completion_tokens": 150,
                "total_tokens": 450,
                "cost_usd": cost,
            },
            "samples": _samples(),
            "usage_rows": _usage_rows(served, cost),
            "provider_failure_count": 0,
            "parse_failure_count": 0,
            "recovered_parse_count": 0,
            "strict_parse_count": 30,
            "truncation_count": 0,
            "capability_pass": True,
            "interface_pass": True,
            "stage_receipt_content_sha256": manifest["capability_source"][
                "stage_receipt"
            ]["content_sha256"],
        }
    failures = {
        model_id: {
            "model_id": model_id,
            "run_id": row["run_id"],
            "status": "failed",
            "provider_calls": 0,
            "cost_usd": 0.0,
            "failure_reason": row["failure_reason"],
            "bindings": {
                key: dict(row[key])
                for key in (
                    "provider_catalog",
                    "run_intent",
                    "failure",
                    "failure_manifest",
                )
            },
        }
        for model_id, row in manifest["failed_preflight_source"]["models"].items()
    }
    return {
        "repo_root": repo_root,
        "parent_root": repo_root / "immutable-parent",
        "manifest": manifest,
        "calibration": {
            "q_ref": 63.50397933257746,
            "selected_utility_profile": {
                "profile_id": "nu-0.5",
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 0.5,
                "consumption_scale": 63.50397933257746,
                "discount_factor": 0.99,
                "budget_tolerance": 1e-8,
                "max_labor_hours": 168.0,
            },
            "stage0_absolute_flow_utility_threshold": {
                "value": 0.05617208967516696,
                "treatment_outcomes_inspected": False,
            },
            "source_bindings": {},
        },
        "capabilities": capabilities,
        "preflight_failures": failures,
    }


@pytest.fixture
def fake_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repo_root = tmp_path / "child"
    repo_root.mkdir()
    audit = _fake_audit(repo_root)
    monkeypatch.setattr(
        parent_import,
        "_audit_sources",
        lambda **_kwargs: audit,
    )
    return repo_root


def _build(repo_root: Path) -> dict[str, object]:
    return parent_import.build_v2111_parent_import(
        repo_root=repo_root,
        child_contract_sha256=CHILD_CONTRACT_SHA256,
        child_git_tag=CHILD_GIT_TAG,
        child_git_commit=CHILD_GIT_COMMIT,
    )


def _verify(
    receipt: dict[str, object],
    repo_root: Path,
) -> dict[str, object]:
    return parent_import.verify_v2111_parent_import_receipt(
        receipt,
        repo_root=repo_root,
        child_contract_sha256=CHILD_CONTRACT_SHA256,
        child_git_tag=CHILD_GIT_TAG,
        child_git_commit=CHILD_GIT_COMMIT,
    )


def test_tracked_source_manifest_is_exact_and_freezes_no_go() -> None:
    raw = (ROOT / "experiments" / "pilot_v2_11_1_source_manifest.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        parent_import.V2111_SOURCE_MANIFEST_FILE_SHA256
    )
    manifest = parent_import._strict_json(raw, name="tracked manifest")
    parent_import._verify_seal(
        manifest,
        schema_version=parent_import.V2111_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="tracked manifest",
    )
    assert manifest["integrity"]["content_sha256"] == (
        parent_import.V2111_SOURCE_MANIFEST_CONTENT_SHA256
    )
    assert manifest["terminal_denominator"] == {
        "all_cells_terminal": True,
        "post_gate_authority_created": False,
        "registered_cells": 136,
        "scientific_matrix_complete": False,
        "stage_receipt_created": False,
        "stage_status_counts": {
            "capability-gate": {"complete": 2},
            "cross-model": {"integrity-stopped": 6},
            "experiment-a": {"integrity-stopped": 20},
            "experiment-b": {"integrity-stopped": 25},
            "experiment-c": {"integrity-stopped": 25},
            "experiment-d": {"integrity-stopped": 55},
            "long-context-preflight": {"failed": 2},
            "parent-import": {"complete": 1},
        },
        "status_counts": {
            "complete": 3,
            "failed": 2,
            "integrity-stopped": 131,
        },
    }
    assert manifest["failed_preflight_source"]["expected_absent"] == [
        (
            "experiment_results/pilot-v2.11/raw/long-context-preflight/"
            "post_gate_authority.json"
        ),
        (
            "experiment_results/pilot-v2.11/raw/long-context-preflight/"
            "stage_receipt.json"
        ),
    ]
    assert all(
        row["provider_calls"] == 0 and row["cost_usd"] == 0.0
        for row in manifest["failed_preflight_source"]["models"].values()
    )


def test_build_verify_wrappers_and_budget_are_zero_provider(
    fake_sources: Path,
) -> None:
    receipt = _build(fake_sources)
    assert _verify(receipt, fake_sources) == receipt
    assert receipt["import_policy"] == {
        "provider_construction_during_import": False,
        "provider_calls_during_import": 0,
        "hosted_provider_calls_during_import": 0,
        "imported_effect_cells": 0,
        "effect_metrics_observed": False,
        "imported_p95_authorities": [],
        "raw_tree_copied": False,
        "copied_file_count": 0,
        "copied_byte_count": 0,
        "imported_calibration_wrappers": 1,
        "imported_capability_wrappers": 2,
        "historical_capability_calls": 60,
        "historical_preflight_calls": 0,
        "historical_effect_cells_imported": 0,
        "validation_before_provider_construction": True,
    }
    wrappers = parent_import.capability_wrappers_from_v2111_receipt(receipt)
    assert set(wrappers) == {"gpt52_main", "gpt56_diagnostic"}
    for wrapper in wrappers.values():
        assert wrapper["provider_calls_current_attempt"] == 0
        assert wrapper["scientific_evidence"] is False
        assert len(wrapper["capability"]["usage_rows"]) == 30
        assert len(wrapper["capability"]["samples"]["action"]) == 24
        assert len(wrapper["capability"]["samples"]["semantic"]) == 6
    calibration = parent_import.calibration_wrapper_from_v2111_receipt(receipt)
    assert calibration["calibration"]["q_ref"] == 63.50397933257746
    assert calibration["provider_calls_current_attempt"] == 0
    debit = parent_import.parent_budget_debit_for_v2111(repo_root=fake_sources)
    assert isinstance(debit, ParentBudgetDebit)
    assert debit.cost_usd == 17.166524062500006
    assert debit.hosted_completions == 876
    assert debit.storage_bytes == 217581135
    assert debit.record_sha256 == (
        "e5b8406c636d5045040677ca0bd09dd72557afdef2998095f0f5775a0ead8b9c"
    )


def test_zero_provider_boundary_survives_poisoned_constructors(
    fake_sources: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_providers
    from verified_memory import pilot_orchestrator

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider construction is forbidden")

    monkeypatch.setattr(llm_providers, "MultiModelLLM", forbidden)
    monkeypatch.setattr(
        pilot_orchestrator,
        "_provider_for_profile",
        forbidden,
    )
    receipt = _build(fake_sources)
    assert receipt["import_policy"]["provider_calls_during_import"] == 0
    source = inspect.getsource(parent_import)
    assert "from llm_providers" not in source
    assert "import llm_providers" not in source


def test_resealed_receipt_tamper_is_rejected(fake_sources: Path) -> None:
    receipt = _build(fake_sources)
    unsigned = json.loads(json.dumps(receipt))
    unsigned.pop("integrity")
    wrapper = unsigned["capability_wrappers"]["gpt52_main"]
    wrapper.pop("integrity")
    wrapper["provider_calls_current_attempt"] = 1
    unsigned["capability_wrappers"]["gpt52_main"] = parent_import._seal(wrapper)
    tampered = parent_import._seal(unsigned)
    with pytest.raises(
        parent_import.PilotV2111ParentImportError,
        match="differs from exact parent replay",
    ):
        _verify(tampered, fake_sources)


def test_path_escape_source_symlink_and_root_symlink_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (root / "experiment_results").mkdir()
    linked = root / "experiment_results" / "linked.json"
    linked.symlink_to(outside)
    with pytest.raises(
        parent_import.PilotV2111ParentImportError,
        match="escaped",
    ):
        parent_import._normalized_relative(
            "experiment_results/../outside.json",
            required_top="experiment_results",
            name="source",
        )
    with pytest.raises(
        parent_import.PilotV2111ParentImportError,
        match="symlink",
    ):
        parent_import._guarded_file(
            root,
            PurePosixPath("experiment_results/linked.json"),
            name="source",
        )
    root_link = tmp_path / "source-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(
        parent_import.PilotV2111ParentImportError,
        match="symlink",
    ):
        parent_import._strict_root(root_link, name="source root")


def test_expected_absence_rejects_file_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "source"
    stage = root / "experiment_results" / "pilot" / "stage"
    stage.mkdir(parents=True)
    relative = PurePosixPath("experiment_results/pilot/stage/post_gate_authority.json")
    parent_import._verify_expected_absent(
        root,
        relative,
        name="post gate",
    )
    target = stage / "post_gate_authority.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        parent_import.PilotV2111ParentImportError,
        match="must remain absent",
    ):
        parent_import._verify_expected_absent(
            root,
            relative,
            name="post gate",
        )
    target.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    target.symlink_to(outside)
    with pytest.raises(
        parent_import.PilotV2111ParentImportError,
        match="must remain absent",
    ):
        parent_import._verify_expected_absent(
            root,
            relative,
            name="post gate",
        )


def test_bound_bytes_change_after_tamper(tmp_path: Path) -> None:
    root = tmp_path / "source"
    path = root / "experiment_results" / "artifact.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"value":1}\n', encoding="utf-8")
    _, original = parent_import._guarded_file(
        root,
        PurePosixPath("experiment_results/artifact.json"),
        name="artifact",
    )
    original_binding = parent_import._binding(
        PurePosixPath("experiment_results/artifact.json"),
        original,
    )
    path.write_text('{"value":2}\n', encoding="utf-8")
    _, tampered = parent_import._guarded_file(
        root,
        PurePosixPath("experiment_results/artifact.json"),
        name="artifact",
    )
    assert (
        parent_import._binding(
            PurePosixPath("experiment_results/artifact.json"),
            tampered,
        )
        != original_binding
    )


def test_persist_is_idempotent_compact_and_rejects_destination_symlink(
    fake_sources: Path,
    tmp_path: Path,
) -> None:
    kwargs = {
        "repo_root": fake_sources,
        "child_contract_sha256": CHILD_CONTRACT_SHA256,
        "child_git_tag": CHILD_GIT_TAG,
        "child_git_commit": CHILD_GIT_COMMIT,
    }
    first = parent_import.persist_v2111_parent_import(**kwargs)
    second = parent_import.persist_v2111_parent_import(**kwargs)
    assert first == second
    path = Path(first["receipt"])
    assert path.stat().st_size < 100_000
    assert first["provider_calls_during_import"] == 0
    assert list(path.parent.iterdir()) == [path]

    outside = tmp_path / "outside.json"
    outside.write_text("sentinel\n", encoding="utf-8")
    linked = fake_sources / "experiment_results" / "linked.json"
    linked.symlink_to(outside)
    with pytest.raises(
        parent_import.PilotV2111ParentImportError,
        match="symlink",
    ):
        parent_import.persist_v2111_parent_import(
            **kwargs,
            destination="experiment_results/linked.json",
        )
    assert outside.read_text(encoding="utf-8") == "sentinel\n"


def test_verified_capability_source_returns_raw_payload_and_exact_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "child"
    repo_root.mkdir()
    audit = _fake_audit(repo_root)
    parent_root = repo_root / "immutable-parent"
    parent_root.mkdir()
    audit["parent_root"] = parent_root
    model_id = "gpt52_main"
    source = parent_import._MODEL_SOURCES[model_id]
    payload = {
        "schema_version": "finevo-capability-gate-v5",
        "taskset_sha256": parent_import.CAPABILITY_TASKSET_SHA256,
        "rows": [{"task_id": f"task-{index}"} for index in range(30)],
    }
    payload_raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    payload_path = parent_root.joinpath(*source["capability_payload"].parts)
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(payload_raw)
    spec = {
        "run_id": parent_import._capability_run_id(model_id),
        "model_id": model_id,
        "stage_id": "capability-gate",
    }
    summary = {"payload": {"capability": payload}, "run_spec": spec}
    summary_path = parent_root.joinpath(*source["capability_summary"].parts)
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    binding = audit["manifest"]["capability_source"]["models"][model_id]["capability"]
    binding["byte_size"] = len(payload_raw)
    binding["file_sha256"] = hashlib.sha256(payload_raw).hexdigest()
    monkeypatch.setattr(
        parent_import,
        "_audit_sources",
        lambda **_kwargs: audit,
    )
    receipt = _build(repo_root)
    verified = parent_import.verified_v2111_capability_source(
        receipt,
        model_id=model_id,
        repo_root=repo_root,
        child_contract_sha256=CHILD_CONTRACT_SHA256,
        child_git_tag=CHILD_GIT_TAG,
        child_git_commit=CHILD_GIT_COMMIT,
    )
    assert verified["payload"] == payload
    assert verified["spec"] == spec
    assert verified["source"]["relative_path"] == (
        source["capability_payload"].as_posix()
    )
    assert verified["source"]["file_sha256"] == hashlib.sha256(payload_raw).hexdigest()
    assert verified["provider_calls_during_verification"] == 0
    parent_import._verify_seal(
        verified,
        schema_version=parent_import.V2111_VERIFIED_CAPABILITY_SOURCE_SCHEMA_VERSION,
        name="verified capability source",
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("child_contract_sha256", "x", "SHA-256"),
        ("child_git_tag", "pilot-v2.11-science", "child_git_tag"),
        ("child_git_commit", "y", "40-hex"),
    ],
)
def test_child_binding_is_strict(
    fake_sources: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    kwargs = {
        "repo_root": fake_sources,
        "child_contract_sha256": CHILD_CONTRACT_SHA256,
        "child_git_tag": CHILD_GIT_TAG,
        "child_git_commit": CHILD_GIT_COMMIT,
    }
    kwargs[field] = value
    with pytest.raises(
        parent_import.PilotV2111ParentImportError,
        match=match,
    ):
        parent_import.build_v2111_parent_import(**kwargs)
