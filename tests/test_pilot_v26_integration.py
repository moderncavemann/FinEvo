from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_evidence as core_evidence
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v24_evidence as lane_evidence
from verified_memory import pilot_v26_parent_import as parent_import
from verified_memory.m0_utility import UtilityConfig
from verified_memory.pilot_budget import PilotBudgetLedger
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.runner import (
    OBSERVED_P95_AUTHORITY_ID,
    OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
    OBSERVED_P95_SOURCE_KIND,
    validate_preflight_p95_reservations,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OVERLAY_PATH = EXPERIMENTS / "pilot_v2_6_overlay.yaml"
EXPANDED_PATH = EXPERIMENTS / "pilot_v2_6.yaml"
EXPECTED_COMMIT = "a" * 40
MODEL_ID = "llama33_local_controlled"
RUNTIME_MODEL = "ollama/llama3.3:70b-instruct-q4_K_M"
SERVED_MODEL = "llama3.3:70b-instruct-q4_K_M"
V26_PROJECTION_SOURCE_KIND = "v2.5-terminal-parent-import-v2.6"


def _reservation(
    *,
    prompt_tokens: float,
    completion_tokens: float,
    sample_count: int,
) -> dict[str, Any]:
    reserved_prompt = math.ceil(prompt_tokens * 1.25)
    reserved_completion = math.ceil(completion_tokens * 1.25)
    return {
        "sample_count": sample_count,
        "raw_p95": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": 0.0,
        },
        "reserved_p95": {
            "prompt_tokens": reserved_prompt,
            "completion_tokens": reserved_completion,
            "total_tokens": reserved_prompt + reserved_completion,
            "cost_usd": 0.0,
        },
        "reserve_multiplier": 1.25,
    }


@pytest.fixture
def synthetic_v26_receipt() -> dict[str, Any]:
    contract = load_pilot_contract(OVERLAY_PATH)
    source_authority = {
        "authority_id": OBSERVED_P95_AUTHORITY_ID,
        "source_kind": OBSERVED_P95_SOURCE_KIND,
        "pilot_contract_hash": contract.canonical_hash,
        "pilot_tag": parent_import.V26_SCIENCE_TAG,
        "source_projection_schema_version": (
            OBSERVED_P95_PROJECTION_SCHEMA_VERSION
        ),
        "source_projection_file_sha256": "1" * 64,
        "source_projection_content_sha256": "2" * 64,
        "source_preflight_run_id": "fixture-v2.5-preflight",
        "source_preflight_run_spec_sha256": "3" * 64,
        "source_model_id": MODEL_ID,
        "source_served_model": SERVED_MODEL,
        "source_execution_artifact_sha256": "4" * 64,
        "source_provider_call_journal_sha256": "5" * 64,
    }
    reservations = {
        RUNTIME_MODEL: {
            "action": {
                "authority": deepcopy(source_authority),
                "reservation": _reservation(
                    prompt_tokens=907.0,
                    completion_tokens=33.0,
                    sample_count=36,
                ),
            },
            "semantic": {
                "authority": deepcopy(source_authority),
                "reservation": _reservation(
                    prompt_tokens=2499.0,
                    completion_tokens=345.0,
                    sample_count=10,
                ),
            },
        }
    }
    receipt = parent_import._seal(
        {
            "schema_version": (
                parent_import.V26_INHERITED_P95_RECEIPT_SCHEMA_VERSION
            ),
            "contract": {
                "path": "experiments/pilot_v2_6.yaml",
                "file_sha256": "6" * 64,
                "contract_id": contract.contract_id,
                "contract_sha256": contract.canonical_hash,
            },
            "git": {
                "tag": parent_import.V26_SCIENCE_TAG,
                "commit": EXPECTED_COMMIT,
            },
            "model": {
                "model_id": MODEL_ID,
                "runtime_model": RUNTIME_MODEL,
                "served_model": SERVED_MODEL,
            },
            "parent_source": {"fixture_sha256": "7" * 64},
            "reservations": reservations,
            "scientific_evidence": False,
            "evidence_use": "synthetic V2.6 inherited-authority fixture",
        }
    )
    ignored_root = ROOT / "experiment_results"
    ignored_root.mkdir(parents=True, exist_ok=True)
    raw_root = Path(
        tempfile.mkdtemp(
            prefix="pytest-v26-observed-adapter-",
            dir=ignored_root,
        )
    )
    receipt_path = raw_root / "observed_p95_authority_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        yield {
            "contract": contract,
            "raw_root": raw_root,
            "receipt": receipt,
            "receipt_path": receipt_path,
            "receipt_relative": receipt_path.relative_to(ROOT).as_posix(),
        }
    finally:
        shutil.rmtree(raw_root, ignore_errors=True)


def test_v26_generic_reader_reaches_runner_config_before_provider(
    synthetic_v26_receipt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = synthetic_v26_receipt
    adapter_calls: list[dict[str, Any]] = []

    def verify_child(
        receipt: dict[str, Any],
        *,
        repo_root: Path,
        expected_git_commit: str,
    ) -> dict[str, Any]:
        adapter_calls.append(deepcopy(receipt))
        assert receipt == case["receipt"]
        assert repo_root == ROOT
        assert expected_git_commit == EXPECTED_COMMIT
        return deepcopy(receipt["reservations"])

    monkeypatch.setattr(
        parent_import,
        "verify_v26_inherited_p95_receipt",
        verify_child,
    )
    binding = authority.verified_observed_p95_authority_binding(
        case["receipt_relative"],
        repo_root=ROOT,
        expected_git_commit=EXPECTED_COMMIT,
    )
    assert binding["reservations"] == case["receipt"]["reservations"]

    runner_reservations = deepcopy(binding["reservations"])
    for by_kind in runner_reservations.values():
        for entry in by_kind.values():
            entry["authority"].update(
                {
                    "source_authority_receipt_path": binding["receipt_path"],
                    "source_authority_receipt_file_sha256": binding[
                        "receipt_file_sha256"
                    ],
                    "source_authority_receipt_content_sha256": binding[
                        "receipt_content_sha256"
                    ],
                    "source_release_commit": binding["git_commit"],
                }
            )

    monkeypatch.setattr(
        "verified_memory.runner._verify_local_release_identity",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "resolve_utility",
        lambda *_args, **_kwargs: UtilityConfig(),
    )
    provider_constructions: list[str] = []

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        provider_constructions.append("forbidden")
        raise AssertionError("provider construction is outside this gate")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden_provider)
    spec = case["contract"].expand(stage="stage0-calibration")[0]
    config = orchestrator.config_for_spec(
        case["contract"],
        spec,
        raw_root=case["raw_root"],
        paid_provenance=SimpleNamespace(
            git_tag=parent_import.V26_SCIENCE_TAG,
            head_commit=EXPECTED_COMMIT,
        ),
        verify_bound_inputs=True,
        preflight_p95_reservations=runner_reservations,
    )
    validated = validate_preflight_p95_reservations(
        config,
        provider_model_name=RUNTIME_MODEL,
    )
    assert set(validated) == {"action"}
    assert provider_constructions == []
    assert adapter_calls


def test_unknown_inherited_observed_p95_schema_fails_closed(
    synthetic_v26_receipt: dict[str, Any],
) -> None:
    receipt = deepcopy(synthetic_v26_receipt["receipt"])
    receipt["schema_version"] = (
        "finevo-pilot-v9.9-inherited-observed-p95-authority-v1"
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="top-level shape or schema drifted",
    ):
        authority.verify_observed_p95_authority_receipt(
            receipt,
            repo_root=ROOT,
            expected_git_commit=EXPECTED_COMMIT,
        )


@pytest.mark.parametrize(
    ("source_kind", "path_suffix"),
    [
        ("v2.5-terminal-parent-import-v2.5", ""),
        (V26_PROJECTION_SOURCE_KIND, ".tampered"),
    ],
)
def test_v26_projection_rejects_source_kind_or_child_receipt_path_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    path_suffix: str,
) -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    raw_root = tmp_path / "raw"
    projection_path = parent_import.inherited_projection_path(raw_root, MODEL_ID)
    projection_path.parent.mkdir(parents=True)
    expected_receipt = parent_import.inherited_p95_receipt_path(raw_root, MODEL_ID)
    projection_path.write_text(
        json.dumps(
            {
                "schema_version": OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
                "model_id": MODEL_ID,
                "served_model": SERVED_MODEL,
                "projection": {},
                "bindings": {
                    "source_kind": source_kind,
                    "source_authority_receipt": (
                        str(expected_receipt) + path_suffix
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v26_parent_import_receipt",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_bound_payload",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="V2.6 inherited p95 projection source binding mismatch",
    ):
        orchestrator._load_v26_inherited_projection(
            contract,
            MODEL_ID,
            raw_root=raw_root,
            paid=SimpleNamespace(head_commit=EXPECTED_COMMIT),
        )


def test_v26_parent_debit_matches_existing_stage_cap_and_budget_ledger(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    debit = core_evidence._expected_parent_budget_debit(contract)
    assert debit is not None
    assert debit["stage_bucket"] == "parent_v23"
    assert debit["cost_usd"] == pytest.approx(3.212770875)
    assert debit["hosted_completions"] == 184
    assert debit["storage_bytes"] == 6_303_635
    assert contract.budgets["stage_usd_caps"]["hosted_confirmatory"] == (
        pytest.approx(495.787229125)
    )

    ledger = PilotBudgetLedger(
        tmp_path / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    ).snapshot()
    assert ledger["committed"]["cost_usd"] == pytest.approx(3.212770875)
    assert ledger["committed"]["completions"] == 184
    assert ledger["committed"]["storage_bytes"] == 6_303_635
    assert ledger["committed"]["stage_cost_usd"]["parent_v23"] == (
        pytest.approx(3.212770875)
    )


def test_v26_terminal_marker_dispatches_to_v26_parent_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    spec = contract.expand(stage="parent-import")[0]
    receipt_hash = "c" * 64
    calls: list[dict[str, Any]] = []

    def verifier(receipt_path, *, repo_root, contract, expected_git_commit):
        calls.append(
            {
                "receipt_path": receipt_path,
                "repo_root": repo_root,
                "contract_id": contract.contract_id,
                "expected_git_commit": expected_git_commit,
            }
        )
        return {"integrity": {"content_sha256": receipt_hash}}

    monkeypatch.setattr(
        parent_import,
        "verify_v26_parent_import_receipt",
        verifier,
    )
    payload = {
        "metrics": {},
        "gate_evidence": {
            "receipt": str(tmp_path / "parent_import_receipt.json"),
            "receipt_content_sha256": receipt_hash,
            "provider_calls": 0,
            "scientific_evidence": False,
        },
        "provider_calls": 0,
    }
    core_evidence._validate_terminal_payload_marker(
        contract,
        spec.to_dict(),
        payload,
        raw_root=tmp_path,
        resolved_git_commit=EXPECTED_COMMIT,
    )
    assert calls == [
        {
            "receipt_path": payload["gate_evidence"]["receipt"],
            "repo_root": ROOT,
            "contract_id": parent_import.V26_CONTRACT_ID,
            "expected_git_commit": EXPECTED_COMMIT,
        }
    ]


def _rows(contract) -> list[dict[str, Any]]:
    _, scientific_stages = core_evidence._stage_sets(contract)
    return [
        {
            **spec.to_dict(),
            "status": "complete",
            "failure": None,
            "artifact_kind": "terminal-summary",
            "artifact_sha256": "a" * 64,
            "scientific_eligible": spec.stage_id in scientific_stages,
            "metrics": {},
            "gate_evidence": {},
            "capability": {},
            "narrative": {},
        }
        for spec in contract.expand()
    ]


def _install_aggregate_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lane_evidence,
        "_experiment_c_gate",
        lambda _contract, _rows, *, stage_id, model_id: {
            "status": "supported",
            "scientific_evidence_complete": True,
            "same_direction_counts": {"false_activation": 5},
            "claim_action": f"retain {model_id}/{stage_id}",
        },
    )
    monkeypatch.setattr(
        lane_evidence,
        "_experiment_a_gate",
        lambda _contract, _rows, *, stage_id, model_id: {
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
        },
    )
    monkeypatch.setattr(
        lane_evidence,
        "_experiment_d_gate",
        lambda _contract, _rows, *, stage_id, model_id, arms: {
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
        },
    )
    monkeypatch.setattr(
        lane_evidence,
        "_experiment_b_summary",
        lambda _rows, *, stage_id, model_id, arms: {
            "comparison_type": "descriptive_preregistered_architecture_arms",
            "selection_rule": "do not select a winner",
            "arms": {arm: {} for arm in arms},
            "binding": f"{model_id}/{stage_id}",
        },
    )


@pytest.mark.parametrize("contract_path", [EXPANDED_PATH, OVERLAY_PATH])
def test_v26_publisher_copies_complete_contract_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_path: Path,
) -> None:
    contract = load_pilot_contract(contract_path)
    _install_aggregate_fixtures(monkeypatch)
    rows = _rows(contract)
    count = len(rows)
    aggregate = lane_evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator={
            "expected_count": count,
            "observed_ledger_count": count,
            "all_rows_present": True,
            "all_rows_terminal": True,
            "status_counts": {"complete": count},
            "all_completed_artifacts_validated": True,
            "pass": True,
        },
        release_controls={
            "pass": True,
            "budget_ledger": {
                "pass": True,
                "raw_root_storage_bytes": 0,
            },
        },
    )

    # Draft contracts deliberately carry null V2.6 source-manifest hashes.
    # Publication is normally gated on a frozen release; this test isolates
    # expanded/overlay copy topology while the amendment is still a draft.
    if contract.status == "draft":
        real_sha256_file = lane_evidence._sha256_file

        def draft_hash(path: Path) -> str | None:
            if Path(path).name == "pilot_v2_6_source_manifest.json":
                return None
            return real_sha256_file(path)

        monkeypatch.setattr(lane_evidence, "_sha256_file", draft_hash)

    manifest_path, checksums_path = lane_evidence._write_v24_package(
        tmp_path / contract_path.stem,
        contract_path=contract_path,
        contract=contract,
        rows=rows,
        aggregate=aggregate,
        common_commit=EXPECTED_COMMIT,
    )
    package = manifest_path.parent
    expected = {
        "contract/pilot_v2_4_parent_source_manifest.json",
        "contract/pilot_v2_5_source_manifest.json",
        "contract/pilot_v2_5.yaml",
        "contract/pilot_v2_6_source_manifest.json",
        f"contract/{contract_path.name}",
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    assert expected.issubset(set(manifest["published_files"]))
    assert expected.issubset({row["path"] for row in checksums["files"]})
    assert load_pilot_contract(
        package / "contract" / contract_path.name
    ).canonical_hash == contract.canonical_hash


def test_v26_modules_import_together_without_cycle() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import run_pilot; "
                "import verified_memory.observed_p95_authority; "
                "import verified_memory.pilot_orchestrator; "
                "import verified_memory.pilot_v26_parent_import"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
