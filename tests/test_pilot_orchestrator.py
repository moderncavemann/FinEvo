from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from llm_providers import MultiModelLLM
from verified_memory.artifacts import verify_manifest
from verified_memory.budget import BudgetLimits, RunBudget, UsageRecord
from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION
from verified_memory.pilot_orchestrator import (
    _preflight_checks,
    _execute_d_seed,
    _execute_q_ref,
    _offline_candidate_admission,
    GitProvenance,
    PilotOrchestrationError,
    PilotRunLedger,
    config_for_spec,
    execute_stage,
    run_development_fake_matrix,
    verify_paid_provenance,
)
from verified_memory.pilot_provider_catalog import (
    PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


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
    path.write_text(json.dumps(value), encoding="utf-8")


def _utility_fixture(contract, raw: Path) -> None:
    _write_json(
        raw / "stage0-calibration" / "stage0_selection.json",
        {
            "schema_version": "finevo-stage0-selection-v1",
            "selected_utility": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 1.0,
                "consumption_scale": 1.5,
                "discount_factor": 0.99,
            },
        },
    )


def test_itt_ledger_retains_failed_and_stopped_cells_on_reload(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    specs = contract.expand(stage="experiment-a")[:3]
    path = tmp_path / "run-ledger.json"
    ledger = PilotRunLedger(path, contract_hash=contract.canonical_hash)
    ledger.register(specs)
    ledger.finalize(
        specs[0].run_id,
        status="failed",
        artifact=None,
        failure={"error_type": "FixtureFailure"},
    )
    ledger.stop_pending(
        specs[1:2],
        status="budget-stopped",
        failure={"error_type": "FixtureBudgetStop"},
    )
    ledger.finalize(
        specs[2].run_id,
        status="capability-no-go",
        artifact=None,
        failure={"error_type": "FixtureCapabilityNoGo"},
    )

    reloaded = PilotRunLedger(path, contract_hash=contract.canonical_hash)
    assert [reloaded.status(spec.run_id) for spec in specs] == [
        "failed",
        "budget-stopped",
        "capability-no-go",
    ]
    assert set(reloaded.snapshot()["runs"]) == {spec.run_id for spec in specs}


def test_arm_mapping_freezes_unverified_policy_and_seed_capability(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    _utility_fixture(contract, tmp_path)
    unverified = contract.expand(
        stage="experiment-b",
        arm="unverified-dual",
    )[0]
    config = config_for_spec(
        contract,
        unverified,
        raw_root=tmp_path,
        paid_provenance=None,
        diagnostic_override=True,
    )
    assert config.context_mode == "full"
    assert config.enable_episodic_retrieval is True
    assert config.enable_semantic is True
    assert config.semantic_policy == "unverified-immediate"
    assert config.max_rule_proposals_per_agent == 4
    assert config.send_decoding_seed is True
    assert config.scientific_scope == "bounded_method_smoke"
    assert config.allow_scientific_scope is False

    opus = next(
        spec
        for spec in contract.expand(
            stage="cross-model-sentinels",
            model="opus48_sentinel",
        )
        if spec.arm_id == "full"
    )
    opus_config = config_for_spec(
        contract,
        opus,
        raw_root=tmp_path,
        paid_provenance=None,
        diagnostic_override=True,
    )
    assert opus.decoding_seed is None
    assert opus_config.send_decoding_seed is False


class _FakeContract:
    def __init__(self, base: str) -> None:
        self.implementation = {
            "required_git_tag": "pilot-v1",
            "p0_base_commit": base,
        }

    def validate_provenance(self, commit: str, tag: str) -> dict:
        if tag != "pilot-v1":
            raise ValueError("wrong tag")
        return {"git_tag": tag, "resolved_git_commit": commit}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_paid_provenance_requires_annotated_peeled_clean_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "pilot@example.test")
    _git(tmp_path, "config", "user.name", "Pilot Test")
    source = tmp_path / "source.txt"
    source.write_text("frozen\n", encoding="utf-8")
    _git(tmp_path, "add", "source.txt")
    _git(tmp_path, "commit", "-m", "frozen")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_pilot_release_attestation",
        lambda _root: SimpleNamespace(
            to_dict=lambda: _release_attestation(commit)
        ),
    )

    with pytest.raises(PilotOrchestrationError):
        verify_paid_provenance(
            _FakeContract(commit),  # type: ignore[arg-type]
            repo_root=tmp_path,
        )
    _git(tmp_path, "tag", "pilot-v1")
    with pytest.raises(PilotOrchestrationError, match="annotated tag"):
        verify_paid_provenance(
            _FakeContract(commit),  # type: ignore[arg-type]
            repo_root=tmp_path,
        )
    _git(tmp_path, "tag", "-d", "pilot-v1")
    _git(tmp_path, "tag", "-a", "pilot-v1", "-m", "pilot")

    provenance = verify_paid_provenance(
        _FakeContract(commit),  # type: ignore[arg-type]
        repo_root=tmp_path,
    )
    assert provenance.head_commit == commit
    assert provenance.tag_commit == commit
    assert provenance.tag_object_type == "tag"
    assert provenance.worktree_clean is True

    source.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(PilotOrchestrationError, match="clean worktree"):
        verify_paid_provenance(
            _FakeContract(commit),  # type: ignore[arg-type]
            repo_root=tmp_path,
        )


def test_one_catalog_no_go_does_not_block_other_capability_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = GitProvenance(
        git_tag="pilot-v1",
        head_commit="1" * 40,
        tag_commit="1" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding=contract.validate_provenance("1" * 40, "pilot-v1"),
        release_attestation=_release_attestation("1" * 40),
    )
    catalog_calls: list[tuple[str, ...]] = []
    dispatched: list[str] = []

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_paid_provenance",
        lambda *args, **kwargs: paid,
    )

    def fail_only_opus(_contract, *, model_ids):
        catalog_calls.append(tuple(model_ids))
        model_id = model_ids[0]
        if model_id == "opus48_sentinel":
            raise RuntimeError("frozen Opus request parameters are unsupported")
        return {
            "schema_version": PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "status": "pass",
            "paid_completions": 0,
            "rows": [{"model_id": model_id, "status": "pass"}],
        }

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        fail_only_opus,
    )

    def zero_projection(_contract, spec, **kwargs):
        return RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=0.0,
            completions=1,
            storage_bytes=1_000,
            basis={
                "method": "test-zero-projection",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
        )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.conservative_projection",
        zero_projection,
    )

    def complete_without_provider(_contract, spec, *, raw_root, **kwargs):
        dispatched.append(spec.model_id)
        artifact = raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json"
        _write_json(artifact, {"test_fixture": True})
        return (
            "complete",
            artifact,
            RunBudget(BudgetLimits(max_calls=1, max_cost_usd=1.0)),
            {"go": True, "reason": None},
        )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._execute_capability_preflight",
        complete_without_provider,
    )
    receipt = execute_stage(
        contract_path=CONTRACT_PATH,
        stage_id="capability-preflight",
        # --resume is the documented invocation and must also work on the
        # first clean launch, before any catalog receipt exists.
        resume=True,
        raw_root=tmp_path / "raw",
        repo_root=ROOT,
    )
    assert receipt["status"] == "complete-with-no-go"
    assert receipt["status_counts"] == {
        "capability-no-go": 1,
        "complete": 5,
    }
    assert len(catalog_calls) == 6
    assert all(len(model_ids) == 1 for model_ids in catalog_calls)
    assert set(dispatched) == {
        model_id
        for model_id in contract.models_for_stage("capability-preflight")
        if model_id != "opus48_sentinel"
    }
    assert receipt["go_models"] == dispatched
    ledger = json.loads((tmp_path / "raw" / "run_ledger.json").read_text())
    by_model = {
        row["spec"]["model_id"]: row["status"]
        for row in ledger["runs"].values()
        if row["spec"]["stage_id"] == "capability-preflight"
    }
    assert by_model["opus48_sentinel"] == "capability-no-go"
    assert set(by_model.values()) == {"complete", "capability-no-go"}


def test_preflight_provider_error_uses_runner_error_type_key() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles["gpt52_main"]

    class Result:
        validation_status = {"status": "pass"}
        summary = {"scientific_evidence": True}

        @staticmethod
        def stream(name: str):
            if name == "actions":
                return tuple(
                    {"decision": {"clipped": False}} for _ in range(12)
                )
            if name == "semantic_proposals":
                return ({"candidate_parse_status": "success"},)
            if name == "api_usage":
                return (
                    {
                        "error_type": "ProviderError",
                        "response_model": profile.served_model,
                        "attempts": 1,
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "cost_usd": 0.01,
                        },
                    },
                )
            raise KeyError(name)

    checks = _preflight_checks(Result(), profile)
    assert checks["no_provider_error"] is False


def test_scripted_a_to_d_matrix_is_complete_and_never_scientific(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_live_provider(*args, **kwargs):
        raise AssertionError("development matrix attempted a live provider")

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        no_live_provider,
    )
    result = run_development_fake_matrix(
        contract_path=CONTRACT_PATH,
        resume=False,
        raw_root=tmp_path,
    )
    assert result["status"] == "pass"
    assert result["registered_cells"] == 23
    assert result["status_counts"] == {"complete": 23}
    assert result["diagnostic_only"] is True
    assert result["scientific_evidence"] is False

    root = tmp_path / "development-fake"
    ledger = json.loads((root / "run_ledger.json").read_text())
    for row in ledger["runs"].values():
        artifact = row["artifact"]
        if isinstance(artifact, str) and artifact.endswith("manifest.json"):
            run_dir = Path(artifact).parent
            assert verify_manifest(run_dir).valid is True
            assert not (run_dir / "pilot_summary.json").exists()
    d_summaries = list(
        (root / "experiment-d" / "diagnostic_summaries").glob("*.json")
    )
    assert len(d_summaries) == 11
    assert all(
        json.loads(path.read_text())["scientific_evidence"] is False
        for path in d_summaries
    )


def _paid_fixture(contract) -> GitProvenance:
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


def _assert_terminal_summary(path: Path, *, scientific: bool) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION
    assert value["scientific_evidence"] is scientific
    assert value["integrity"]["canonicalization"] == "json-sort-keys-utf8-v1"
    assert len(value["integrity"]["content_sha256"]) == 64
    assert path.stat().st_mode & 0o222 == 0
    return value


def test_real_qref_offline_and_d_cells_point_to_sealed_terminal_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid_fixture(contract)
    raw = tmp_path / "raw"
    _utility_fixture(contract, raw)

    qref_spec = contract.expand(stage="q-ref-resolution")[0]
    qref_projection = RunProjection(
        run_id=qref_spec.run_id,
        stage_bucket=qref_spec.budget_bucket,
        cost_usd=0.0,
        completions=48,
        storage_bytes=20_000_000,
        basis={
            "method": "test-scripted",
            "prompt_tokens": 500_000,
            "completion_tokens": 100_000,
        },
    )
    qref_terminal, _, _ = _execute_q_ref(
        contract,
        qref_spec,
        raw_root=raw,
        paid=paid,
        projection=qref_projection,
    )
    qref_value = _assert_terminal_summary(qref_terminal, scientific=False)
    assert qref_value["diagnostic_only"] is True
    qref_manifest = (
        raw
        / "q-ref-resolution"
        / "runs"
        / qref_spec.run_id
        / "manifest.json"
    )
    assert verify_manifest(qref_manifest.parent).valid is True

    offline_spec = next(
        spec
        for spec in contract.expand(stage="experiment-c")
        if spec.execution_mode == "offline_candidate_admission"
    )
    offline_terminal = _offline_candidate_admission(
        contract,
        offline_spec,
        raw_root=raw,
        diagnostic=False,
        paid=paid,
    )
    offline_value = _assert_terminal_summary(
        offline_terminal,
        scientific=True,
    )
    reliability = offline_value["payload"]["metrics"]["rule_reliability"]
    assert reliability["unsupported_candidate_rejected"] is True
    assert reliability["false_rule_ever_active"] is False
    assert reliability["unverified_false_rule_ever_active"] is True
    assert reliability["same_candidate_content"] is True
    assert reliability["provider_calls"] == 0

    d_specs = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.environment_seed == contract.seeds["sets"]["main"][0]
    )
    run_ledger = PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    run_ledger.register(d_specs)
    budget_ledger = PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        lambda profile, **kwargs: MultiModelLLM(
            ScriptedDiagnosticProvider(),
            num_workers=4,
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._preflight_usage_caps",
        lambda *args, **kwargs: {
            "action": UsageRecord(10_000, 2_000, 0.0),
            "semantic": UsageRecord(10_000, 2_000, 0.0),
        },
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._d_group_projection",
        lambda _contract, representative, **kwargs: RunProjection(
            run_id=f"test-d-group-{representative.environment_seed}",
            stage_bucket=representative.budget_bucket,
            cost_usd=0.0,
            completions=400,
            storage_bytes=80_000_000,
            basis={
                "method": "test-scripted",
                "prompt_tokens": 4_000_000,
                "completion_tokens": 1_000_000,
            },
        ),
    )
    _execute_d_seed(
        contract,
        d_specs,
        raw_root=raw,
        paid=paid,
        diagnostic=False,
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )
    snapshot = run_ledger.snapshot()
    assert {
        snapshot["runs"][spec.run_id]["status"] for spec in d_specs
    } == {"complete"}
    for spec in d_specs:
        artifact = Path(snapshot["runs"][spec.run_id]["artifact"])
        value = _assert_terminal_summary(artifact, scientific=True)
        assert value["run_spec"] == spec.to_dict()
