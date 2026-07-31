from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_pilot_v2111_parent_import as parent_fixtures
from tests import test_pilot_v211_gate as v211_fixtures
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v2111_parent_import as parent_import
from verified_memory.pilot_contract import PilotContract, load_pilot_contract
from verified_memory.pilot_v2111_gate import verify_v2111_gate_receipt
from verified_memory.pilot_v2111_parent_import import (
    V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
)
from verified_memory.pilot_v211_gate import (
    V211_PREFLIGHT_CHECKPOINT_RUN_SUFFIX,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_1.yaml"


def _contract():
    return load_pilot_contract(CONTRACT_PATH)


def _paid() -> orchestrator.GitProvenance:
    return orchestrator.GitProvenance(
        git_tag="pilot-v2.11.1-science",
        head_commit="2" * 40,
        tag_commit="2" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )


def _fake_preflight_artifact(
    contract,
    spec,
) -> dict:
    artifact = deepcopy(v211_fixtures._preflight_artifact(spec.model_id))
    artifact["ledger_run_id"] = spec.run_id
    artifact["checkpoint_run_id"] = (
        spec.run_id + V211_PREFLIGHT_CHECKPOINT_RUN_SUFFIX
    )
    artifact["run_spec_sha256"] = orchestrator.canonical_sha256(
        spec.to_dict()
    )

    checkpoint = artifact["checkpoint"]
    checkpoint["run_config"]["run_id"] = artifact["checkpoint_run_id"]
    checkpoint["run_config"]["pilot_contract_hash"] = (
        contract.canonical_hash
    )
    journal = checkpoint["provider_call_journal_binding"]
    journal["run_id"] = artifact["checkpoint_run_id"]
    journal["contract_hash"] = contract.canonical_hash
    journal["path_name"] = (
        artifact["checkpoint_run_id"] + "-provider-calls.json"
    )
    checkpoint["provider_call_journal_binding_hash"] = (
        orchestrator.canonical_sha256(journal)
    )
    checkpoint.pop("checkpoint_hash", None)
    checkpoint["checkpoint_hash"] = orchestrator.canonical_sha256(
        checkpoint
    )

    exactness = artifact["exactness"]
    exactness["checkpoint_hash"] = checkpoint["checkpoint_hash"]
    exactness["provider_call_journal_binding_hash"] = checkpoint[
        "provider_call_journal_binding_hash"
    ]
    exactness.pop("receipt_hash", None)
    exactness["receipt_hash"] = orchestrator.canonical_sha256(exactness)
    artifact["checkpoint_artifact_sha256"] = (
        orchestrator.canonical_sha256(checkpoint)
    )
    artifact["exactness_artifact_sha256"] = (
        orchestrator.canonical_sha256(exactness)
    )
    return artifact


def test_capability_authority_import_is_the_unique_preflight_source() -> None:
    contract = _contract()

    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        source = contract.expand(
            stage="capability-gate",
            model=model_id,
        )
        target = contract.expand(
            stage="long-context-preflight",
            model=model_id,
        )

        assert len(source) == len(target) == 1
        assert source[0].execution_mode == "capability_authority_import"
        assert target[0].execution_mode == "closed_loop_preflight"
        assert orchestrator._capability_source_stage(
            contract,
            target[0].stage_id,
            model_id,
        ) == source[0].stage_id


def test_v2111_long_context_projection_reserves_exactly_32_calls() -> None:
    contract = _contract()
    spec = contract.expand(
        stage="long-context-preflight",
        model="gpt52_main",
    )[0]

    assert orchestrator._max_call_projection(contract, spec) == (
        32,
        32 * 200_000,
        32 * 4_096,
    )
    projection = orchestrator.conservative_projection(contract, spec)

    assert projection.completions == 32
    assert projection.basis == {
        "method": "preflight-conservative-token-ceiling",
        "diagnostic": False,
        "run_call_limit": 32,
        "hosted_completion_cap_counted": True,
        "prompt_tokens": 32 * 200_000,
        "completion_tokens": 32 * 4_096,
    }


def test_v2111_capability_output_contracts_map_to_runtime_call_kinds() -> None:
    contract = _contract()
    call_kind_map = orchestrator._capability_projection_call_kind_map(
        contract
    )
    capability = {
        "rows": [
            {
                "served_model": "gpt-5.2-2025-12-11",
                "category": "utility-ranking",
                "output_contract_id": "actor-action",
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
            {
                "served_model": "gpt-5.2-2025-12-11",
                "category": "rule-proposal",
                "output_contract_id": "semantic-proposal",
                "usage": {"prompt_tokens": 13, "completion_tokens": 5},
            },
        ]
    }
    preflight = SimpleNamespace(stream=lambda _stream_id: ())

    rows = orchestrator._usage_projection_rows(
        capability,
        preflight,
        capability_call_kind_map=call_kind_map,
    )

    assert [row["call_kind"] for row in rows] == ["action", "semantic"]
    assert call_kind_map == {
        "actor-action": "action",
        "semantic-proposal": "semantic",
        "action": "action",
        "semantic": "semantic",
    }


def test_v2111_wrapper_closed_loop_semantics_recompute_to_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    source_spec = contract.expand(
        stage="capability-gate",
        model="gpt52_main",
    )[0]
    spec = contract.expand(
        stage="long-context-preflight",
        model="gpt52_main",
    )[0]
    raw_root = tmp_path / "raw"
    source_dir = (
        raw_root
        / source_spec.stage_id
        / "runs"
        / source_spec.run_id
    )
    run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
    source_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)

    wrapper = {
        "schema_version": V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
        "model_id": spec.model_id,
        "capability": {
            "capability_pass": True,
            "interface_pass": True,
        },
        "integrity": {"fixture": "provider-free"},
    }
    checks = {
        "all_actions_parseable": True,
        "proposal_outcomes_accounted_8_of_8": True,
        "provider_calls_accounted_32_of_32": True,
    }
    capability = {
        **wrapper,
        "preflight_go": True,
        "preflight_checks": checks,
    }
    projection_path = run_dir / "projection_p95.json"
    checkpoint_path = run_dir / "preflight_checkpoint.json"
    exactness_path = run_dir / "preflight_checkpoint_exactness.json"
    bootstrap_path = (
        source_dir / orchestrator.V2111_BOOTSTRAP_PROJECTION_FILENAME
    )
    gate = {
        "preflight_checks": checks,
        "go": True,
        "projection": str(projection_path),
        "preflight_manifest": None,
        "preflight_checkpoint": str(checkpoint_path),
        "preflight_checkpoint_exactness": str(exactness_path),
        "bootstrap_projection": str(bootstrap_path),
    }
    terminal_path = run_dir / "terminal_summary.json"
    orchestrator._atomic_json(source_dir / "capability.json", wrapper)
    orchestrator._atomic_json(run_dir / "capability.json", capability)
    orchestrator._atomic_json(run_dir / "gate_receipt.json", gate)
    orchestrator._atomic_json(checkpoint_path, {"fixture": True})
    orchestrator._atomic_json(
        terminal_path,
        {"schema_version": orchestrator.PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION},
    )

    monkeypatch.setattr(
        orchestrator,
        "_load_v2_terminal_summary",
        lambda *_args, **_kwargs: {
            "payload": {
                "capability": capability,
                "gate_evidence": gate,
            }
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v2111_parent_import_receipt",
        lambda *_args, **_kwargs: {"fixture": "verified"},
    )
    monkeypatch.setattr(
        orchestrator,
        "capability_wrappers_from_v2111_receipt",
        lambda _receipt: {spec.model_id: wrapper},
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_projection",
        lambda *_args, **_kwargs: ({"fixture": True}, projection_path),
    )
    monkeypatch.setattr(orchestrator, "PilotCheckpoint", lambda value: value)
    monkeypatch.setattr(
        orchestrator,
        "_CheckpointPreflightResult",
        lambda value: value,
    )
    monkeypatch.setattr(
        orchestrator,
        "_preflight_checks",
        lambda *_args, **_kwargs: checks,
    )

    assert orchestrator._v2_capability_semantic_go(
        contract,
        spec,
        {"status": "complete", "artifact": str(terminal_path)},
        raw_root=raw_root,
        paid=_paid(),
    ) is True


def test_v2111_stage_control_paths_bind_bootstrap_projection(
    tmp_path: Path,
) -> None:
    contract = _contract()
    spec = contract.expand(
        stage="capability-gate",
        model="gpt52_main",
    )[0]
    bootstrap = (
        tmp_path
        / spec.stage_id
        / "runs"
        / spec.run_id
        / orchestrator.V2111_BOOTSTRAP_PROJECTION_FILENAME
    )
    bootstrap.parent.mkdir(parents=True)
    orchestrator._atomic_json(bootstrap, {"fixture": "bootstrap"})

    paths = orchestrator._v2_stage_control_paths(
        contract,
        spec.stage_id,
        raw_root=tmp_path,
    )

    assert bootstrap in paths


def test_v2111_family_verifier_is_selected_over_v211(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    observed: list[str] = []
    expected = {"schema_version": "fixture", "go": False}

    monkeypatch.setattr(
        orchestrator,
        "verify_v2111_gate_receipt",
        lambda value, **_kwargs: observed.append("v2111") or dict(value),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v211_gate_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("V2.11 verifier selected for a V2.11.1 contract")
        ),
    )

    assert orchestrator._verify_v211_family_post_gate_receipt(
        contract,
        expected,
        paid=_paid(),
    ) == expected
    assert observed == ["v2111"]


def test_v2111_terminal_preflight_failures_publish_no_go_without_authority(
    tmp_path: Path,
) -> None:
    contract = _contract()
    raw_root = tmp_path / "raw"
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    failure = {
        "error_type": "VerifiedRunError",
        "message": "preflight stopped before provider dispatch",
    }
    for spec in contract.expand(stage="long-context-preflight"):
        ledger.finalize(
            spec.run_id,
            status="failed",
            artifact=None,
            failure=failure,
        )

    bindings, go_models = orchestrator._v2_stage_receipt_bindings(
        contract,
        "long-context-preflight",
        raw_root=raw_root,
        ledger=ledger,
        paid=_paid(),
    )

    assert go_models == ()
    assert bindings["source_files"] == []
    assert not (
        raw_root
        / "long-context-preflight"
        / orchestrator.PILOT_V211_POST_GATE_AUTHORITY_FILENAME
    ).exists()


def test_v2111_post_gate_status_inference_preserves_model_scoped_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "gpt52_main": "eligible",
        "gpt56_diagnostic": "interface-no-go",
    }

    def fake_build(**inputs):
        statuses = inputs["model_terminal_statuses"]
        if statuses != expected:
            raise orchestrator.PilotV2111GateError("status mismatch")
        return {
            "go": True,
            "denominator": {"eligible_model_ids": ["gpt52_main"]},
        }

    monkeypatch.setattr(
        orchestrator,
        "build_v2111_post_gate_authority",
        fake_build,
    )

    receipt, statuses = (
        orchestrator._build_v2111_post_gate_with_inferred_statuses({})
    )

    assert receipt["go"] is True
    assert receipt["denominator"]["eligible_model_ids"] == ["gpt52_main"]
    assert statuses == expected


def test_v2111_zero_provider_parent_capability_fake_preflight_postgate_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    repo_root = tmp_path / "child"
    raw_root = (
        repo_root / "experiment_results" / "pilot-v2.11.1" / "raw"
    )
    repo_root.mkdir()
    fake_audit = parent_fixtures._fake_audit(repo_root)
    for model_id, capability in fake_audit["capabilities"].items():
        fake_audit["manifest"]["capability_source"]["models"][model_id][
            "actual_usage"
        ] = deepcopy(capability["actual_usage"])
    monkeypatch.setattr(
        parent_import,
        "_audit_sources",
        lambda **_kwargs: fake_audit,
    )
    monkeypatch.setattr(
        orchestrator,
        "__file__",
        str(repo_root / "verified_memory" / "pilot_orchestrator.py"),
    )

    release_attestation = {
        "schema_version": (
            orchestrator.SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION
        ),
        "status": "pass",
        "head_commit": "2" * 40,
        "diagnostic_only": True,
    }
    release_attestation["attestation_sha256"] = (
        orchestrator.canonical_sha256(release_attestation)
    )
    paid = orchestrator.GitProvenance(
        git_tag="pilot-v2.11.1-science",
        head_commit="2" * 40,
        tag_commit="2" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={"diagnostic_only": True},
        release_attestation=release_attestation,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )
    monkeypatch.setattr(
        PilotContract,
        "validate_provenance",
        lambda self, git_commit, git_tag: {
            "diagnostic_only": True,
            "contract_canonical_sha256": self.canonical_hash,
            "resolved_git_commit": git_commit,
            "git_tag": git_tag,
        },
    )

    def forbidden_provider(*_args, **_kwargs):
        raise AssertionError("zero-provider integration constructed a provider")

    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        forbidden_provider,
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_live_provider_catalog",
        forbidden_provider,
    )

    def fake_bootstrap(
        _contract,
        spec,
        *,
        raw_root,
        **_kwargs,
    ):
        path = (
            raw_root
            / spec.stage_id
            / "runs"
            / spec.run_id
            / orchestrator.V2111_BOOTSTRAP_PROJECTION_FILENAME
        )
        value = {
            "schema_version": "zero-provider-bootstrap-fixture-v1",
            "model_id": spec.model_id,
            "provider_construction": False,
            "provider_calls": 0,
        }
        orchestrator._persist_exact_json(path, value)
        return value, path

    monkeypatch.setattr(
        orchestrator,
        "_persist_v2111_bootstrap_projection",
        fake_bootstrap,
    )
    parent_root = repo_root / "immutable-parent"

    parent_receipt = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="parent-import",
        resume=False,
        raw_root=raw_root,
        repo_root=repo_root,
        parent_repo_root=parent_root,
    )
    capability_receipt = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="capability-gate",
        resume=False,
        raw_root=raw_root,
        repo_root=repo_root,
        parent_repo_root=parent_root,
    )

    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    budget_ledger = orchestrator.PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(
            contract,
            repo_root=repo_root,
            parent_repo_root=parent_root,
        ),
    )

    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        spec = contract.expand(
            stage="long-context-preflight",
            model=model_id,
        )[0]
        artifact = _fake_preflight_artifact(contract, spec)
        run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
        orchestrator._persist_exact_json(
            run_dir / "preflight_checkpoint.json",
            artifact["checkpoint"],
        )
        orchestrator._persist_exact_json(
            run_dir / "preflight_checkpoint_exactness.json",
            artifact["exactness"],
        )
        preflight_checks = {
            "action_parse_success_24_of_24": True,
            "proposal_outcomes_accounted_8_of_8": True,
            "provider_calls_accounted_32_of_32": True,
        }
        orchestrator._persist_exact_json(
            run_dir / "gate_receipt.json",
            {
                "capability_pass": True,
                "preflight_checks": preflight_checks,
                "go": True,
            },
        )
        run_ledger.finalize(
            spec.run_id,
            status="complete",
            artifact=None,
        )
        projection = orchestrator.conservative_projection(contract, spec)
        budget_ledger.reserve(projection)
        storage_bytes = sum(
            path.stat().st_size
            for path in run_dir.rglob("*")
            if path.is_file()
        )
        budget_ledger.finalize(
            spec.run_id,
            status="complete",
            cost_usd=0.64,
            completions=32,
            storage_bytes=storage_bytes,
        )

    postgate_path, postgate = (
        orchestrator._persist_v2111_post_gate_authority(
            contract,
            raw_root=raw_root,
            paid=paid,
            budget_ledger=budget_ledger,
            run_ledger=run_ledger,
        )
    )
    verified = verify_v2111_gate_receipt(
        postgate,
        expected_contract_sha256=contract.canonical_hash,
        expected_git_commit=paid.head_commit,
    )

    assert parent_receipt["status"] == "complete"
    assert parent_receipt["registered_run_count"] == 1
    assert capability_receipt["status"] == "complete"
    assert capability_receipt["go_models"] == [
        "gpt52_main",
        "gpt56_diagnostic",
    ]
    assert postgate_path == (
        raw_root
        / "long-context-preflight"
        / orchestrator.PILOT_V211_POST_GATE_AUTHORITY_FILENAME
    )
    assert verified == postgate
    assert postgate["go"] is True
    assert postgate["provider_construction_during_authority"] is False
    assert postgate["provider_calls_during_authority"] == 0
    assert postgate["denominator"]["eligible_model_ids"] == [
        "gpt52_main",
        "gpt56_diagnostic",
    ]
    assert postgate["denominator"]["fresh_preflight_calls"] == 64
    assert (
        postgate["denominator"]["registered_fresh_full_matrix_calls"]
        == 5_880
    )
    assert postgate["denominator"]["cumulative_full_matrix_calls"] == 6_756
    assert all(
        decision["sample_counts"] == {"action": 48, "semantic": 14}
        for decision in postgate["model_decisions"].values()
    )
    assert postgate["projection"]["go"] is True
    assert (
        postgate["projection"]["cumulative_full_matrix"][
            "projected_cost_usd"
        ]
        <= 500.0
    )
    assert orchestrator._v2_control_gate_ok(
        contract,
        "long-context-preflight",
        raw_root=raw_root,
        paid=paid,
    ) is True
