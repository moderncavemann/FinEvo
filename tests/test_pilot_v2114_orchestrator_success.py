from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_budget import ParentBudgetDebit
from verified_memory.pilot_contract import PilotContract, load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_4.yaml"
SOURCE_MANIFEST_PATH = ROOT / "experiments" / "pilot_v2_11_4_source_manifest.json"
MODELS = {"gpt52_main", "gpt56_diagnostic"}


def _contract() -> PilotContract:
    return load_pilot_contract(CONTRACT_PATH)


def _paid() -> orchestrator.GitProvenance:
    return orchestrator.GitProvenance(
        git_tag="pilot-v2.11.4-science",
        head_commit="c" * 40,
        tag_commit="c" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )


def _ledgers(
    contract: PilotContract,
    raw_root: Path,
) -> tuple[orchestrator.PilotRunLedger, orchestrator.PilotBudgetLedger]:
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    boundary = contract.v2114_forward_boundary
    assert boundary is not None
    budget = orchestrator.PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=ParentBudgetDebit.from_dict(boundary["parent_budget_debit"]),
    )
    return ledger, budget


def _source_wrappers() -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source = manifest["source_gate_binding"]
    runtime_by_model = {
        "gpt52_main": "openai/gpt-5.2-2025-12-11",
        "gpt56_diagnostic": "openai/gpt-5.6-sol",
    }
    wrappers: dict[str, dict[str, Any]] = {}
    enriched: dict[str, dict[str, Any]] = {}
    authority_fields = {
        "source_authority_receipt_path": source["receipt_path"],
        "source_authority_receipt_file_sha256": source["receipt_file_sha256"],
        "source_authority_receipt_content_sha256": source[
            "receipt_content_sha256"
        ],
        "source_release_commit": source["git_commit"],
    }
    for model_id, runtime_model in runtime_by_model.items():
        reservations = deepcopy(source["reservations"][runtime_model])
        wrappers[model_id] = {
            "source_gate_receipt": {
                "path": source["receipt_path"],
                "file_sha256": source["receipt_file_sha256"],
                "content_sha256": source["receipt_content_sha256"],
                "git_commit": source["git_commit"],
            },
            "reservations": reservations,
        }
        resealed = deepcopy(reservations)
        for call_kind in ("action", "semantic"):
            resealed[call_kind]["authority"].update(authority_fields)
        enriched[model_id] = {runtime_model: resealed}
    return wrappers, enriched


def _forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("V2.11.4 operational reseal constructed a provider")


def test_v2114_reseal_normalization_accepts_only_four_receipt_fields() -> None:
    wrappers, enriched = _source_wrappers()
    runtime = "openai/gpt-5.2-2025-12-11"

    assert orchestrator._resealed_reservations_match_source_wrapper(
        source_wrapper=wrappers["gpt52_main"],
        resealed_reservations=enriched["gpt52_main"][runtime],
    )

    drifted = deepcopy(enriched["gpt52_main"][runtime])
    drifted["action"]["authority"]["unregistered_normalization"] = True
    assert not orchestrator._resealed_reservations_match_source_wrapper(
        source_wrapper=wrappers["gpt52_main"],
        resealed_reservations=drifted,
    )

    frozen_v2113 = inspect.getsource(
        orchestrator._execute_v2113_preflight_authority_import_stage_impl
    )
    normalized_v2114 = inspect.getsource(
        orchestrator._execute_v2114_preflight_authority_import_stage_impl
    )
    assert "_resealed_reservations_match_source_wrapper" not in frozen_v2113
    assert 'wrapper.get("reservations")' in frozen_v2113
    assert "_resealed_reservations_match_source_wrapper" in normalized_v2114


def test_v2114_preflight_reseal_success_is_zero_provider_and_keeps_science_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    paid = _paid()
    raw_root = tmp_path / "raw"
    ledger, budget = _ledgers(contract, raw_root)
    wrappers, enriched = _source_wrappers()
    provider_calls: list[str] = []

    monkeypatch.setattr(orchestrator, "_assert_prerequisites", lambda *_a, **_k: {})
    monkeypatch.setattr(
        orchestrator,
        "verify_v2114_parent_import_receipt",
        lambda *_a, **_k: {"receipt": "verified"},
    )
    monkeypatch.setattr(
        orchestrator,
        "preflight_wrappers_from_v2114_receipt",
        lambda *_a, **_k: wrappers,
    )

    paths_by_model: dict[str, tuple[Path, Path]] = {}

    def persist_reseal(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        model_id = str(kwargs["model_id"])
        model_root = raw_root / "resealed" / model_id
        authority_path = model_root / "authority.json"
        projection_path = model_root / "projection.json"
        orchestrator._atomic_json(authority_path, {"model_id": model_id})
        orchestrator._atomic_json(projection_path, {"model_id": model_id})
        paths_by_model[model_id] = (authority_path, projection_path)
        return {
            "receipt": str(authority_path),
            "projection": str(projection_path),
            "gate_receipt": {"model_id": model_id},
            "receipt_content_sha256": "a" * 64,
            "projection_content_sha256": "b" * 64,
            "provider_construction_during_reseal": False,
            "provider_calls_during_reseal": 0,
            "hosted_provider_calls_during_reseal": 0,
            "scientific_evidence": False,
        }

    monkeypatch.setattr(
        orchestrator,
        "persist_v2114_resealed_observed_p95_authority",
        persist_reseal,
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_v2114_preflight_import_gate",
        lambda value, **_kwargs: {"go": True, **value},
    )

    def authority_binding(path: Path, **_kwargs: Any) -> dict[str, Any]:
        model_id = next(
            model
            for model, (authority_path, _projection_path) in paths_by_model.items()
            if authority_path == path
        )
        return {"reservations": deepcopy(enriched[model_id])}

    def projection_binding(path: Path, **_kwargs: Any) -> dict[str, Any]:
        model_id = next(
            model
            for model, (_authority_path, projection_path) in paths_by_model.items()
            if projection_path == path
        )
        return {
            "reservations": deepcopy(enriched[model_id]),
            "payload": {"bindings": {"contract_sha256": contract.canonical_hash}},
        }

    monkeypatch.setattr(
        orchestrator,
        "verified_v2114_observed_p95_authority_binding",
        authority_binding,
    )
    monkeypatch.setattr(
        orchestrator,
        "verified_v2114_observed_p95_projection_binding",
        projection_binding,
    )

    def write_terminal(path: Path, **kwargs: Any) -> None:
        orchestrator._atomic_json(path, {"payload": kwargs["payload"]})

    monkeypatch.setattr(orchestrator, "write_terminal_summary", write_terminal)

    def persist_post_gate(**kwargs: Any) -> tuple[Path, dict[str, Any]]:
        path = raw_root / "long-context-preflight" / "post_gate_authority.json"
        receipt = {
            "receipt_sha256": "e" * 64,
            "denominator": {"eligible_model_ids": sorted(MODELS)},
            "bindings": {"ledger_event_chain_head": kwargs["ledger_event_chain_head"]},
            "go": True,
        }
        orchestrator._atomic_json(path, receipt)
        return path, receipt

    monkeypatch.setattr(
        orchestrator, "persist_v2114_post_gate_authority", persist_post_gate
    )
    monkeypatch.setattr(orchestrator, "verify_v2114_gate_receipt", lambda *_a, **_k: {})

    def write_stage_receipt(
        _contract: PilotContract,
        stage_id: str,
        *,
        raw_root: Path,
        status: str,
        go_models: list[str],
        artifacts: Mapping[str, Any],
        **_kwargs: Any,
    ) -> Path:
        path = raw_root / stage_id / "stage_receipt.json"
        orchestrator._atomic_json(
            path,
            {
                "status": status,
                "go_models": go_models,
                "artifacts": dict(artifacts),
            },
        )
        return path

    monkeypatch.setattr(orchestrator, "_write_stage_receipt", write_stage_receipt)
    for name in (
        "_provider_for_profile",
        "create_llm_provider",
        "validate_live_provider_catalog",
    ):
        monkeypatch.setattr(
            orchestrator,
            name,
            lambda *_a, _name=name, **_k: provider_calls.append(_name)
            or _forbidden_provider(),
        )

    result = orchestrator._execute_v2114_preflight_authority_import_stage(
        contract,
        contract.expand(stage="long-context-preflight"),
        raw_root=raw_root,
        repo_root=ROOT,
        paid=paid,
        run_ledger=ledger,
        budget_ledger=budget,
    )

    assert result["status"] == "complete"
    assert set(result["go_models"]) == MODELS
    assert provider_calls == []
    preflight_specs = contract.expand(stage="long-context-preflight")
    assert all(ledger.status(spec.run_id) == "complete" for spec in preflight_specs)
    budget_rows = budget.snapshot()["runs"]
    assert all(
        budget_rows[spec.run_id]["actual"]["completions"] == 0
        and budget_rows[spec.run_id]["actual"]["cost_usd"] == 0.0
        for spec in preflight_specs
    )
    scientific_specs = tuple(
        spec
        for stage_id in orchestrator._scientific_stage_ids(contract)
        for spec in contract.expand(stage=stage_id)
    )
    assert len(scientific_specs) == 131
    assert all(ledger.status(spec.run_id) == "scheduled" for spec in scientific_specs)
