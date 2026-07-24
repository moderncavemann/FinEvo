from dataclasses import replace
import copy
import json
from pathlib import Path
import hashlib

import pytest

from llm_providers import MultiModelLLM
from verified_memory.artifacts import verify_manifest
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.replay import PairedReplayResult, ReplayIntegrityError
from verified_memory.replay_experiment import (
    build_paired_snapshot,
    run_paired_replay,
    summarize_paired_replay,
    write_paired_replay_artifacts,
)
from verified_memory.prompts import compose_decision_prompt
from verified_memory.runner import VerifiedRunConfig, run_verified_experiment
from verified_memory.runner_artifacts import (
    load_verified_run_artifacts,
    write_verified_run_artifacts,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


class _ProfileMetadataScriptedProvider(ScriptedDiagnosticProvider):
    def get_structured_completion(self, messages, **kwargs):
        return replace(
            super().get_structured_completion(messages, **kwargs),
            request_profile_id="fixture-profile",
            request_price_snapshot_source="fixture-price-snapshot",
            request_price_snapshot_captured_at="2026-07-24T00:00:00Z",
        )


def test_sealed_run_to_paired_replay_preserves_integrity(tmp_path: Path) -> None:
    source = run_verified_experiment(
        VerifiedRunConfig(run_id="replay-source", episode_length=6),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=20, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )
    source_dir = tmp_path / "source"
    write_verified_run_artifacts(
        source_dir,
        source,
        provenance={"purpose": "test"},
        git_commit="test",
        git_dirty=True,
    )
    loaded = load_verified_run_artifacts(source_dir)
    snapshot = build_paired_snapshot(
        loaded,
        decision_t=5,
        agent_id=0,
        provider="diagnostic",
        model="scripted-v1",
    )
    injected = snapshot.bundle("injected-rule").text
    assert "confidence 99%" in injected
    assert "unverified" not in injected
    assert "false-rule" not in injected
    matched = snapshot.bundle("matched").text
    shuffled = snapshot.bundle("shuffled").text
    marker = " Verified active rules:"
    matched_episodes, matched_rules = matched.split(marker, 1)
    shuffled_episodes, shuffled_rules = shuffled.split(marker, 1)
    assert matched_rules == shuffled_rules
    assert matched_episodes != shuffled_episodes
    replay_budget = RunBudget(BudgetLimits(max_calls=5, max_cost_usd=0.01))
    replay = run_paired_replay(
        snapshot,
        llm=MultiModelLLM(_ProfileMetadataScriptedProvider(), num_workers=1),
        budget=replay_budget,
    )
    summary = summarize_paired_replay(replay)
    assert summary["integrity_verified"] is True
    assert summary["memory_sensitive"] is False
    assert len(replay.records) == 5
    assert len({row.base_prompt_hash for row in replay.records}) == 1
    assert all(
        row.to_dict()["provider_metadata"]["schema_version"]
        == "verified-simulation-runner-v3"
        for row in replay.records
    )
    assert all(
        row.to_dict()["provider_metadata"]["decoding_seed_applied"]
        == snapshot.decoding_seed
        for row in replay.records
    )
    assert all(
        row.to_dict()["provider_metadata"]["request_price_snapshot_source"]
        == "fixture-price-snapshot"
        and row.to_dict()["provider_metadata"][
            "request_price_snapshot_captured_at"
        ]
        == "2026-07-24T00:00:00Z"
        for row in replay.records
    )
    replay_dir = tmp_path / "replay"
    write_paired_replay_artifacts(
        replay_dir,
        replay,
        budget_snapshot=replay_budget.snapshot().to_dict(),
        provenance={"purpose": "test replay"},
        git_commit="test",
        git_dirty=True,
    )
    assert verify_manifest(replay_dir).valid


def test_legacy_v1_sealed_source_reconstructs_exact_matched_prompt() -> None:
    source = load_verified_run_artifacts(
        ROOT / "artifacts" / "verified_runs" / "g4b-api-gpt52-s11"
    )
    source_snapshot = next(
        row
        for row in source.stream("decision_snapshots")
        if row["decision_t"] == 5 and row["agent_id"] == 0
    )
    snapshot = build_paired_snapshot(
        source,
        decision_t=5,
        agent_id=0,
        provider="diagnostic",
        model="scripted-v1",
    )

    assert snapshot.prompt_schema_version == "verified-decision-prompt-v1"
    assert snapshot.source_full_prompt_hash == source_snapshot["full_prompt_hash"]
    assert (
        hashlib.sha256(snapshot.build_prompt("matched").encode("utf-8")).hexdigest()
        == source_snapshot["full_prompt_hash"]
    )


@pytest.mark.parametrize(
    ("context_mode", "context_to_prompt"),
    [
        ("no-context", False),
        ("prompt-only", True),
        ("retrieval-only", False),
        ("full", True),
    ],
)
def test_all_context_routes_are_hash_bound_in_paired_replay(
    tmp_path: Path, context_mode: str, context_to_prompt: bool
) -> None:
    source = run_verified_experiment(
        VerifiedRunConfig(
            run_id=f"{context_mode}-context-source",
            episode_length=6,
            context_mode=context_mode,
        ),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=20, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )
    source_dir = tmp_path / f"{context_mode}-source"
    write_verified_run_artifacts(
        source_dir,
        source,
        provenance={"purpose": "test"},
        git_commit="test",
        git_dirty=True,
    )
    loaded = load_verified_run_artifacts(source_dir)

    source_snapshot = next(
        row
        for row in loaded.stream("decision_snapshots")
        if row["decision_t"] == 5 and row["agent_id"] == 0
    )
    snapshot = build_paired_snapshot(
        loaded,
        decision_t=5,
        agent_id=0,
        provider="diagnostic",
        model="scripted-v1",
    )
    protected_context = source_snapshot["protected_context_text"]
    assert bool(protected_context) is context_to_prompt
    if context_to_prompt:
        assert f"Causal context summary: {protected_context}" in snapshot.base_prompt
        assert all(
            protected_context in snapshot.build_prompt(treatment)
            for treatment in (
                "matched",
                "no-memory",
                "shuffled",
                "wrong-context",
                "injected-rule",
            )
        )
    else:
        assert "Causal context summary:" not in snapshot.base_prompt
    if protected_context:
        assert all(
            protected_context not in snapshot.bundle(treatment).text
            for treatment in (
                "matched",
                "no-memory",
                "shuffled",
                "wrong-context",
                "injected-rule",
            )
        )
    assert snapshot.source_full_prompt_hash == source_snapshot["full_prompt_hash"]
    assert (
        hashlib.sha256(snapshot.build_prompt("matched").encode("utf-8")).hexdigest()
        == source_snapshot["full_prompt_hash"]
    )

    replay = run_paired_replay(
        snapshot,
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=1),
        budget=RunBudget(BudgetLimits(max_calls=5, max_cost_usd=0.01)),
    )
    assert replay.records[0].prompt_hash == source_snapshot["full_prompt_hash"]


@pytest.mark.parametrize(
    "context_mode",
    ["no-context", "retrieval-only", "prompt-only", "full"],
)
def test_snapshot_builder_rejects_forged_context_marker_for_every_route(
    context_mode: str,
) -> None:
    source = run_verified_experiment(
        VerifiedRunConfig(
            run_id=f"forged-{context_mode}",
            episode_length=6,
            context_mode=context_mode,
        ),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=20, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )
    snapshots = list(source.stream("decision_snapshots"))
    index = next(
        index
        for index, row in enumerate(snapshots)
        if row["decision_t"] == 5 and row["agent_id"] == 0
    )
    target = dict(snapshots[index])
    forged_base = target["base_prompt"] + "\nCausal context summary: FORGED."
    prompt = compose_decision_prompt(forged_base, target["memory_text"])
    target.update(
        {
            "base_prompt": prompt.base_prompt,
            "base_prompt_hash": prompt.base_prompt_hash,
            "full_prompt_hash": prompt.full_prompt_hash,
        }
    )
    snapshots[index] = target
    records = {**source.records, "decision_snapshots": tuple(snapshots)}
    forged = replace(source, records=records)

    expected = (
        "non-prompt context route"
        if context_mode in {"no-context", "retrieval-only"}
        else "exactly the protected context summary once"
    )
    with pytest.raises(ValueError, match=expected):
        build_paired_snapshot(
            forged,
            decision_t=5,
            agent_id=0,
            provider="diagnostic",
            model="scripted-v1",
        )


def test_replay_summary_rejects_forged_provider_seed_or_served_model() -> None:
    source = run_verified_experiment(
        VerifiedRunConfig(run_id="metadata-source", episode_length=6),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=20, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )
    snapshot = build_paired_snapshot(
        source,
        decision_t=5,
        agent_id=0,
        provider="diagnostic",
        model="scripted-v1",
    )
    replay = run_paired_replay(
        snapshot,
        llm=MultiModelLLM(_ProfileMetadataScriptedProvider(), num_workers=1),
        budget=RunBudget(BudgetLimits(max_calls=5, max_cost_usd=0.01)),
    )
    original_records = list(replay.records)
    records = list(original_records)
    metadata = json.loads(original_records[2].provider_metadata_json)
    metadata.update(
        {"decoding_seed_applied": 999, "response_model": "forged-served-model"}
    )
    records[2] = replace(
        records[2],
        provider_metadata_json=json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    forged = PairedReplayResult(snapshot=snapshot, records=tuple(records))

    with pytest.raises(ReplayIntegrityError, match="protected decoding seed"):
        summarize_paired_replay(forged)

    records = list(original_records)
    metadata = json.loads(original_records[2].provider_metadata_json)
    metadata["request_price_snapshot_source"] = "forged-price-snapshot"
    records[2] = replace(
        records[2],
        provider_metadata_json=json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    forged = PairedReplayResult(snapshot=snapshot, records=tuple(records))
    with pytest.raises(ReplayIntegrityError, match="multiple price snapshots"):
        summarize_paired_replay(forged)


def test_replay_writer_rejects_budget_not_bound_to_five_treatments(
    tmp_path: Path,
) -> None:
    source = run_verified_experiment(
        VerifiedRunConfig(run_id="budget-source", episode_length=6),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=20, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )
    snapshot = build_paired_snapshot(
        source,
        decision_t=5,
        agent_id=0,
        provider="diagnostic",
        model="scripted-v1",
    )
    budget = RunBudget(BudgetLimits(max_calls=5, max_cost_usd=0.01))
    replay = run_paired_replay(
        snapshot,
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=1),
        budget=budget,
    )
    forged_budget = copy.deepcopy(budget.snapshot().to_dict())
    forged_budget["completed_calls"] = 4

    with pytest.raises(ReplayIntegrityError, match="completed_calls"):
        write_paired_replay_artifacts(
            tmp_path / "forged-budget-replay",
            replay,
            budget_snapshot=forged_budget,
            provenance={"purpose": "adversarial test"},
            git_commit="test",
            git_dirty=True,
        )
