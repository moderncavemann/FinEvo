from pathlib import Path

from llm_providers import MultiModelLLM
from verified_memory.artifacts import verify_manifest
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.replay_experiment import (
    build_paired_snapshot,
    run_paired_replay,
    summarize_paired_replay,
    write_paired_replay_artifacts,
)
from verified_memory.runner import VerifiedRunConfig, run_verified_experiment
from verified_memory.runner_artifacts import (
    load_verified_run_artifacts,
    write_verified_run_artifacts,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


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
    replay_budget = RunBudget(BudgetLimits(max_calls=5, max_cost_usd=0.01))
    replay = run_paired_replay(
        snapshot,
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=1),
        budget=replay_budget,
    )
    summary = summarize_paired_replay(replay)
    assert summary["integrity_verified"] is True
    assert summary["memory_sensitive"] is False
    assert len(replay.records) == 5
    assert len({row.base_prompt_hash for row in replay.records}) == 1
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
