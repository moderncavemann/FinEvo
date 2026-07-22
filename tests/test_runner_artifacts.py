import json
from pathlib import Path

from llm_providers import MultiModelLLM
from verified_memory.artifacts import verify_manifest
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.runner import VerifiedRunConfig, run_verified_experiment
from verified_memory.runner_artifacts import write_verified_run_artifacts
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_result_is_sealed_and_reverified(tmp_path: Path) -> None:
    result = run_verified_experiment(
        VerifiedRunConfig(run_id="artifact-diagnostic", episode_length=5),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=20, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )
    run_dir = tmp_path / "run"
    manifest_path = write_verified_run_artifacts(
        run_dir,
        result,
        provenance={"purpose": "unit diagnostic", "scientific_evidence": False},
        git_commit="test-commit",
        git_dirty=True,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["result"]["complete"] is True
    assert manifest["validation_status"]["status"] == "pass"
    assert verify_manifest(run_dir).valid is True
    assert (run_dir / "streams/actions.jsonl").read_text().count("\n") == 10
