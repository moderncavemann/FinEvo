from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT.parent / ".github" / "workflows" / "egrm-ci.yml"


def test_egrm_ci_is_independent_and_covers_release_gates():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "name: EGRM CI" in workflow
    assert '- "EGRM/**"' in workflow
    assert "- ubuntu-24.04" in workflow
    assert "- macos-14" in workflow
    assert "fetch-depth: 0" in workflow
    assert "Verify complete Git source provenance" in workflow
    assert "--source-repo-root ." in workflow
    assert "python -m pytest" in workflow
    assert "Run controlled fixture twice" in workflow
    assert "Expand non-dispatchable scientific design twice" in workflow
    assert "Compile tracked EGRM Python sources" in workflow
    assert "Scan tracked EGRM files for high-confidence secrets" in workflow
    assert "Check tracked-tree hygiene and clean checkout" in workflow


def test_egrm_ci_has_no_scientific_dispatch_or_provider_credentials():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "run_pilot.py" not in workflow
    assert "simulate_verified.py" not in workflow
    assert "secrets." not in workflow
    assert "OPENAI_API_KEY" not in workflow
    assert "OPENROUTER_API_KEY" not in workflow
    assert "GEMINI_API_KEY" not in workflow
