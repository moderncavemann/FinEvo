from pathlib import Path

from egrm.benchmark import run_controlled_fixture


ROOT = Path(__file__).resolve().parents[1]


def test_controlled_fixture_reproduces_admission_activation_and_retirement():
    result = run_controlled_fixture(ROOT / "configs" / "controlled_benchmark_v1.json")
    assert result["fixture_passed"] is True
    assert result["scientific_evidence"] is False
    assert result["observed"]["supported_lifecycle"] == [
        "provisional",
        "active",
        "active",
        "retired",
    ]
    assert result["verified_policy_metrics"]["rule_admission_precision"] == 1.0
    assert result["verified_policy_metrics"]["unsupported_ever_active_count"] == 0
    assert result["unverified_policy_metrics"]["unsupported_ever_active_count"] == 1
    assert result["verified_policy_metrics"]["provenance_coverage"] == 1.0


def test_fixture_preserves_rejected_candidate_in_denominator():
    result = run_controlled_fixture(ROOT / "configs" / "controlled_benchmark_v1.json")
    assert result["verified_policy_metrics"]["candidate_count"] == 2
    assert result["verified_policy_metrics"]["admitted_count"] == 1
    assert sorted(rule["status"] for rule in result["verified_rules"]) == [
        "rejected",
        "retired",
    ]
