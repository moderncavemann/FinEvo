from pathlib import Path

import pytest

from egrm.design import expand_scientific_design


ROOT = Path(__file__).resolve().parents[1]


def test_scientific_design_expands_to_exact_unique_cells():
    inventory = expand_scientific_design(
        ROOT / "configs" / "scientific_contract_v1.json"
    )
    assert inventory["registered_cells"] == 105
    assert inventory["stage_counts"] == {
        "E0-known-truth-ruleswitch": 48,
        "E1-fresh-closed-loop": 25,
        "E2-checkpoint-continuation": 20,
        "E3-second-model-diagnostic": 12,
    }
    assert len({cell["cell_id"] for cell in inventory["cells"]}) == 105
    assert inventory["dispatchable"] is False
    assert inventory["scientific_evidence"] is False
    assert {cell["arm_implementation_status"] for cell in inventory["cells"]} == {
        "implemented",
        "planned",
        "manifest_only",
    }


def test_implementation_fixture_is_not_a_scientific_design():
    with pytest.raises(ValueError, match="design drafts only"):
        expand_scientific_design(ROOT / "configs" / "controlled_benchmark_v1.json")
