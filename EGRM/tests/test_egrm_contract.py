import copy
from pathlib import Path

import pytest

from egrm.contracts import content_hash, load_contract, validate_contract


ROOT = Path(__file__).resolve().parents[1]


def test_both_not_run_contracts_validate_and_hash_deterministically():
    for name in ("controlled_benchmark_v1.json", "scientific_contract_v1.json"):
        contract, first = load_contract(ROOT / "configs" / name)
        _, second = load_contract(ROOT / "configs" / name)
        assert first == second == content_hash(contract)


def test_contract_rejects_denominator_drift():
    contract, _ = load_contract(ROOT / "configs" / "scientific_contract_v1.json")
    altered = copy.deepcopy(contract)
    altered["denominator"]["failed_seed_replacement"] = True
    with pytest.raises(ValueError, match="must not be replaced"):
        validate_contract(altered)


def test_fixture_cannot_be_relabeled_as_scientific_evidence():
    contract, _ = load_contract(ROOT / "configs" / "controlled_benchmark_v1.json")
    altered = copy.deepcopy(contract)
    altered["scientific_evidence"] = True
    with pytest.raises(ValueError, match="cannot be scientific evidence"):
        validate_contract(altered)


def test_design_draft_cannot_be_relabeled_as_scientific_evidence():
    contract, _ = load_contract(ROOT / "configs" / "scientific_contract_v1.json")
    altered = copy.deepcopy(contract)
    altered["scientific_evidence"] = True
    with pytest.raises(ValueError, match="cannot be scientific evidence"):
        validate_contract(altered)


def test_controlled_fixture_rejects_arm_and_seed_drift():
    contract, _ = load_contract(ROOT / "configs" / "controlled_benchmark_v1.json")
    changed_arm = copy.deepcopy(contract)
    changed_arm["arms"][0]["name"] = "looks-better"
    with pytest.raises(ValueError, match="arms are frozen"):
        validate_contract(changed_arm)

    changed_seed = copy.deepcopy(contract)
    changed_seed["randomization"]["seeds"].append(20260803)
    with pytest.raises(ValueError, match="exactly one seed"):
        validate_contract(changed_seed)


def test_scientific_stage_count_is_derived_from_units_and_arms():
    contract, _ = load_contract(ROOT / "configs" / "scientific_contract_v1.json")
    altered = copy.deepcopy(contract)
    altered["fixture"]["stages"][0]["registered_cells"] = 49
    altered["denominator"]["registered_cells"] = 106
    with pytest.raises(ValueError, match="unit-by-arm product"):
        validate_contract(altered)
