from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_3,
    PILOT_CONTRACT_ID_V2_4,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_4,
    PILOT_CONTRACT_TAG_V2_4,
    PILOT_CONTRACT_V2_3_CANONICAL_SHA256,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "experiments" / "pilot_v2_3.yaml"
OVERLAY_PATH = ROOT / "experiments" / "pilot_v2_4_overlay.yaml"
FULL_PATH = ROOT / "experiments" / "pilot_v2_4.yaml"
PARENT_SOURCE_PATH = (
    ROOT / "experiments" / "pilot_v2_4_parent_source_manifest.json"
)


def _overlay_document() -> dict[str, Any]:
    return json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))


def _write_resealed_overlay(
    tmp_path: Path,
    value: dict[str, Any],
) -> Path:
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)
    (tmp_path / "pilot_v2_3.yaml").write_text(
        BASE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "pilot_v2_4_parent_source_manifest.json").write_text(
        PARENT_SOURCE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    path = tmp_path / "pilot_v2_4_overlay.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_v2_4_draft_overlay_and_expanded_contract_match() -> None:
    source = _overlay_document()
    overlay = load_pilot_contract(OVERLAY_PATH)
    full = load_pilot_contract(FULL_PATH)

    assert source["schema_version"] == (
        PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_4
    )
    assert source["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(source)
    )
    assert overlay.contract_id == full.contract_id == PILOT_CONTRACT_ID_V2_4
    assert overlay.status == full.status == "draft"
    assert overlay.to_dict() == full.to_dict()
    assert overlay.canonical_hash == full.canonical_hash
    assert overlay.declared_sha256 == overlay.canonical_hash
    assert overlay.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_4
    assert overlay.release_requirements is not None
    assert overlay.release_requirements.tag == PILOT_CONTRACT_TAG_V2_4
    assert all(
        value is None
        for value in overlay.release_requirements.expected_ci.values()
    )
    with pytest.raises(PilotContractError, match="draft contract"):
        overlay.validate_provenance("1" * 40, PILOT_CONTRACT_TAG_V2_4)


def test_v2_4_has_exact_local_first_211_cell_denominator() -> None:
    contract = load_pilot_contract(FULL_PATH)
    specs = contract.expand()
    counts = Counter(spec.stage_id for spec in specs)

    assert contract.stage_ids == (
        "parent-import",
        "q-ref-resolution",
        "stage0-calibration",
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    )
    assert counts == {
        "parent-import": 1,
        "q-ref-resolution": 1,
        "stage0-calibration": 14,
        "local-experiment-c": 25,
        "local-experiment-a": 20,
        "local-experiment-d": 35,
        "local-experiment-b": 25,
        "experiment-c": 25,
        "experiment-a": 20,
        "experiment-d": 30,
        "experiment-b": 15,
    }
    assert len(specs) == 211
    assert sum(
        count
        for stage_id, count in counts.items()
        if stage_id not in {"parent-import", "q-ref-resolution"}
    ) == 209
    assert "capability-gate" not in counts
    assert "closed-loop-preflight" not in counts
    assert set(contract.provider_profiles) == {
        "gpt52_main",
        "llama33_local_controlled",
        "qref_scripted",
    }
    assert {
        spec.model_id
        for spec in specs
        if spec.stage_id.startswith("local-experiment")
        or spec.stage_id == "stage0-calibration"
    } == {"llama33_local_controlled"}
    assert {
        spec.model_id
        for spec in specs
        if spec.stage_id in {
            "experiment-c",
            "experiment-a",
            "experiment-d",
            "experiment-b",
        }
    } == {"gpt52_main"}
    parent_spec = next(
        spec for spec in specs if spec.stage_id == "parent-import"
    )
    assert parent_spec.execution_mode == "parent_authority_import"
    assert parent_spec.model_id == "qref_scripted"


def test_v2_4_parent_authority_and_budget_are_exactly_bound() -> None:
    contract = load_pilot_contract(FULL_PATH)
    amendment = contract.matrix_amendment
    assert amendment is not None

    parent = amendment["parent"]
    assert parent["contract_id"] == PILOT_CONTRACT_ID_V2_3
    assert parent["contract_sha256"] == PILOT_CONTRACT_V2_3_CANONICAL_SHA256
    assert parent["release_tag"] == "pilot-v2.3-science"
    assert parent["release_commit"] == (
        "ab32e3c9dcf581a40f3093652e144b56f853c782"
    )
    assert parent["registered_cells"] == parent["terminal_cells"] == 174
    assert parent["terminal_status"] == "complete-with-no-go"

    prospective = amendment["prospective_registration"]
    assert prospective["outcome_blind"] is True
    assert prospective["scientific_outcomes_observed_before_amendment"] is False
    assert prospective["scientific_effect_outcomes_observed"] is False
    assert prospective["parent_science_outputs_inspected"] is False
    assert prospective["parent_artifacts_modified"] is False
    assert prospective["parent_runs_resumed"] is False

    carry = amendment["budget_carry_forward"]
    assert carry["cost_usd"] == 3.212770875
    assert carry["hosted_completions"] == 184
    assert carry["storage_bytes"] == 4_196_087
    assert carry["debit_before_new_dispatch"] is True
    assert carry["parent_import_cell_additional_cost_usd"] == 0.0
    assert carry["parent_import_cell_additional_hosted_completions"] == 0

    source = amendment["parent_source_manifest"]
    assert source == {
        "path": "experiments/pilot_v2_4_parent_source_manifest.json",
        "schema_version": "finevo-pilot-v2.4-parent-source-manifest-v1",
        "file_sha256": (
            "d6a867cd7add43818127af7778a447d579ac1ab31ed6d053bcd29d69b3cf0f33"
        ),
        "content_sha256": (
            "7ae427fe6eac5aa6e04eddd3efa9e63405e128c782013ed3f67c35808be3cec5"
        ),
    }

    authority = amendment["parent_authority_import"]
    assert authority["provider_calls"] == 0
    assert authority["authority_remains_parent_labeled"] is True
    assert authority["import_is_scientific_evidence"] is False
    assert set(authority["profiles"]) == {
        "gpt52_main",
        "llama33_local_controlled",
    }
    assert contract.budgets["total_usd"] == 150.0
    assert dict(contract.budgets["stage_usd_caps"]) == {
        "parent_v23": 3.212770875,
        "local": 0.0,
        "hosted_confirmatory": 145.787229125,
        "manual_reserve": 1.0,
    }
    assert (
        amendment["budget_projection"]["hard_cap_status"]
        == "proposed-pending-explicit-authorization"
    )
    assert (
        amendment["budget_projection"]["paid_dispatch_allowed_while_draft"]
        is False
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["matrix_amendment"]["budget_carry_forward"].update(
                {"cost_usd": 3.0}
            ),
            "matrix amendment drifted",
        ),
        (
            lambda value: value["matrix_amendment"]["parent_authority_import"][
                "profiles"
            ]["gpt52_main"].update(
                {"authority_receipt_file_sha256": "a" * 64}
            ),
            "matrix amendment drifted",
        ),
        (
            lambda value: value["matrix_amendment"]["matrix"][
                "hosted_arms"
            ].update({"B": ["full", "no-memory"]}),
            "matrix amendment drifted",
        ),
        (
            lambda value: value["changes"]["active_provider_profile_ids"].append(
                "gpt56_diagnostic"
            ),
            "active provider profile list drifted",
        ),
    ],
)
def test_v2_4_resealed_science_or_parent_drift_fails(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    value = _overlay_document()
    mutate(value)
    path = _write_resealed_overlay(tmp_path, value)
    with pytest.raises(PilotContractError, match=message):
        load_pilot_contract(path)


def test_v2_4_cannot_be_relabelled_frozen_before_ci_freeze(
    tmp_path: Path,
) -> None:
    value = _overlay_document()
    value["status"] = "frozen"
    path = _write_resealed_overlay(tmp_path, value)
    with pytest.raises(PilotContractError, match="cannot be frozen"):
        load_pilot_contract(path)


def test_v2_4_parent_source_manifest_file_hash_is_enforced(
    tmp_path: Path,
) -> None:
    path = _write_resealed_overlay(tmp_path, _overlay_document())
    manifest = tmp_path / "pilot_v2_4_parent_source_manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PilotContractError, match="manifest file hash drifted"):
        load_pilot_contract(path)


def test_v2_3_remains_immutable_and_readable() -> None:
    contract = load_pilot_contract(BASE_PATH)
    assert contract.contract_id == PILOT_CONTRACT_ID_V2_3
    assert contract.canonical_hash == PILOT_CONTRACT_V2_3_CANONICAL_SHA256
    assert len(contract.expand()) == 174
    assert contract.matrix_amendment is None
