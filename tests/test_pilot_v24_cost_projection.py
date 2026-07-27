from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_4.yaml"
PROJECTION_PATH = ROOT / "experiments" / "pilot_v2_4_cost_projection.json"


def _load_projection() -> dict:
    return json.loads(PROJECTION_PATH.read_text(encoding="utf-8"))


def _expected_calls(contract, stage_id: str) -> tuple[int, int, int]:
    specs = contract.expand(stage=stage_id)
    if stage_id in {"parent-import", "q-ref-resolution"}:
        return 0, 0, len(specs)
    if stage_id.endswith("experiment-d"):
        action_calls = 0
        semantic_calls = 0
        groups = {
            (spec.model_id, spec.environment_seed)
            for spec in specs
        }
        for model_id, seed in groups:
            group = [
                spec
                for spec in specs
                if spec.model_id == model_id
                and spec.environment_seed == seed
            ]
            action_calls += 4 * 6 * (1 + len(group))
            semantic_calls += 4 * 2
        return action_calls, semantic_calls, len(groups)

    action_calls = 0
    semantic_calls = 0
    execution_groups = 0
    for spec in specs:
        execution_groups += 1
        if spec.execution_mode == "offline_candidate_admission":
            continue
        action_calls += spec.num_agents * spec.episode_length
        if bool(
            contract.arms[spec.arm_id]["parameters"].get(
                "semantic_actor_exposure",
                True,
            )
        ):
            due = sum(
                current_t >= 3 and (current_t - 3) % 3 == 0
                for current_t in range(1, spec.episode_length + 1)
            )
            semantic_calls += spec.num_agents * min(due, 4)
    return action_calls, semantic_calls, execution_groups


def test_v24_cost_projection_matches_contract_and_frozen_p95() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    payload = _load_projection()
    assert payload["status"] == "draft-preauthorization"
    assert payload["contract"] == {
        "contract_id": contract.contract_id,
        "draft_canonical_sha256": contract.canonical_hash,
        "scientific_outcomes_observed_before_amendment": False,
    }

    rows = {row["stage_id"]: row for row in payload["rows"]}
    assert list(rows) == [stage.stage_id for stage in contract.stages]
    expected_cells = 0
    expected_scientific_cells = 0
    local_calls = 0
    hosted_calls = 0
    hosted_prompt = 0
    hosted_completion = 0
    hosted_cost = Decimal("0")
    storage = 0

    for stage in contract.stages:
        stage_id = stage.stage_id
        row = rows[stage_id]
        specs = contract.expand(stage=stage_id)
        action_calls, semantic_calls, execution_groups = _expected_calls(
            contract,
            stage_id,
        )
        assert row["registered_cells"] == len(specs)
        assert row["execution_groups"] == execution_groups
        assert row["actor_action_calls"] == action_calls
        assert row["semantic_proposal_calls"] == semantic_calls
        assert row["logical_calls"] == action_calls + semantic_calls
        expected_cells += len(specs)
        if stage_id not in {"parent-import", "q-ref-resolution"}:
            expected_scientific_cells += len(specs)

        if row["model_id"] == "llama33_local_controlled":
            local_calls += row["logical_calls"]
        if row["model_id"] == "gpt52_main":
            reservations = payload["per_call_reservations"]["gpt52_main"]
            prompt = (
                action_calls
                * reservations["actor-action"]["prompt_tokens"]
                + semantic_calls
                * reservations["semantic-proposal"]["prompt_tokens"]
            )
            completion = (
                action_calls
                * reservations["actor-action"]["completion_tokens"]
                + semantic_calls
                * reservations["semantic-proposal"]["completion_tokens"]
            )
            cost = (
                Decimal(action_calls)
                * Decimal(str(reservations["actor-action"]["cost_usd"]))
                + Decimal(semantic_calls)
                * Decimal(str(reservations["semantic-proposal"]["cost_usd"]))
            )
            assert row["hosted_completions"] == row["logical_calls"]
            assert row["reserved_prompt_tokens"] == prompt
            assert row["reserved_completion_tokens"] == completion
            assert Decimal(str(row["reserved_cost_usd"])) == cost
            hosted_calls += row["hosted_completions"]
            hosted_prompt += prompt
            hosted_completion += completion
            hosted_cost += cost
        else:
            assert row["hosted_completions"] == 0

        storage += int(row.get("reserved_storage_bytes", 0))

    totals = payload["totals"]
    assert expected_cells == totals["registered_cells"] == 211
    assert expected_scientific_cells == totals["scientific_cells"] == 209
    assert local_calls == totals["new_local_logical_calls"] == 5672
    assert hosted_calls == totals["new_hosted_completions"] == 4240
    assert hosted_prompt == totals["new_hosted_prompt_tokens"] == 6_253_920
    assert (
        hosted_completion
        == totals["new_hosted_completion_tokens"]
        == 9_744_560
    )
    assert (
        hosted_prompt + hosted_completion
        == totals["new_hosted_total_tokens"]
        == 15_998_480
    )
    assert hosted_cost == Decimal("143.6043000000")
    assert Decimal(str(totals["new_hosted_reserved_cost_usd"])) == hosted_cost
    assert storage == totals["new_reserved_storage_bytes"] == 3_520_000_000


def test_v24_cost_projection_preserves_parent_and_requires_authorization() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    payload = _load_projection()
    totals = payload["totals"]
    parent = Decimal(str(payload["authority"]["parent_debit_usd"]))
    hosted = Decimal(str(totals["new_hosted_reserved_cost_usd"]))
    reserve = Decimal(str(totals["manual_reserve_usd"]))
    required = parent + hosted + reserve
    hard_cap = Decimal(str(totals["proposed_total_hard_cap_usd"]))

    assert required == Decimal(str(totals["parent_plus_matrix_plus_manual_reserve_usd"]))
    assert hard_cap - required == Decimal(
        str(totals["unallocated_hosted_headroom_usd"])
    )
    assert contract.status == "draft"
    assert contract.budgets["total_usd"] == float(hard_cap)
    assert payload["authorization"] == {
        "hard_cap_authorized": False,
        "paid_dispatch_authorized": False,
        "freeze_or_tag_authorized": False,
        "rule": (
            "No paid or scientific dispatch while the V2.4 contract "
            "remains draft."
        ),
    }
