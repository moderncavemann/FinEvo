from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
from pathlib import Path

import pytest

from verified_memory.pilot_contract import (
    _v2_11_11_expected_dispatch_refresh,
    canonical_sha256,
    load_pilot_contract,
)
from verified_memory.pilot_v21111_fresh_cohort import (
    D_BRANCH_IDS,
    DBranchCoordinator,
    FRESH_MAIN_SEEDS,
    OLD_MAIN_SEEDS,
    PilotV21111FreshCohortError,
    _json_copy,
    build_provider_free_acceptance,
    call_plan_for_v21111,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_11.yaml"
HISTORICAL_SEEDS = {
    617_806_385,
    760_687_867,
    959_809_858,
    1_099_057_501,
    1_421_875_452,
    1_769_977_770,
    1_942_013_315,
    2_010_922_376,
}


def _contract():
    return load_pilot_contract(CONTRACT_PATH)


def test_v21111_draft_registers_exactly_87_cells_and_86_science_cells() -> None:
    contract = _contract()
    specs = tuple(contract.expand())
    counts = Counter(spec.stage_id for spec in specs)

    assert contract.contract_id == "finevo-pilot-v2.11.11"
    assert contract.status == "draft"
    assert len(specs) == 87
    assert counts == {
        "parent-import": 1,
        "experiment-b": 25,
        "experiment-d": 55,
        "cross-model": 6,
    }
    assert sum(counts[stage] for stage in counts if stage != "parent-import") == 86
    assert len({spec.run_id for spec in specs}) == 87
    assert {
        contract.stage(spec.stage_id).evidence_class
        for spec in specs
        if spec.stage_id == "parent-import"
    } == {"operational"}
    assert {
        contract.stage(spec.stage_id).evidence_class
        for spec in specs
        if spec.stage_id != "parent-import"
    } == {"scientific"}


def test_v21111_fresh_seed_vector_replays_exact_sha256_derivation() -> None:
    contract = _contract()
    generation = contract.seeds["generation"]
    values = tuple(generation["values"])
    trace = tuple(generation["derivation_trace"])

    assert values == FRESH_MAIN_SEEDS
    assert set(values).isdisjoint(HISTORICAL_SEEDS)
    assert set(values).isdisjoint(OLD_MAIN_SEEDS)
    assert generation["fresh_values_overlap_historical_registry"] == ()
    assert tuple(generation["historical_seed_registry"]) == tuple(
        sorted(HISTORICAL_SEEDS)
    )
    assert (
        canonical_sha256(sorted(HISTORICAL_SEEDS))
        == generation["historical_seed_registry_sha256"]
    )

    accepted: list[int] = []
    for counter, row in enumerate(trace):
        preimage = f"finevo-pilot-v2.11.11|fresh-seed-v1|main|{counter}"
        digest = hashlib.sha256(preimage.encode("utf-8")).digest()
        candidate = int.from_bytes(digest[:8], "big") % 2_147_483_647
        assert row == {
            "candidate": candidate,
            "counter": counter,
            "digest_sha256": digest.hex(),
            "preimage": preimage,
            "rejected_reason": None,
            "status": "accepted",
        }
        assert 1 <= candidate <= 2_147_483_646
        assert candidate not in HISTORICAL_SEEDS
        assert candidate not in accepted
        accepted.append(candidate)
    assert tuple(accepted) == values
    assert generation["provenance_class"] == (
        "pre-dispatch locally recorded, agent-proposed SHA-256 derivation"
    )
    assert generation["public_preregistration_claimed"] is False
    assert generation["user_selected_claimed"] is False
    assert generation["random_sampling_claimed"] is False

    unused = generation["unused_preflight_candidate"]
    unused_preimage = "finevo-pilot-v2.11.11|fresh-seed-v1|preflight|0"
    unused_digest = hashlib.sha256(unused_preimage.encode("utf-8")).digest()
    assert unused == {
        "candidate": int.from_bytes(unused_digest[:8], "big") % 2_147_483_647,
        "denominator_role": "unused-non-denominator",
        "digest_sha256": unused_digest.hex(),
        "preimage": unused_preimage,
    }
    assert unused["candidate"] == 1_483_834_206


def test_v21111_refresh_and_science_fit_exact_500_dollar_and_7500_call_caps() -> None:
    contract = _contract()
    boundary = contract.v21111_fresh_cohort_boundary
    assert boundary is not None
    refresh = boundary["dispatch_refresh"]
    budget = boundary["budget_envelope"]

    assert canonical_sha256(_json_copy(refresh)) == canonical_sha256(
        _v2_11_11_expected_dispatch_refresh()
    )
    assert refresh["provider_calls"] == 20
    assert len(refresh["rows"]) == 20
    assert len({row["run_id"] for row in refresh["rows"]}) == 20
    assert refresh["reserved_cost_usd"] == 3.40198125
    assert refresh["service_tier"] == "default"
    assert refresh["short_context_prompt_token_ceiling"] == 272_000
    assert refresh["failure_policy"] == ("global-science-no-go-no-retry-no-replacement")

    assert budget["parent_hosted_completions"] == 4_192
    assert budget["projected_cumulative_hosted_completions"] == 7_468
    assert budget["remaining_hosted_completions"] == 32
    assert budget["hard_completion_cap"] == 7_500
    assert 4_192 + 20 + 3_256 == 7_468
    assert budget["parent_cost_usd"] == 78.3237413125
    assert budget["projected_cumulative_cost_usd"] == 486.1955625625
    assert budget["remaining_cost_usd"] == 13.8044374375
    assert budget["hard_cap_usd"] == 500.0
    assert Decimal(str(budget["projected_cumulative_cost_usd"])) + Decimal(
        str(budget["remaining_cost_usd"])
    ) == Decimal("500")
    science = budget["fresh_full_cap_reserve_usd"]
    assert science == {
        "experiment-b": 155.02914,
        "experiment-d": 170.47086,
        "cross-model": 78.96984,
        "total": 404.46984,
    }
    assert Decimal(str(budget["parent_cost_usd"])) + Decimal(
        str(refresh["reserved_cost_usd"])
    ) + Decimal(str(science["total"])) == Decimal(
        str(budget["projected_cumulative_cost_usd"])
    )


def test_v21111_provider_free_call_plan_is_exactly_3256_and_not_evidence() -> None:
    contract = _contract()
    plan = call_plan_for_v21111(contract)
    acceptance = build_provider_free_acceptance(contract)

    assert plan["registered_cells"] == 87
    assert plan["scientific_cells"] == 86
    assert len(plan["call_counts_by_run"]) == 87
    assert plan["calls_by_stage"] == {
        "experiment-b": 1_440,
        "experiment-d": 1_480,
        "cross-model": 336,
    }
    assert plan["simulated_provider_calls"] == 3_256
    assert plan["provider_construction"] is False
    assert plan["provider_calls"] == 0
    assert plan["hosted_cost_usd"] == 0.0
    assert plan["scientific_evidence"] is False
    assert acceptance["status"] == "go"
    assert acceptance["provider_boundary"] == {
        "provider_construction": False,
        "provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "simulated_provider_calls_are_not_calls": True,
    }
    assert acceptance["scientific_evidence"] is False
    assert contract.task_output_contracts["actor-action"].max_completion_tokens == 8_192
    assert (
        contract.task_output_contracts["semantic-proposal"].max_completion_tokens
        == 4_096
    )


def test_v21111_d_coordinator_isolates_branch_crash_and_forbids_redispatch() -> None:
    coordinator = DBranchCoordinator(seed=FRESH_MAIN_SEEDS[0])
    coordinator.start_prefix()
    coordinator.finish_prefix(success=True)
    coordinator.start_branch("matched-a")

    assert coordinator.recover_after_interruption() == ("matched-a",)
    assert coordinator.branch_status is not None
    assert coordinator.branch_status["matched-a"] == "integrity-stopped"
    assert coordinator.untouched_branches == D_BRANCH_IDS[1:]
    with pytest.raises(PilotV21111FreshCohortError, match="cannot be retried"):
        coordinator.start_branch("matched-a")

    coordinator.start_branch("matched-b")
    coordinator.finish_branch("matched-b", success=True)
    restored = DBranchCoordinator.from_dict(coordinator.to_dict())
    assert restored.branch_status == coordinator.branch_status
    assert restored.active_branch is None
    assert restored.untouched_branches == D_BRANCH_IDS[2:]


def test_v21111_d_prefix_failure_terminalizes_exactly_all_eleven_branches() -> None:
    coordinator = DBranchCoordinator(seed=FRESH_MAIN_SEEDS[-1])
    coordinator.start_prefix()
    coordinator.finish_prefix(success=False)

    assert coordinator.prefix_status == "failed"
    assert coordinator.branch_status is not None
    assert len(coordinator.branch_status) == 11
    assert Counter(coordinator.branch_status.values()) == {"failed-prefix": 11}
    assert coordinator.untouched_branches == ()
    with pytest.raises(
        PilotV21111FreshCohortError,
        match="requires a complete prefix",
    ):
        coordinator.start_branch("matched-a")
