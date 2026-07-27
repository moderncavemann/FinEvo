"""Lane-separated evidence publication for the FinEvo V2.4/V2.5/V2.6 pilot.

V2.4 is deliberately not a continuation of the terminal V2.3 denominator.
Its scientific matrix contains two independently interpreted lanes:

* a complete local Llama-3.3 mechanism lane; and
* a bounded GPT-5.2 confirmatory lane.

This adapter never pools seed directions or treatment effects across those
lanes.  Every C -> A -> D -> B stage receives its own 4-of-5 complete-pair
gate, while all failed, stopped, nonterminal, and missing ITT cells remain in
the package denominator.  Narrative intervention is explicitly deferred and
is not silently treated as either completed or failed evidence.
"""

from __future__ import annotations

import ctypes
import errno
import json
import math
import os
from pathlib import Path
import shutil
from statistics import median
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .pilot_contract import PilotContract, load_pilot_contract
from .pilot_evidence import (
    HISTORICAL_SCOPE,
    PILOT_CHECKSUM_SCHEMA_VERSION,
    PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
    PilotEvidenceError,
    PilotEvidencePackage,
    _aggregate_csv,
    _atomic_bytes,
    _evidence_namespace,
    _experiment_a_gate,
    _experiment_b_summary,
    _experiment_c_gate,
    _experiment_d_gate,
    _json_copy,
    _method_scaffold,
    _normalize_ledger,
    _pretty_bytes,
    _sha256_file,
    _strict_json_load,
    _validated_release_controls,
)


PILOT_V24_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.4-evidence-package-v1"
PILOT_V24_CONTRACT_ID = "finevo-pilot-v2.4"
PILOT_V25_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.5-evidence-package-v1"
PILOT_V25_CONTRACT_ID = "finevo-pilot-v2.5"
PILOT_V26_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.6-evidence-package-v1"
PILOT_V26_CONTRACT_ID = "finevo-pilot-v2.6"
PILOT_V24_STAGE_ORDER = (
    "experiment-c",
    "experiment-a",
    "experiment-d",
    "experiment-b",
)
PILOT_V24_MIN_PAIRED_SEEDS = 4
PILOT_V24_TOTAL_PAIRED_SEEDS = 5

_V24_STAGE_IDS = (
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
_V24_LANES: Mapping[str, Mapping[str, Any]] = {
    "local": {
        "model_id": "llama33_local_controlled",
        "stage_ids": {
            "experiment-c": "local-experiment-c",
            "experiment-a": "local-experiment-a",
            "experiment-d": "local-experiment-d",
            "experiment-b": "local-experiment-b",
        },
        "arms": {
            "experiment-c": (
                "full",
                "unverified-dual",
                "verified-error-candidate",
                "verified-error-forced",
                "unverified-error-forced",
            ),
            "experiment-a": (
                "no-context",
                "prompt-only",
                "retrieval-only",
                "full",
            ),
            "experiment-d": (
                "matched-a",
                "matched-b",
                "no-memory",
                "shuffled-episodic",
                "wrong-context",
                "error-verified",
                "error-unverified",
            ),
            "experiment-b": (
                "no-memory",
                "episodic-only",
                "semantic-only",
                "unverified-dual",
                "full",
            ),
        },
    },
    "gpt52": {
        "model_id": "gpt52_main",
        "stage_ids": {
            "experiment-c": "experiment-c",
            "experiment-a": "experiment-a",
            "experiment-d": "experiment-d",
            "experiment-b": "experiment-b",
        },
        "arms": {
            "experiment-c": (
                "full",
                "unverified-dual",
                "verified-error-candidate",
                "verified-error-forced",
                "unverified-error-forced",
            ),
            "experiment-a": (
                "no-context",
                "prompt-only",
                "retrieval-only",
                "full",
            ),
            "experiment-d": (
                "matched-a",
                "matched-b",
                "no-memory",
                "wrong-context",
                "error-verified",
                "error-unverified",
            ),
            "experiment-b": (
                "full",
                "episodic-only",
                "no-memory",
            ),
        },
    },
}


def _contract_id_version_label(contract_id: Any) -> str:
    if contract_id == PILOT_V24_CONTRACT_ID:
        return "V2.4"
    if contract_id == PILOT_V25_CONTRACT_ID:
        return "V2.5"
    if contract_id == PILOT_V26_CONTRACT_ID:
        return "V2.6"
    raise PilotEvidenceError(
        "lane-separated evidence adapter received another contract"
    )


def _contract_version_label(contract: PilotContract) -> str:
    return _contract_id_version_label(contract.contract_id)


def _evidence_schema_version(contract: PilotContract) -> str:
    if contract.contract_id == PILOT_V24_CONTRACT_ID:
        return PILOT_V24_EVIDENCE_SCHEMA_VERSION
    if contract.contract_id == PILOT_V25_CONTRACT_ID:
        return PILOT_V25_EVIDENCE_SCHEMA_VERSION
    if contract.contract_id == PILOT_V26_CONTRACT_ID:
        return PILOT_V26_EVIDENCE_SCHEMA_VERSION
    raise PilotEvidenceError(
        "lane-separated evidence adapter received another contract"
    )


def _atomic_install_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically install one directory while refusing any destination entry."""

    source = source.absolute()
    target = target.absolute()
    if source.parent != target.parent:
        raise PilotEvidenceError(
            "V2.4 evidence source and destination must share one parent"
        )
    if not source.is_dir() or source.is_symlink():
        raise PilotEvidenceError(
            "V2.4 evidence install source must be a real directory"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    target_raw = os.fsencode(target)
    result: int
    if sys.platform == "darwin":
        rename_exclusive = 0x00000004
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:
            raise PilotEvidenceError(
                "Darwin atomic no-replace rename primitive is unavailable"
            )
        renamex.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renamex.restype = ctypes.c_int
        result = int(renamex(source_raw, target_raw, rename_exclusive))
    elif sys.platform.startswith("linux"):
        rename_noreplace = 0x00000001
        at_fdcwd = -100
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise PilotEvidenceError(
                "Linux atomic no-replace rename primitive is unavailable"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = int(
            renameat2(
                at_fdcwd,
                source_raw,
                at_fdcwd,
                target_raw,
                rename_noreplace,
            )
        )
    else:
        raise PilotEvidenceError(
            f"atomic no-replace publication is unsupported on {sys.platform!r}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PilotEvidenceError(
            f"refusing to overwrite V2.4 evidence package: {target}"
        )
    raise PilotEvidenceError(
        "atomic V2.4 evidence installation failed: " f"{os.strerror(error_number)}"
    )


def _validate_v24_contract_matrix(contract: PilotContract) -> None:
    version_label = _contract_version_label(contract)
    if tuple(contract.stage_ids) != _V24_STAGE_IDS:
        raise PilotEvidenceError(
            f"{version_label} evidence stages differ from the fixed "
            "local-first matrix"
        )
    expected_seeds = tuple(int(seed) for seed in contract.seeds["sets"]["main"])
    if (
        len(expected_seeds) != PILOT_V24_TOTAL_PAIRED_SEEDS
        or len(set(expected_seeds)) != PILOT_V24_TOTAL_PAIRED_SEEDS
    ):
        raise PilotEvidenceError(
            f"{version_label} evidence requires five unique main seeds"
        )

    for lane_id, lane in _V24_LANES.items():
        model_id = str(lane["model_id"])
        stage_ids = lane["stage_ids"]
        arms_by_stage = lane["arms"]
        observed_order = tuple(
            canonical
            for canonical in PILOT_V24_STAGE_ORDER
            if str(stage_ids[canonical]) in contract.stage_ids
        )
        if observed_order != PILOT_V24_STAGE_ORDER:
            raise PilotEvidenceError(
                f"{version_label} {lane_id} lane does not preserve "
                "C -> A -> D -> B"
            )
        for canonical in PILOT_V24_STAGE_ORDER:
            stage_id = str(stage_ids[canonical])
            expected_arms = tuple(str(arm) for arm in arms_by_stage[canonical])
            specs = contract.expand(stage=stage_id)
            if (
                {spec.model_id for spec in specs} != {model_id}
                or {spec.environment_seed for spec in specs} != set(expected_seeds)
                or {spec.arm_id for spec in specs} != set(expected_arms)
                or len(specs) != len(expected_arms) * len(expected_seeds)
            ):
                raise PilotEvidenceError(
                    f"{version_label} {lane_id}/{canonical} registered "
                    "matrix drifted"
                )
    if any(
        spec.arm_id == "narrative-content" or spec.narrative_id != "none"
        for spec in contract.expand()
    ):
        raise PilotEvidenceError(
            f"{version_label} narrative intervention must remain deferred "
            "and unregistered"
        )


def _paired_stage_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage_id: str,
    model_id: str,
    arms: Sequence[str],
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    registered_arms = tuple(str(arm) for arm in arms)
    by_identity: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("stage_id") != stage_id or row.get("model_id") != model_id:
            continue
        identity = (str(row.get("arm_id")), int(row["environment_seed"]))
        by_identity.setdefault(identity, []).append(row)

    seed_rows: dict[str, Any] = {}
    complete_seeds: list[int] = []
    for seed in expected_seeds:
        arm_status: dict[str, Any] = {}
        for arm in registered_arms:
            candidates = by_identity.get((arm, int(seed)), [])
            unique = len(candidates) == 1
            row = candidates[0] if unique else None
            eligible = bool(
                row is not None
                and row.get("status") == "complete"
                and row.get("scientific_eligible") is True
            )
            arm_status[arm] = {
                "ledger_row_count": len(candidates),
                "status": None if row is None else row.get("status"),
                "scientific_eligible": (
                    False if row is None else row.get("scientific_eligible") is True
                ),
                "complete_and_eligible": eligible,
            }
        complete = all(item["complete_and_eligible"] for item in arm_status.values())
        if complete:
            complete_seeds.append(int(seed))
        seed_rows[str(seed)] = {
            "complete_pair": complete,
            "arms": arm_status,
        }
    return {
        "stage_id": stage_id,
        "model_id": model_id,
        "registered_arms": list(registered_arms),
        "expected_seeds": [int(seed) for seed in expected_seeds],
        "complete_paired_seeds": complete_seeds,
        "incomplete_or_failed_seeds": [
            int(seed) for seed in expected_seeds if int(seed) not in complete_seeds
        ],
        "complete_pair_count": len(complete_seeds),
        "required_complete_pair_count": PILOT_V24_MIN_PAIRED_SEEDS,
        "total_registered_pair_count": PILOT_V24_TOTAL_PAIRED_SEEDS,
        "pass": len(complete_seeds) >= PILOT_V24_MIN_PAIRED_SEEDS,
        "seed_rows": seed_rows,
    }


def _lane_aggregate(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    *,
    lane_id: str,
) -> dict[str, Any]:
    lane = _V24_LANES[lane_id]
    model_id = str(lane["model_id"])
    stage_ids = lane["stage_ids"]
    arms = lane["arms"]
    expected_seeds = tuple(int(seed) for seed in contract.seeds["sets"]["main"])
    paired_gates = {
        canonical: _paired_stage_gate(
            rows,
            stage_id=str(stage_ids[canonical]),
            model_id=model_id,
            arms=arms[canonical],
            expected_seeds=expected_seeds,
        )
        for canonical in PILOT_V24_STAGE_ORDER
    }
    detailed = {
        "experiment-c": _experiment_c_gate(
            contract,
            rows,
            stage_id=str(stage_ids["experiment-c"]),
            model_id=model_id,
        ),
        "experiment-a": _experiment_a_gate(
            contract,
            rows,
            stage_id=str(stage_ids["experiment-a"]),
            model_id=model_id,
        ),
        "experiment-d": _experiment_d_gate(
            contract,
            rows,
            stage_id=str(stage_ids["experiment-d"]),
            model_id=model_id,
            arms=arms["experiment-d"],
        ),
        "experiment-b": _experiment_b_summary(
            rows,
            stage_id=str(stage_ids["experiment-b"]),
            model_id=model_id,
            arms=arms["experiment-b"],
        ),
    }
    detailed["experiment-b"] = {
        **detailed["experiment-b"],
        "status": (
            "descriptive-complete" if paired_gates["experiment-b"]["pass"] else "no-go"
        ),
        "scientific_evidence_complete": paired_gates["experiment-b"]["pass"],
        "claim_action": (
            "report the registered architecture comparison descriptively; "
            "do not select a winner by wealth"
            if paired_gates["experiment-b"]["pass"]
            else "report the incomplete architecture denominator without a winner"
        ),
    }
    paired_matrix_pass = all(gate["pass"] for gate in paired_gates.values())
    effect_claims_supported = all(
        detailed[stage]["status"] == "supported"
        for stage in ("experiment-c", "experiment-a", "experiment-d")
    )
    return {
        "lane_id": lane_id,
        "model_id": model_id,
        "stage_order": list(PILOT_V24_STAGE_ORDER),
        "stage_ids": {
            canonical: str(stage_ids[canonical]) for canonical in PILOT_V24_STAGE_ORDER
        },
        "paired_seed_gates": paired_gates,
        "paired_matrix_complete": paired_matrix_pass,
        "effect_claims_supported": effect_claims_supported,
        "gates": detailed,
        "direction_count_scope": (
            f"{lane_id}-only; no direction count is pooled across backbones"
        ),
    }


def _finite_direction(values: Any) -> str | None:
    if not isinstance(values, Mapping) or not values:
        return None
    normalized = []
    for value in values.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        normalized.append(float(value))
    center = median(normalized)
    return "positive" if center > 0 else "negative" if center < 0 else "zero"


def _mechanism_direction(
    stage: str,
    gate: Mapping[str, Any],
) -> Any:
    if gate.get("status") != "supported":
        return None
    if stage == "experiment-c":
        return "verified-lowers-registered-harm"
    if stage == "experiment-a":
        primary = gate.get("primary_contrast")
        return (
            _finite_direction(primary.get("raw_paired_deltas"))
            if isinstance(primary, Mapping)
            else None
        )
    if stage == "experiment-d":
        treatment_gates = gate.get("treatment_gates")
        supported = gate.get("supported_treatments")
        if (
            not isinstance(treatment_gates, Mapping)
            or not isinstance(supported, Sequence)
            or isinstance(supported, (str, bytes))
        ):
            return None
        directions: dict[str, str] = {}
        for treatment in supported:
            treatment_gate = treatment_gates.get(str(treatment))
            if not isinstance(treatment_gate, Mapping):
                continue
            utility_gate = treatment_gate.get("six_step_discounted_utility_gate")
            direction = (
                _finite_direction(utility_gate.get("treatment_deltas"))
                if isinstance(utility_gate, Mapping)
                else None
            )
            if direction is not None:
                directions[str(treatment)] = direction
        return directions or None
    return None


def _cross_lane_mechanism_comparison(
    lanes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for stage in PILOT_V24_STAGE_ORDER:
        local = lanes["local"]
        hosted = lanes["gpt52"]
        local_gate = local["gates"][stage]
        hosted_gate = hosted["gates"][stage]
        local_paired = local["paired_seed_gates"][stage]
        hosted_paired = hosted["paired_seed_gates"][stage]
        local_direction = _mechanism_direction(stage, local_gate)
        hosted_direction = _mechanism_direction(stage, hosted_gate)
        both_supported = bool(
            local_paired["pass"]
            and hosted_paired["pass"]
            and local_gate["status"] == "supported"
            and hosted_gate["status"] == "supported"
        )
        common_registered_treatments: list[str] = []
        local_only_registered_treatments: list[str] = []
        gpt52_only_registered_treatments: list[str] = []
        common_direction_qualified_treatments: list[str] = []
        if stage == "experiment-d":
            excluded_controls = {"matched-a", "matched-b"}
            local_registered = set(local_paired["registered_arms"]) - excluded_controls
            hosted_registered = (
                set(hosted_paired["registered_arms"]) - excluded_controls
            )
            common_registered_treatments = sorted(local_registered & hosted_registered)
            local_only_registered_treatments = sorted(
                local_registered - hosted_registered
            )
            gpt52_only_registered_treatments = sorted(
                hosted_registered - local_registered
            )
            local_directions = (
                dict(local_direction) if isinstance(local_direction, Mapping) else {}
            )
            hosted_directions = (
                dict(hosted_direction) if isinstance(hosted_direction, Mapping) else {}
            )
            common_direction_qualified_treatments = sorted(
                set(common_registered_treatments)
                & set(local_directions)
                & set(hosted_directions)
            )
            directions_known = bool(common_direction_qualified_treatments)
            interaction = bool(
                both_supported
                and any(
                    local_directions[treatment] != hosted_directions[treatment]
                    for treatment in common_direction_qualified_treatments
                )
            )
            same_direction = bool(
                both_supported and directions_known and not interaction
            )
        else:
            directions_known = (
                local_direction is not None and hosted_direction is not None
            )
            same_direction = bool(
                both_supported
                and directions_known
                and local_direction == hosted_direction
            )
            interaction = bool(
                both_supported
                and directions_known
                and local_direction != hosted_direction
            )
        classification = (
            "same-direction-in-two-backbone-micro-pilots"
            if same_direction
            else "backbone-interaction" if interaction else "inconclusive"
        )
        rows.append(
            {
                "stage": stage,
                "local_status": local_gate["status"],
                "gpt52_status": hosted_gate["status"],
                "local_4_of_5_pass": local_paired["pass"],
                "gpt52_4_of_5_pass": hosted_paired["pass"],
                "local_direction": _json_copy(local_direction),
                "gpt52_direction": _json_copy(hosted_direction),
                "common_registered_treatments": (common_registered_treatments),
                "local_only_registered_treatments": (local_only_registered_treatments),
                "gpt52_only_registered_treatments": (gpt52_only_registered_treatments),
                "common_direction_qualified_treatments": (
                    common_direction_qualified_treatments
                ),
                "direction_agreement": same_direction,
                "classification": classification,
                "claim_boundary": (
                    (
                        "the registered direction appeared separately in the "
                        "local and GPT-5.2 micro-pilots; this is not a "
                        "backbone-independent claim"
                    )
                    if same_direction
                    else (
                        (
                            "report a possible backbone interaction; do not pool "
                            "seed directions"
                        )
                        if interaction
                        else (
                            "cross-backbone mechanism direction is inconclusive; "
                            "do not pool seed directions"
                        )
                    )
                ),
            }
        )
    return {
        "aggregation_policy": (
            "compare lane-level registered directions only; never add or pool "
            "seed direction counts"
        ),
        "direction_counts_merged": False,
        "rows": rows,
    }


def _claim_rows(
    contract: PilotContract,
    lanes: Mapping[str, Mapping[str, Any]],
    *,
    denominator: Mapping[str, Any],
    cross_lane: Mapping[str, Any],
) -> list[dict[str, Any]]:
    version_label = _contract_version_label(contract)
    claims: list[dict[str, Any]] = []
    definitions = {
        "experiment-c": (
            "Evidence grounding improves erroneous-rule reliability",
            "false activation, harmful exposure, and cumulative utility-loss directions",
        ),
        "experiment-a": (
            "M1 retrieval contributes beyond regime prompting",
            "full minus prompt-only shock+recovery discounted utility",
        ),
        "experiment-d": (
            "A focal memory/error pulse changes the matched six-step continuation",
            "matched-null- and action-bin-qualified continuation deltas",
        ),
        "experiment-b": (
            "Registered memory architectures can be compared descriptively",
            "seed-level utility, action, retrieval, proposal, and lifecycle summaries",
        ),
    }
    for lane_id, lane in lanes.items():
        for stage in PILOT_V24_STAGE_ORDER:
            gate = lane["gates"][stage]
            claim, metric = definitions[stage]
            claims.append(
                {
                    "lane": lane_id,
                    "claim": claim,
                    "metric": metric,
                    "artifact": f"aggregate.json#/lanes/{lane_id}/gates/{stage}",
                    "status": gate["status"],
                    "boundary": gate["claim_action"],
                }
            )
    for comparison in cross_lane["rows"]:
        claims.append(
            {
                "lane": "cross-lane",
                "claim": (
                    f"{comparison['stage']} direction appears in two "
                    "backbone micro-pilots"
                ),
                "metric": (
                    "separate lane-level 4/5 gate, mechanism status, and "
                    "primary-direction agreement"
                ),
                "artifact": (
                    "aggregate.json#/cross_lane_mechanism_comparison/"
                    f"{comparison['stage']}"
                ),
                "status": comparison["classification"],
                "boundary": comparison["claim_boundary"],
            }
        )
    claims.extend(
        [
            {
                "lane": "not-applicable",
                "claim": "Narrative channel shows controlled semantic response",
                "metric": f"not registered in the {version_label} core matrix",
                "artifact": "aggregate.json#/narrative",
                "status": "deferred-unregistered",
                "boundary": (
                    f"make no {version_label} narrative or "
                    "real-news-understanding claim"
                ),
            },
            {
                "lane": "cross-lane",
                "claim": "Backbone-independent improvement",
                "metric": "prohibited pooled inference",
                "artifact": "aggregate.json#/cross_lane_policy",
                "status": "prohibited",
                "boundary": (
                    "report local and GPT-5.2 directions separately; never pool "
                    "direction counts or use backbone-independent wording"
                ),
            },
            {
                "lane": "all",
                "claim": f"Complete {version_label} preregistered ITT denominator",
                "metric": "one terminal ledger row for every expanded contract cell",
                "artifact": "failure_ledger.json",
                "status": "supported" if denominator.get("pass") else "no-go",
                "boundary": (
                    "retain every failed, stopped, nonterminal, and missing cell"
                ),
            },
        ]
    )
    return claims


def _claim_narrowing(
    lanes: Mapping[str, Mapping[str, Any]],
    *,
    denominator: Mapping[str, Any],
    release_controls: Mapping[str, Any],
    cross_lane: Mapping[str, Any],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for lane_id, lane in lanes.items():
        for stage in PILOT_V24_STAGE_ORDER:
            gate = lane["gates"][stage]
            paired = lane["paired_seed_gates"][stage]
            if not paired["pass"]:
                output.append(
                    {
                        "scope": f"{lane_id}/{stage}",
                        "reason": (
                            f"{paired['complete_pair_count']}/5 complete paired "
                            "seeds; the registered minimum is 4/5"
                        ),
                        "required_wording": (
                            "denominator/failure report only; no effectiveness claim"
                        ),
                    }
                )
            elif gate["status"] not in {"supported", "descriptive-complete"}:
                output.append(
                    {
                        "scope": f"{lane_id}/{stage}",
                        "reason": "the preregistered mechanism gate was not supported",
                        "required_wording": str(gate["claim_action"]),
                    }
                )
    if denominator.get("pass") is not True:
        output.append(
            {
                "scope": "full-denominator",
                "reason": "one or more registered ITT rows are missing or nonterminal",
                "required_wording": (
                    "incomplete; do not publish an immutable evidence package"
                ),
            }
        )
    if release_controls.get("pass") is not True:
        output.append(
            {
                "scope": "release-stage0-budget",
                "reason": "release, Stage-0, or budget controls did not all pass",
                "required_wording": (
                    "complete-with-no-go; do not report scientific effectiveness"
                ),
            }
        )
    for comparison in cross_lane["rows"]:
        if (
            comparison["classification"]
            != "same-direction-in-two-backbone-micro-pilots"
        ):
            output.append(
                {
                    "scope": f"cross-lane/{comparison['stage']}",
                    "reason": (
                        "the two independently gated lanes do not establish "
                        "the same registered primary direction"
                    ),
                    "required_wording": comparison["claim_boundary"],
                }
            )
    output.extend(
        [
            {
                "scope": "narrative",
                "reason": "narrative intervention is deferred and unregistered",
                "required_wording": ("no narrative or real-news-understanding claim"),
            },
            {
                "scope": "cross-lane",
                "reason": "the local and GPT lanes are separate replications",
                "required_wording": (
                    "report each lane's seed directions separately; never pool them"
                ),
            },
        ]
    )
    return output


def aggregate_v24_evidence(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    *,
    denominator: Mapping[str, Any],
    release_controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the pure lane-separated aggregate without artifact I/O."""

    _validate_v24_contract_matrix(contract)
    version_label = _contract_version_label(contract)
    schema_version = _evidence_schema_version(contract)
    lanes = {
        lane_id: _lane_aggregate(
            contract,
            rows,
            lane_id=lane_id,
        )
        for lane_id in ("local", "gpt52")
    }
    cross_lane = _cross_lane_mechanism_comparison(lanes)
    claims = _claim_rows(
        contract,
        lanes,
        denominator=denominator,
        cross_lane=cross_lane,
    )
    narrowing = _claim_narrowing(
        lanes,
        denominator=denominator,
        release_controls=release_controls,
        cross_lane=cross_lane,
    )
    scientific_matrix_complete = bool(
        denominator.get("pass") is True
        and release_controls.get("pass") is True
        and all(lane["paired_matrix_complete"] for lane in lanes.values())
    )
    scientific_claim_gates_supported = all(
        lane["effect_claims_supported"] for lane in lanes.values()
    )
    scientific_complete = bool(
        scientific_matrix_complete and scientific_claim_gates_supported
    )
    denominator_terminal = denominator.get("pass") is True
    publication_status = (
        "incomplete"
        if not denominator_terminal
        else "complete" if scientific_complete else "complete-with-no-go"
    )
    return {
        "schema_version": schema_version,
        "evidence_namespace": _evidence_namespace(contract),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "fixed_matrix_order": list(PILOT_V24_STAGE_ORDER),
        "denominator": _json_copy(denominator),
        "budget": _json_copy(release_controls.get("budget_ledger", {})),
        "release_controls": _json_copy(release_controls),
        "lanes": _json_copy(lanes),
        "cross_lane_mechanism_comparison": _json_copy(cross_lane),
        "cross_lane_policy": {
            "direction_counts_merged": False,
            "effect_estimates_pooled": False,
            "allowed_interpretation": (
                "separate local and GPT-5.2 mechanism-pilot directions only"
            ),
            "prohibited_wording": "backbone-independent",
        },
        "narrative": {
            "status": "deferred-unregistered",
            "registered_cells": 0,
            "claim_boundary": (
                f"no {version_label} narrative or real-news-understanding claim"
            ),
        },
        "claims": claims,
        "claim_narrowing": narrowing,
        "scientific_matrix_complete": scientific_matrix_complete,
        "scientific_claim_gates_supported": scientific_claim_gates_supported,
        "scientific_complete": scientific_complete,
        "publication_status": publication_status,
    }


def _require_publishable_terminal_denominator(
    aggregate: Mapping[str, Any],
) -> None:
    if aggregate.get("publication_status") == "incomplete":
        version_label = _contract_id_version_label(aggregate.get("contract_id"))
        denominator = aggregate.get("denominator")
        status_counts = (
            denominator.get("status_counts")
            if isinstance(denominator, Mapping)
            else None
        )
        raise PilotEvidenceError(
            f"{version_label} immutable evidence publication requires all "
            "211 ITT cells "
            f"to be present and terminal; status_counts={status_counts!r}"
        )


def _sanitized_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "run_id",
        "contract_id",
        "stage_id",
        "model_id",
        "requested_model",
        "arm_id",
        "narrative_id",
        "environment_seed",
        "decoding_seed",
        "utility_profile_id",
        "shock_id",
        "budget_bucket",
        "num_agents",
        "episode_length",
        "execution_mode",
        "status",
        "failure",
        "artifact_kind",
        "artifact_sha256",
        "scientific_eligible",
        "metrics",
        "gate_evidence",
        "capability",
        "narrative",
    )
    return [{field: _json_copy(row.get(field)) for field in fields} for row in rows]


def _report_markdown(
    aggregate: Mapping[str, Any],
) -> str:
    version_label = _contract_id_version_label(aggregate.get("contract_id"))
    lines = [
        f"# FinEvo {version_label} local-first mechanism pilot evidence report",
        "",
        f"- Contract: `{aggregate['contract_id']}` / "
        f"`{aggregate['contract_sha256']}`",
        f"- Publication status: `{aggregate['publication_status']}`",
        f"- Registered denominator: "
        f"`{aggregate['denominator']['expected_count']}` cells",
        "- Matrix order: `C -> A -> D -> B` in each lane.",
        "- Local and GPT-5.2 directions are never pooled.",
        "- Narrative intervention: `deferred-unregistered`.",
        "",
        "## Claim -> metric -> artifact",
        "",
        "| Lane | Claim | Metric | Artifact | Status | Boundary |",
        "|---|---|---|---|---|---|",
    ]
    for claim in aggregate["claims"]:
        lines.append(
            "| "
            + " | ".join(
                str(claim[field]).replace("|", "\\|")
                for field in (
                    "lane",
                    "claim",
                    "metric",
                    "artifact",
                    "status",
                    "boundary",
                )
            )
            + " |"
        )
    for lane_id, lane in aggregate["lanes"].items():
        lines.extend(
            [
                "",
                f"## {lane_id} lane",
                "",
                f"- Model profile: `{lane['model_id']}`",
                f"- 4/5 paired matrix complete: "
                f"`{str(lane['paired_matrix_complete']).lower()}`",
            ]
        )
        for stage in PILOT_V24_STAGE_ORDER:
            paired = lane["paired_seed_gates"][stage]
            gate = lane["gates"][stage]
            lines.append(
                f"- `{stage}`: {paired['complete_pair_count']}/5 complete "
                f"paired seeds; gate `{gate['status']}`; "
                f"claim action: {gate['claim_action']}."
            )
    lines.extend(
        [
            "",
            "## Cross-lane mechanism comparison",
            "",
            "| Stage | Local status / 4-of-5 | GPT-5.2 status / 4-of-5 | "
            "Direction agreement | Classification | Boundary |",
            "|---|---|---|---|---|---|",
        ]
    )
    for comparison in aggregate["cross_lane_mechanism_comparison"]["rows"]:
        lines.append(
            f"| `{comparison['stage']}` | "
            f"`{comparison['local_status']}` / "
            f"`{str(comparison['local_4_of_5_pass']).lower()}` | "
            f"`{comparison['gpt52_status']}` / "
            f"`{str(comparison['gpt52_4_of_5_pass']).lower()}` | "
            f"`{str(comparison['direction_agreement']).lower()}` | "
            f"`{comparison['classification']}` | "
            f"{comparison['claim_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Denominator, failures, and budget",
            "",
            f"- ITT denominator pass: "
            f"`{str(aggregate['denominator']['pass']).lower()}`",
            f"- Status counts: "
            f"`{json.dumps(aggregate['denominator']['status_counts'], sort_keys=True)}`",
            f"- Budget control: "
            f"`{str(bool(aggregate['budget'].get('pass'))).lower()}`",
            "- Every failed, stopped, nonterminal, and missing cell remains in "
            "`failure_ledger.json` and the aggregate rows.",
            "",
            "## Explicit claim narrowing",
            "",
        ]
    )
    for item in aggregate["claim_narrowing"]:
        lines.append(
            f"- `{item['scope']}`: {item['reason']}; " f"{item['required_wording']}."
        )
    return "\n".join(lines) + "\n"


def _write_v24_package(
    root: Path,
    *,
    contract_path: Path,
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    common_commit: str | None,
) -> tuple[Path, Path]:
    version_label = _contract_version_label(contract)
    schema_version = _evidence_schema_version(contract)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise PilotEvidenceError(
            f"temporary {version_label} package directory is not empty: {root}"
        )
    contract_target = root / "contract" / contract_path.name
    contract_target.parent.mkdir(parents=True, exist_ok=True)
    if contract.matrix_amendment is None:
        raise PilotEvidenceError(
            f"{version_label} contract lacks its matrix amendment"
        )
    parent_binding_raw = contract.matrix_amendment.get(
        "parent_source_manifest"
    )
    if not isinstance(parent_binding_raw, Mapping):
        raise PilotEvidenceError(
            f"{version_label} contract lacks its parent source manifest binding"
        )
    parent_binding = dict(parent_binding_raw)
    parent_manifest_name = "pilot_v2_4_parent_source_manifest.json"
    parent_manifest_path = str(parent_binding.get("path", ""))
    if (
        parent_manifest_path
        != f"experiments/{parent_manifest_name}"
        or Path(parent_manifest_path).name != parent_manifest_name
    ):
        raise PilotEvidenceError(
            f"{version_label} parent source manifest package path drifted"
        )
    parent_manifest_source = contract_path.with_name(parent_manifest_name)
    if not parent_manifest_source.is_file():
        raise PilotEvidenceError(
            f"{version_label} parent source manifest sibling is missing"
        )
    parent_manifest_target = contract_target.with_name(parent_manifest_name)
    shutil.copyfile(contract_path, contract_target)
    shutil.copyfile(parent_manifest_source, parent_manifest_target)
    if _sha256_file(parent_manifest_target) != parent_binding.get(
        "file_sha256"
    ):
        raise PilotEvidenceError(
            f"copied {version_label} parent source manifest failed "
            "hash revalidation"
        )

    retry_binding: dict[str, Any] | None = None
    retry_manifest_name: str | None = None
    inherited_retry_binding: dict[str, Any] | None = None
    inherited_retry_manifest_name: str | None = None
    base_binding: dict[str, Any] | None = None
    base_contract_name: str | None = None
    if contract.contract_id == PILOT_V25_CONTRACT_ID:
        retry_amendment = contract.parent_import_retry_amendment
        if not isinstance(retry_amendment, Mapping):
            raise PilotEvidenceError(
                "V2.5 contract lacks its parent-import retry amendment"
            )
        retry_binding_raw = retry_amendment.get("source_manifest")
        if not isinstance(retry_binding_raw, Mapping):
            raise PilotEvidenceError(
                "V2.5 contract lacks its retry source manifest binding"
            )
        retry_binding = dict(retry_binding_raw)
        retry_manifest_name = "pilot_v2_5_source_manifest.json"
        retry_manifest_path = str(retry_binding.get("path", ""))
        if (
            retry_manifest_path != f"experiments/{retry_manifest_name}"
            or Path(retry_manifest_path).name != retry_manifest_name
        ):
            raise PilotEvidenceError(
                "V2.5 retry source manifest package path drifted"
            )
        retry_manifest_source = contract_path.with_name(retry_manifest_name)
        if not retry_manifest_source.is_file():
            raise PilotEvidenceError(
                "V2.5 retry source manifest sibling is missing"
            )
        retry_manifest_target = contract_target.with_name(retry_manifest_name)
        shutil.copyfile(retry_manifest_source, retry_manifest_target)
        if _sha256_file(retry_manifest_target) != retry_binding.get(
            "file_sha256"
        ):
            raise PilotEvidenceError(
                "copied V2.5 retry source manifest failed hash revalidation"
            )

        contract_document = _strict_json_load(contract_path)
        base_binding_raw = contract_document.get("base_contract")
        if base_binding_raw is not None:
            if not isinstance(base_binding_raw, Mapping):
                raise PilotEvidenceError(
                    "V2.5 overlay base contract binding is malformed"
                )
            base_binding = dict(base_binding_raw)
            base_contract_name = "pilot_v2_4.yaml"
            if (
                base_binding.get("path") != base_contract_name
                or Path(str(base_binding.get("path", ""))).name
                != base_contract_name
            ):
                raise PilotEvidenceError(
                    "V2.5 overlay base contract package path drifted"
                )
            base_contract_source = contract_path.with_name(base_contract_name)
            if not base_contract_source.is_file():
                raise PilotEvidenceError(
                    "V2.5 overlay base contract sibling is missing"
                )
            shutil.copyfile(
                base_contract_source,
                contract_target.with_name(base_contract_name),
            )
    elif contract.contract_id == PILOT_V26_CONTRACT_ID:
        retry_amendment = getattr(
            contract,
            "p95_authority_retry_amendment",
            None,
        )
        if not isinstance(retry_amendment, Mapping):
            raise PilotEvidenceError(
                "V2.6 contract lacks its p95-authority retry amendment"
            )
        retry_binding_raw = retry_amendment.get("source_manifest")
        if not isinstance(retry_binding_raw, Mapping):
            raise PilotEvidenceError(
                "V2.6 contract lacks its retry source manifest binding"
            )
        retry_binding = dict(retry_binding_raw)
        retry_manifest_name = "pilot_v2_6_source_manifest.json"
        retry_manifest_path = str(retry_binding.get("path", ""))
        if (
            retry_manifest_path != f"experiments/{retry_manifest_name}"
            or Path(retry_manifest_path).name != retry_manifest_name
        ):
            raise PilotEvidenceError(
                "V2.6 retry source manifest package path drifted"
            )
        retry_manifest_source = contract_path.with_name(retry_manifest_name)
        if not retry_manifest_source.is_file():
            raise PilotEvidenceError(
                "V2.6 retry source manifest sibling is missing"
            )
        retry_manifest_target = contract_target.with_name(retry_manifest_name)
        shutil.copyfile(retry_manifest_source, retry_manifest_target)
        if _sha256_file(retry_manifest_target) != retry_binding.get(
            "file_sha256"
        ):
            raise PilotEvidenceError(
                "copied V2.6 retry source manifest failed hash revalidation"
            )

        inherited_retry = contract.parent_import_retry_amendment
        if not isinstance(inherited_retry, Mapping):
            raise PilotEvidenceError(
                "V2.6 contract lacks the inherited V2.5 retry amendment"
            )
        inherited_retry_binding_raw = inherited_retry.get("source_manifest")
        if not isinstance(inherited_retry_binding_raw, Mapping):
            raise PilotEvidenceError(
                "V2.6 contract lacks the inherited V2.5 source binding"
            )
        inherited_retry_binding = dict(inherited_retry_binding_raw)
        inherited_retry_manifest_name = "pilot_v2_5_source_manifest.json"
        if (
            inherited_retry_binding.get("path")
            != f"experiments/{inherited_retry_manifest_name}"
        ):
            raise PilotEvidenceError(
                "V2.6 inherited V2.5 source-manifest path drifted"
            )
        inherited_retry_source = contract_path.with_name(
            inherited_retry_manifest_name
        )
        inherited_retry_target = contract_target.with_name(
            inherited_retry_manifest_name
        )
        if not inherited_retry_source.is_file():
            raise PilotEvidenceError(
                "V2.6 inherited V2.5 source manifest is missing"
            )
        shutil.copyfile(inherited_retry_source, inherited_retry_target)
        if _sha256_file(inherited_retry_target) != inherited_retry_binding.get(
            "file_sha256"
        ):
            raise PilotEvidenceError(
                "copied V2.5 source manifest failed hash revalidation"
            )

        contract_document = _strict_json_load(contract_path)
        base_binding_raw = contract_document.get("base_contract")
        base_contract_name = "pilot_v2_5.yaml"
        if base_binding_raw is None:
            base_binding = {
                "path": base_contract_name,
                "schema_version": "finevo-pilot-contract-v2",
                "contract_id": PILOT_V25_CONTRACT_ID,
                "canonical_sha256": (
                    contract.p95_authority_retry_amendment[
                        "failure_classification"
                    ]["parent_contract_sha256"]
                ),
            }
        elif isinstance(base_binding_raw, Mapping):
            base_binding = dict(base_binding_raw)
        else:
            raise PilotEvidenceError(
                "V2.6 overlay base contract binding is malformed"
            )
        if base_binding.get("path") != base_contract_name:
            raise PilotEvidenceError(
                "V2.6 base contract package path drifted"
            )
        base_contract_source = contract_path.with_name(base_contract_name)
        if not base_contract_source.is_file():
            raise PilotEvidenceError(
                "V2.6 base contract sibling is missing"
            )
        shutil.copyfile(
            base_contract_source,
            contract_target.with_name(base_contract_name),
        )
        copied_base = load_pilot_contract(
            contract_target.with_name(base_contract_name)
        )
        if (
            copied_base.contract_id != PILOT_V25_CONTRACT_ID
            or copied_base.canonical_hash
            != base_binding.get("canonical_sha256")
        ):
            raise PilotEvidenceError(
                "copied V2.5 base contract failed identity revalidation"
            )

    copied = load_pilot_contract(contract_target)
    if copied.canonical_hash != contract.canonical_hash:
        raise PilotEvidenceError(
            f"copied {version_label} contract failed hash revalidation"
        )

    sanitized = _sanitized_rows(rows)
    aggregate_payload = {
        **_json_copy(aggregate),
        "resolved_git_commit": common_commit,
        "rows": sanitized,
    }
    _atomic_bytes(root / "aggregate.json", _pretty_bytes(aggregate_payload))
    _atomic_bytes(root / "aggregate.csv", _aggregate_csv(sanitized))
    _atomic_bytes(
        root / "claim_metric_artifact.json",
        _pretty_bytes(
            {
                "schema_version": schema_version,
                "contract_sha256": contract.canonical_hash,
                "claims": aggregate["claims"],
            }
        ),
    )
    _atomic_bytes(
        root / "claim_narrowing.json",
        _pretty_bytes(
            {
                "schema_version": schema_version,
                "contract_sha256": contract.canonical_hash,
                "rows": aggregate["claim_narrowing"],
            }
        ),
    )
    failures = [
        {
            "run_id": row["run_id"],
            "stage_id": row["stage_id"],
            "model_id": row["model_id"],
            "arm_id": row["arm_id"],
            "environment_seed": row["environment_seed"],
            "status": row["status"],
            "failure": row["failure"],
        }
        for row in sanitized
        if row["status"] != "complete"
    ]
    _atomic_bytes(
        root / "failure_ledger.json",
        _pretty_bytes(
            {
                "schema_version": PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
                "contract_sha256": contract.canonical_hash,
                "denominator": aggregate["denominator"],
                "rows": failures,
            }
        ),
    )
    _atomic_bytes(
        root / "method_differences_scaffold.json",
        _pretty_bytes(_method_scaffold(contract_path.name)),
    )
    _atomic_bytes(
        root / "reviewer_report.md",
        _report_markdown(aggregate).encode("utf-8"),
    )
    published_files = sorted(
        [
            "aggregate.csv",
            "aggregate.json",
            "claim_metric_artifact.json",
            "claim_narrowing.json",
            f"contract/{contract_path.name}",
            f"contract/{parent_manifest_name}",
            "failure_ledger.json",
            "method_differences_scaffold.json",
            "reviewer_report.md",
        ]
        + (
            [f"contract/{retry_manifest_name}"]
            if retry_manifest_name is not None
            else []
        )
        + (
            [f"contract/{inherited_retry_manifest_name}"]
            if inherited_retry_manifest_name is not None
            else []
        )
        + (
            [f"contract/{base_contract_name}"]
            if base_contract_name is not None
            else []
        )
    )
    manifest = {
        "schema_version": schema_version,
        "evidence_namespace": _evidence_namespace(contract),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "resolved_git_commit": common_commit,
        "scientific_matrix_complete": aggregate["scientific_matrix_complete"],
        "scientific_claim_gates_supported": aggregate[
            "scientific_claim_gates_supported"
        ],
        "scientific_complete": aggregate["scientific_complete"],
        "publication_status": aggregate["publication_status"],
        "lane_separated": True,
        "direction_counts_merged": False,
        "narrative_status": "deferred-unregistered",
        "parent_source_manifest": {
            **parent_binding,
            "package_path": f"contract/{parent_manifest_name}",
        },
        "published_files": published_files,
        "excluded_sources": [
            HISTORICAL_SCOPE,
            "V2.3 scientific outcomes",
            "pooled local/GPT direction counts",
            "unregistered narrative intervention",
            "raw prompts and raw provider outputs",
        ],
    }
    if retry_binding is not None and retry_manifest_name is not None:
        manifest["retry_source_manifest"] = {
            **retry_binding,
            "package_path": f"contract/{retry_manifest_name}",
        }
    if (
        inherited_retry_binding is not None
        and inherited_retry_manifest_name is not None
    ):
        manifest["inherited_retry_source_manifest"] = {
            **inherited_retry_binding,
            "package_path": f"contract/{inherited_retry_manifest_name}",
        }
    if base_binding is not None and base_contract_name is not None:
        manifest["base_contract"] = {
            **base_binding,
            "package_path": f"contract/{base_contract_name}",
        }
    manifest_path = root / "package_manifest.json"
    _atomic_bytes(manifest_path, _pretty_bytes(manifest))
    checksum_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    )
    checksums = {
        "schema_version": PILOT_CHECKSUM_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "byte_size": path.stat().st_size,
            }
            for path in checksum_files
        ],
    }
    checksums_path = root / "checksums.json"
    _atomic_bytes(checksums_path, _pretty_bytes(checksums))
    for row in checksums["files"]:
        path = root / row["path"]
        if (
            _sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["byte_size"]
        ):
            raise PilotEvidenceError(
                f"{version_label} package checksum self-verification failed"
            )
    raw_storage = aggregate["budget"].get("raw_root_storage_bytes")
    package_storage = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    if (
        not isinstance(raw_storage, (int, float))
        or isinstance(raw_storage, bool)
        or float(raw_storage) < 0
        or float(raw_storage) + package_storage
        > float(contract.budgets["max_storage_bytes"])
    ):
        raise PilotEvidenceError(
            f"{version_label} raw evidence plus reviewer package exceeds "
            "storage cap"
        )
    return manifest_path, checksums_path


def build_pilot_v24_evidence_package(
    *,
    contract_path: str | Path,
    run_ledger_path: str | Path,
    raw_root: str | Path,
    build_root: str | Path,
) -> PilotEvidencePackage:
    """Validate and publish a lane-separated package without provider calls."""

    contract_source = Path(contract_path).resolve()
    contract = load_pilot_contract(contract_source)
    _validate_v24_contract_matrix(contract)
    version_label = _contract_version_label(contract)
    if contract.status != "frozen":
        raise PilotEvidenceError(
            f"{version_label} evidence publication requires the frozen "
            "science contract"
        )
    raw = Path(raw_root).resolve()
    if not raw.is_dir():
        raise PilotEvidenceError(
            f"{version_label} pilot raw root does not exist: {raw}"
        )
    ledger = _strict_json_load(Path(run_ledger_path).resolve())
    rows, denominator, common_commit = _normalize_ledger(
        contract,
        ledger,
        raw_root=raw,
    )
    release_controls = _validated_release_controls(
        contract,
        raw_root=raw,
        rows=rows,
        common_commit=common_commit,
    )
    aggregate = aggregate_v24_evidence(
        contract,
        rows,
        denominator=denominator,
        release_controls=release_controls,
    )
    _require_publishable_terminal_denominator(aggregate)

    target = Path(build_root).resolve() / _evidence_namespace(contract)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}-build-",
            dir=target.parent,
        )
    )
    try:
        manifest, checksums = _write_v24_package(
            temporary,
            contract_path=contract_source,
            contract=contract,
            rows=rows,
            aggregate=aggregate,
            common_commit=common_commit,
        )
        _atomic_install_directory_no_replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PilotEvidencePackage(
        package_dir=target,
        manifest_path=target / manifest.name,
        checksums_path=target / checksums.name,
        contract_hash=contract.canonical_hash,
        scientific_complete=bool(aggregate["scientific_complete"]),
        claim_gates={
            "lanes": _json_copy(aggregate["lanes"]),
            "cross_lane_mechanism_comparison": _json_copy(
                aggregate["cross_lane_mechanism_comparison"]
            ),
            "narrative": _json_copy(aggregate["narrative"]),
            "cross_lane_policy": _json_copy(aggregate["cross_lane_policy"]),
        },
    )


aggregate_lane_separated_evidence = aggregate_v24_evidence
build_lane_separated_evidence_package = build_pilot_v24_evidence_package


__all__ = [
    "PILOT_V24_CONTRACT_ID",
    "PILOT_V24_EVIDENCE_SCHEMA_VERSION",
    "PILOT_V24_MIN_PAIRED_SEEDS",
    "PILOT_V24_STAGE_ORDER",
    "PILOT_V24_TOTAL_PAIRED_SEEDS",
    "PILOT_V25_CONTRACT_ID",
    "PILOT_V25_EVIDENCE_SCHEMA_VERSION",
    "PILOT_V26_CONTRACT_ID",
    "PILOT_V26_EVIDENCE_SCHEMA_VERSION",
    "aggregate_lane_separated_evidence",
    "aggregate_v24_evidence",
    "build_lane_separated_evidence_package",
    "build_pilot_v24_evidence_package",
]
