"""Provider-free, fail-closed evidence consumer for FinEvo V2.11.10.

V2.11.10 contains one operational lineage row and 86 continuation rows.  The
paper denominator, however, is the cross-release logical V2.11.5 matrix: the
50 terminal V2.11.5 parent rows plus the 86 V2.11.10 continuation rows.  This
module is deliberately separate from :mod:`pilot_evidence`; treating V2.11.10
as an ordinary 136-row V2.11 contract either drops its imported A/C evidence
or invents cells that do not exist in the current ledger.

The consumer never constructs a provider, retries a cell, changes a status, or
rewrites raw evidence.  It validates the current V2.11.10, failed V2.11.9,
and scientific-authority V2.11.5 release roots and their ledgers,
replays the V2.11.10 authority/projection/acceptance/terminal-artifact controls,
retains every current ITT failure, and writes a new reviewer package atomically.
The 87 failed-release rows are lineage-only and never enter or overwrite the
136-row logical scientific matrix.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from . import pilot_v2117_continuation as v2117
from .pilot_analysis import paired_delta_summary
from .pilot_budget import PilotBudgetLedger
from .pilot_contract import PilotContract, canonical_sha256, load_pilot_contract
from .pilot_evidence import (
    PILOT_CHECKSUM_SCHEMA_VERSION,
    PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
    TERMINAL_STATUSES,
    PilotEvidenceError,
    PilotEvidencePackage,
    _aggregate_csv,
    _claims,
    _experiment_a_gate,
    _experiment_b_summary,
    _experiment_c_gate,
    _experiment_d_gate,
    _json_copy,
    _load_completed_artifact,
    _method_scaffold,
    _metric,
    _narrative_gate,
    _sha256_file,
    _strict_json_load,
)
from .pilot_orchestrator import (
    PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION,
    PilotOrchestrationError,
    PilotRunLedger,
    _assert_v21110_local_release_guard,
    _budget_caps,
    _load_v2_terminal_summary,
)
from .pilot_v2112_evidence import _validate_release, _validate_stage_receipts
from .pilot_v2115_evidence import (
    V2115_CONTRACT_ID,
    V2115_CONTRACT_RELATIVE,
    V2115_RAW_RELATIVE,
    _normalize_v2115_partial_ledger,
    _normalized_stage_receipts,
    _v2115_capability_by_model,
    _install_package_no_overwrite,
    _resolve_source_paths as _resolve_v2115_source_paths,
    _validate_offline_candidate_admission,
    _validated_post_gate,
    _validate_source_git as _validate_v2115_source_git,
    _validated_acceptance_and_budget as _validated_v2115_acceptance_and_budget,
)
from .pilot_v21110_continuation import (
    V21110_ACCEPTANCE_FILENAME,
    V21110_CONTRACT_ID,
    V21110_RAW_ROOT,
    V21110_SCIENCE_TAG,
    V21110_SOURCE_MANIFEST_PATH,
    V2119_CONTRACT_ID,
    V2119_CONTRACT_PATH,
    V2119_CONTRACT_SHA256,
    V2119_FAILED_RAW_ROOT,
    V2119_SOURCE_MANIFEST_PATH,
    _real_root,
    _require_distinct_roots,
    audit_v21110_scientific_stage_namespace,
    current_authority_path,
    parent_budget_debit_for_v21110,
    require_v21110_provider_keys_absent,
    validate_v21110_source_manifest,
    verify_v2119_terminal_no_go,
    verify_v21110_current_authority,
    verify_v21110_parent_import_receipt,
    verify_v21110_scientific_dispatch_acceptance,
    verify_v21110_terminal_scientific_artifacts,
    verified_v21110_projection,
    _verified_parent_import_budget_actual,
    _verify_current_accepted_budget_rows,
)


V21110_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.11.10-evidence-package-v1"
V21110_DENOMINATOR_SCHEMA_VERSION = "finevo-pilot-v2.11.10-logical-denominator-v1"
V21110_CONTROLS_SCHEMA_VERSION = "finevo-pilot-v2.11.10-release-controls-v1"
V21110_RUN_LEDGER_AUDIT_SCHEMA_VERSION = "finevo-pilot-v2.11.10-run-ledger-audit-v1"
V21110_BUDGET_AUDIT_SCHEMA_VERSION = "finevo-pilot-v2.11.10-budget-audit-v1"
V21110_FAILED_LINEAGE_AUDIT_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.10-v2.11.9-failed-lineage-audit-v1"
)

CURRENT_STAGE_IDS = ("parent-import", "experiment-d", "experiment-b", "cross-model")
CURRENT_STAGE_COUNTS = {
    "parent-import": 1,
    "experiment-d": 55,
    "experiment-b": 25,
    "cross-model": 6,
}
PARENT_TERMINAL_STAGE_IDS = (
    "parent-import",
    "capability-gate",
    "long-context-preflight",
    "experiment-c",
    "experiment-a",
)
PARENT_TERMINAL_STAGE_COUNTS = {
    "parent-import": 1,
    "capability-gate": 2,
    "long-context-preflight": 2,
    "experiment-c": 25,
    "experiment-a": 20,
}
CURRENT_LEDGER_DENOMINATOR = 87
CURRENT_SCIENTIFIC_DENOMINATOR = 86
PARENT_TERMINAL_DENOMINATOR = 50
LOGICAL_REGISTERED_DENOMINATOR = 136
LOGICAL_SCIENTIFIC_DENOMINATOR = 131
CAPABILITY_MODELS = ("gpt52_main", "gpt56_diagnostic")
SCIENCE_STAGES = frozenset({"experiment-d", "experiment-b", "cross-model"})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotEvidenceError(f"{name} must be an object")
    return value


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_contract_layout(
    current: PilotContract,
    parent: PilotContract,
    *,
    require_frozen: bool,
) -> None:
    if current.contract_id != V21110_CONTRACT_ID:
        raise PilotEvidenceError("V2.11.10 consumer received a different contract")
    if parent.contract_id != V2115_CONTRACT_ID:
        raise PilotEvidenceError("V2.11.10 consumer requires V2.11.5 authority")
    if require_frozen and (current.status != "frozen" or parent.status != "frozen"):
        raise PilotEvidenceError("publication requires both contracts to be frozen")
    if tuple(current.stage_ids) != CURRENT_STAGE_IDS:
        raise PilotEvidenceError("V2.11.10 stage order drifted")
    current_counts = {
        stage_id: len(current.expand(stage=stage_id)) for stage_id in current.stage_ids
    }
    if current_counts != CURRENT_STAGE_COUNTS or sum(current_counts.values()) != 87:
        raise PilotEvidenceError("V2.11.10 current denominator is not 87 cells")
    parent_counts = {
        stage_id: len(parent.expand(stage=stage_id)) for stage_id in parent.stage_ids
    }
    if any(
        parent_counts.get(stage_id) != count
        for stage_id, count in PARENT_TERMINAL_STAGE_COUNTS.items()
    ):
        raise PilotEvidenceError("V2.11.5 terminal-prefix denominator drifted")
    boundary = _mapping(
        current.v21110_recovery_boundary,
        "V2.11.10 recovery boundary",
    )
    failed = _mapping(
        boundary.get("failed_release_no_go"),
        "V2.11.9 immutable failed-release boundary",
    )
    failed_run = _mapping(failed.get("run_ledger"), "V2.11.9 run-ledger boundary")
    failed_budget = _mapping(
        failed.get("budget_ledger"),
        "V2.11.9 budget-ledger boundary",
    )
    failed_actual = _mapping(
        failed_budget.get("current_actual"),
        "V2.11.9 current actual boundary",
    )
    if (
        failed.get("contract_id") != V2119_CONTRACT_ID
        or failed.get("contract_sha256") != V2119_CONTRACT_SHA256
        or failed_run.get("registered_rows") != 87
        or failed_run.get("status_counts") != {"complete": 1, "failed": 86}
        or failed_budget.get("status_counts") != {"complete": 1, "failed": 36}
        or failed_actual
        != {"cost_usd": 0.0, "hosted_completions": 0, "storage_bytes": 800_162}
        or failed.get("provider_construction") is not False
        or failed.get("provider_calls") != 0
        or failed.get("hosted_completions") != 0
        or failed.get("scientific_evidence") is not False
        or failed.get("resume_forbidden") is not True
        or failed.get("failure_reclassification_forbidden") is not True
    ):
        raise PilotEvidenceError("V2.11.9 immutable no-go boundary drifted")
    matrix = _mapping(boundary.get("continuation_matrix"), "continuation matrix")
    expected = {
        "ledger_cells": CURRENT_LEDGER_DENOMINATOR,
        "operational_import_cells": 1,
        "fresh_scientific_cells": CURRENT_SCIENTIFIC_DENOMINATOR,
        "combined_parent_terminal_rows": PARENT_TERMINAL_DENOMINATOR,
        "logical_registered_denominator_after_cross_release_dedup": (
            LOGICAL_REGISTERED_DENOMINATOR
        ),
        "logical_scientific_denominator_after_cross_release_dedup": (
            LOGICAL_SCIENTIFIC_DENOMINATOR
        ),
    }
    if any(matrix.get(key) != value for key, value in expected.items()):
        raise PilotEvidenceError("V2.11.10 cross-release denominator boundary drifted")
    if (
        matrix.get("imported_a_c_remain_parent_evidence") is not True
        or matrix.get("a_c_reclassified_as_v21110") is not False
        or matrix.get("per_row_source_contract_id") != V2115_CONTRACT_ID
    ):
        raise PilotEvidenceError("V2.11.10 A/C authority boundary drifted")


def _expected_mapping(
    current: PilotContract,
    parent: PilotContract,
) -> dict[str, Any]:
    source_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in parent.expand(stage=stage_id)
    )
    child_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in current.expand(stage=stage_id)
    )
    child_by_id = {spec.run_id: spec.to_dict() for spec in child_specs}
    if len(source_specs) != 86 or len(child_by_id) != 86:
        raise PilotEvidenceError("continuation mapping denominator drifted")
    rows: list[dict[str, Any]] = []
    prefix = f"{V2115_CONTRACT_ID}--"
    for source_spec in sorted(source_specs, key=lambda item: item.run_id):
        source = source_spec.to_dict()
        if not source_spec.run_id.startswith(prefix):
            raise PilotEvidenceError("V2.11.5 continuation run id is malformed")
        child = _json_copy(source)
        child["run_id"] = f"{V21110_CONTRACT_ID}--{source_spec.run_id[len(prefix):]}"
        child["contract_id"] = V21110_CONTRACT_ID
        child["budget_bucket"] = "hosted_v21110"
        if child_by_id.get(child["run_id"]) != child:
            raise PilotEvidenceError("V2.11.10 continuation spec mapping drifted")
        logical = _json_copy(source)
        logical.pop("run_id")
        logical.pop("contract_id")
        logical["budget_bucket"] = "normalized-hosted-continuation"
        rows.append(
            {
                "source_run_id": source_spec.run_id,
                "child_run_id": child["run_id"],
                "logical_cell_sha256": canonical_sha256(logical),
                "source_spec_sha256": canonical_sha256(source),
                "child_spec_sha256": canonical_sha256(child),
                "normalized_spec": logical,
            }
        )
    result = {
        "schema_version": "finevo-pilot-v2.11.10-canonical-cell-mapping-v1",
        "row_count": len(rows),
        "mapping_sha256": canonical_sha256(rows),
        "rows": rows,
    }
    declared = current.v21110_recovery_boundary["continuation_matrix"][
        "canonical_86_row_mapping_sha256"
    ]
    if result["mapping_sha256"] != declared:
        raise PilotEvidenceError("computed continuation mapping differs from contract")
    return result


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: PilotContract,
    *,
    stage_ids: Sequence[str],
    require_all_terminal: bool,
) -> dict[str, Mapping[str, Any]]:
    expected_specs = {
        spec.run_id: spec.to_dict()
        for stage_id in stage_ids
        for spec in contract.expand(stage=stage_id)
    }
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or run_id in observed:
            raise PilotEvidenceError("normalized ledger has a missing/duplicate run id")
        observed[run_id] = row
    if set(observed) != set(expected_specs):
        raise PilotEvidenceError(
            "normalized ledger differs from the preregistered denominator"
        )
    for run_id, spec in expected_specs.items():
        row = observed[run_id]
        if any(row.get(key) != value for key, value in spec.items()):
            raise PilotEvidenceError(f"normalized run spec drifted: {run_id}")
        status = row.get("status")
        if require_all_terminal and status not in TERMINAL_STATUSES:
            raise PilotEvidenceError(f"nonterminal ITT row: {run_id}")
        if status == "complete":
            if row.get("artifact_kind") is None:
                raise PilotEvidenceError(
                    f"completed row lacks validated artifact: {run_id}"
                )
            if (
                row["stage_id"] in SCIENCE_STAGES
                and row.get("scientific_eligible") is not True
            ):
                raise PilotEvidenceError(
                    f"completed science row is not eligible: {run_id}"
                )
        elif row.get("scientific_eligible") is True:
            raise PilotEvidenceError(f"failed row became scientific evidence: {run_id}")
    return observed


def inherited_capability_by_model(
    parent_import_receipt: Mapping[str, Any],
    *,
    parent_capability_by_model: Mapping[str, Any],
    parent_preflight_authority: Mapping[str, Any],
    current_authority: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Join capability competence to independent closed-loop eligibility."""

    summaries = _mapping(
        parent_import_receipt.get("capability_authority"),
        "V2.11.10 inherited capability authority",
    )
    if set(summaries) != set(CAPABILITY_MODELS):
        raise PilotEvidenceError("inherited capability model denominator drifted")
    if set(parent_capability_by_model) != set(CAPABILITY_MODELS):
        raise PilotEvidenceError("independent capability denominator drifted")
    dispatch = _mapping(
        parent_import_receipt.get("dispatch_authority_source"),
        "V2.11.10 inherited dispatch authority",
    )
    source_gate = _mapping(dispatch.get("source_gate"), "source gate binding")
    stable = _mapping(
        dispatch.get("stable_source_authorities"), "stable source authorities"
    )
    current_stable = _mapping(
        current_authority.get("stable_source_authorities"),
        "current stable source authorities",
    )
    current_gate = _mapping(
        _mapping(
            current_authority.get("authority_release"),
            "current authority release",
        ).get("source_gate"),
        "current source gate binding",
    )
    if (
        not _is_sha256(source_gate.get("file_sha256"))
        or not _is_sha256(source_gate.get("content_sha256"))
        or canonical_sha256(stable) != canonical_sha256(current_stable)
        or canonical_sha256(source_gate) != canonical_sha256(current_gate)
    ):
        raise PilotEvidenceError("closed-loop dispatch authority binding drifted")
    preflight_denominator = _mapping(
        parent_preflight_authority.get("denominator"),
        "parent preflight denominator",
    )
    preflight_sources = _mapping(
        parent_preflight_authority.get("authority_sources"),
        "parent preflight authority sources",
    )
    operational_imports = _mapping(
        parent_preflight_authority.get("operational_imports"),
        "parent preflight operational imports",
    )
    if (
        parent_preflight_authority.get("available") is not True
        or parent_preflight_authority.get("go") is not True
        or set(preflight_denominator.get("eligible_model_ids", []))
        != set(CAPABILITY_MODELS)
        or set(preflight_sources) != set(CAPABILITY_MODELS)
        or set(operational_imports) != set(CAPABILITY_MODELS)
        or not Path(str(parent_preflight_authority.get("path")))
        .as_posix()
        .endswith("/" + str(source_gate.get("path")))
        or parent_preflight_authority.get("file_sha256")
        != source_gate.get("file_sha256")
        or parent_preflight_authority.get("content_sha256")
        != source_gate.get("content_sha256")
    ):
        raise PilotEvidenceError("preflight authority eligibility drifted")
    output: dict[str, dict[str, Any]] = {}
    expected_denominators = {
        "utility-ranking": 12,
        "rule-application": 12,
        "rule-proposal": 6,
    }
    for model_id in CAPABILITY_MODELS:
        summary = _mapping(summaries[model_id], f"{model_id} capability summary")
        independent_wrapper = _mapping(
            _mapping(
                parent_capability_by_model[model_id],
                f"{model_id} independent capability row",
            ).get("capability"),
            f"{model_id} independent capability wrapper",
        )
        independent_summary = _mapping(
            independent_wrapper.get("capability"),
            f"{model_id} independent capability summary",
        )
        independent_integrity = _mapping(
            independent_wrapper.get("integrity"),
            f"{model_id} independent capability integrity",
        )
        independent_compact = {
            "model_id": model_id,
            "runtime_model": independent_summary.get("runtime_model"),
            "requested_model": independent_summary.get("requested_model"),
            "capability_pass": independent_summary.get("capability_pass"),
            "interface_pass": independent_summary.get("interface_pass"),
            "parse_failure_count": independent_summary.get(
                "parse_failure_count"
            ),
            "provider_failure_count": independent_summary.get(
                "provider_failure_count"
            ),
            "category_totals": _json_copy(
                independent_summary.get("category_totals")
            ),
            "source_wrapper_content_sha256": independent_integrity.get(
                "content_sha256"
            ),
        }
        categories = _mapping(summary.get("category_totals"), "category totals")
        if (
            summary.get("model_id") != model_id
            or summary != independent_compact
            or type(summary.get("capability_pass")) is not bool
            or type(summary.get("interface_pass")) is not bool
            or type(summary.get("parse_failure_count")) is not int
            or type(summary.get("provider_failure_count")) is not int
            or summary.get("source_wrapper_content_sha256")
            != v2117.V2115_CAPABILITY_CONTENT_SHA256[model_id]
            or set(categories) != set(expected_denominators)
            or any(
                not isinstance(categories[name], Mapping)
                or categories[name].get("denominator") != denominator
                for name, denominator in expected_denominators.items()
            )
        ):
            raise PilotEvidenceError(f"inherited capability drifted: {model_id}")
        runtime_model = str(summary["runtime_model"])
        by_kind = _mapping(stable.get(runtime_model), f"{model_id} p95 authority")
        source_preflight = _mapping(
            _mapping(
                preflight_sources.get(model_id),
                f"{model_id} post-gate authority source",
            ).get("source_preflight"),
            f"{model_id} source preflight",
        )
        authority_matched = bool(
            set(by_kind) == {"action", "semantic"}
            and all(
                isinstance(by_kind[kind], Mapping)
                and by_kind[kind].get("source_kind")
                == "sealed-closed-loop-observed-p95"
                and by_kind[kind].get("source_model_id") == model_id
                and source_preflight.get("model_id") == model_id
                and by_kind[kind].get("source_served_model")
                == source_preflight.get("served_model")
                and by_kind[kind].get("source_preflight_run_id")
                == source_preflight.get("run_id")
                and by_kind[kind].get("source_preflight_run_spec_sha256")
                == source_preflight.get("run_spec_sha256")
                and by_kind[kind].get("source_execution_artifact_sha256")
                == source_preflight.get("execution_artifact_sha256")
                and by_kind[kind].get("source_provider_call_journal_sha256")
                == source_preflight.get("provider_call_journal_sha256")
                and all(
                    _is_sha256(source_preflight.get(name))
                    for name in (
                        "run_spec_sha256",
                        "execution_artifact_sha256",
                        "provider_call_journal_sha256",
                    )
                )
                for kind in ("action", "semantic")
            )
            and operational_imports[model_id].get("provider_calls_current_attempt") == 0
            and operational_imports[model_id].get(
                "provider_construction_current_attempt"
            )
            is False
        )
        if not authority_matched:
            raise PilotEvidenceError(
                f"closed-loop preflight authority drifted: {model_id}"
            )
        capability_passed = bool(
            summary["capability_pass"] is True and summary["interface_pass"] is True
        )
        passed = bool(capability_passed and authority_matched)
        output[model_id] = {
            "source": (
                "inherited V2.11.5 capability plus independently sealed "
                "closed-loop preflight authority"
            ),
            "ledger_status": "complete-imported-authority",
            "artifact_validated": True,
            "capability": {
                "capability": _json_copy(summary),
                "preflight_go": authority_matched,
                "preflight_interface_pass": authority_matched,
                "provider_calls_current_attempt": 0,
                "provider_construction_current_attempt": False,
                "scientific_evidence": False,
            },
            "capability_and_preflight_pass": passed,
            "closed_loop_preflight_authority_matched": authority_matched,
            "provider_calls_current_attempt": 0,
            "provider_construction_current_attempt": False,
            "scientific_evidence": False,
            "claim_boundary": (
                "historical capability and interface authority only; zero "
                "V2.11.10 capability/preflight provider calls"
            ),
        }
    return output


def _cross_model_summary_v21110(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    capability: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate only the two models actually registered by V2.11.10."""

    expected_seeds = tuple(int(seed) for seed in contract.seeds["sets"]["cross-model"])
    source_stages = {"gpt52_main": "experiment-b", "gpt56_diagnostic": "cross-model"}
    if set(capability) != set(source_stages):
        raise PilotEvidenceError("cross-model capability denominator drifted")
    output: dict[str, Any] = {}
    for model_id, stage_id in source_stages.items():
        registered = [
            row
            for row in rows
            if row.get("stage_id") == stage_id
            and row.get("model_id") == model_id
            and row.get("arm_id") in {"full", "no-memory"}
            and row.get("environment_seed") in expected_seeds
        ]
        if len(registered) != 6:
            raise PilotEvidenceError(f"{model_id} cross-model denominator is not six")
        by_arm = {
            arm: {
                int(row["environment_seed"]): row
                for row in registered
                if row["arm_id"] == arm
                and row["status"] == "complete"
                and row["scientific_eligible"] is True
            }
            for arm in ("full", "no-memory")
        }
        usable = [
            seed
            for seed in expected_seeds
            if seed in by_arm["full"]
            and seed in by_arm["no-memory"]
            and _metric(by_arm["full"][seed], "utility.shock_recovery_discounted")
            is not None
            and _metric(by_arm["no-memory"][seed], "utility.shock_recovery_discounted")
            is not None
        ]
        delta = None
        if usable:
            delta = paired_delta_summary(
                {
                    seed: float(
                        _metric(
                            by_arm["full"][seed],
                            "utility.shock_recovery_discounted",
                        )
                    )
                    for seed in usable
                },
                {
                    seed: float(
                        _metric(
                            by_arm["no-memory"][seed],
                            "utility.shock_recovery_discounted",
                        )
                    )
                    for seed in usable
                },
            )
        raw = (
            {
                int(seed): float(value)
                for seed, value in delta["raw_paired_deltas"].items()
            }
            if delta is not None
            else {}
        )
        positive = len(raw) == 3 and all(value > 0 for value in raw.values())
        negative = len(raw) == 3 and all(value < 0 for value in raw.values())
        capability_summary = capability[model_id]
        capability_payload = capability_summary["capability"]["capability"]
        capability_ok = capability_summary["capability_and_preflight_pass"] is True
        replicated = bool(capability_ok and (positive or negative))
        output[model_id] = {
            "source": (
                "V2.11.10 experiment-b first-three preregistered seeds"
                if model_id == "gpt52_main"
                else "V2.11.10 cross-model"
            ),
            "registered_pair_count": 3,
            "usable_paired_seeds": usable,
            "registered_seed_status_and_failures": {
                f"{row['arm_id']}:{row['environment_seed']}": {
                    "status": row["status"],
                    "failure": _json_copy(row.get("failure")),
                    "provider_failure_count": _metric(
                        row, "guardrails.provider_failure_count"
                    ),
                }
                for row in sorted(
                    registered,
                    key=lambda value: (value["arm_id"], value["environment_seed"]),
                )
            },
            "capability_and_preflight_pass": capability_ok,
            "utility_ranking_competence": _json_copy(
                capability_payload["category_totals"]["utility-ranking"]
            ),
            "rule_application_competence": _json_copy(
                capability_payload["category_totals"]["rule-application"]
            ),
            "proposal_competence": _json_copy(
                capability_payload["category_totals"]["rule-proposal"]
            ),
            "capability_parse_failure_count": capability_payload["parse_failure_count"],
            "capability_provider_failure_count": capability_payload[
                "provider_failure_count"
            ],
            "paired_delta": delta,
            "direction": (
                "positive"
                if positive
                else "negative" if negative else "mixed-or-incomplete"
            ),
            "directional_micro_pilot_replication": replicated,
            "seed_dispatch_mode": "documented_unsupported_omitted",
            "matched_a_a_null_registered": False,
            "matched_null_resolution": "directional-only-no-model-specific-repeatability-null",
            "repeatability_or_effect_size_claim_allowed": False,
            "claim_boundary": (
                "direction replicated in this model-family micro-pilot only; "
                "no model-specific repeatability or effect-size claim"
                if replicated
                else "no cross-model effectiveness claim"
            ),
        }
    return output


def _status_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["status"]) for row in rows)
    stage_counts = {
        stage_id: dict(
            sorted(
                Counter(
                    str(row["status"]) for row in rows if row["stage_id"] == stage_id
                ).items()
            )
        )
        for stage_id in dict.fromkeys(str(row["stage_id"]) for row in rows)
    }
    completed_valid = all(
        row.get("artifact_kind") is not None
        for row in rows
        if row["status"] == "complete"
    )
    return {
        "row_count": len(rows),
        "all_rows_terminal": all(row["status"] in TERMINAL_STATUSES for row in rows),
        "status_counts": dict(sorted(counts.items())),
        "stage_status_counts": stage_counts,
        "itt_failures_retained": sum(
            count for status, count in counts.items() if status != "complete"
        ),
        "all_completed_artifacts_validated": completed_valid,
    }


def _expected_failed_lineage_audit(contract: PilotContract) -> dict[str, Any]:
    """Return the only admissible V2.11.9 lineage-only audit projection."""

    boundary = _mapping(
        contract.to_dict().get("v21110_recovery_boundary"),
        "V2.11.10 recovery boundary",
    )
    failed = _mapping(
        boundary.get("failed_release_no_go"),
        "V2.11.9 immutable failed release",
    )
    return {
        "schema_version": V21110_FAILED_LINEAGE_AUDIT_SCHEMA_VERSION,
        "contract_id": failed.get("contract_id"),
        "contract_sha256": failed.get("contract_sha256"),
        "contract_file_sha256": failed.get("contract_file_sha256"),
        "science_tag": failed.get("science_tag"),
        "science_tag_object": failed.get("science_tag_object"),
        "science_commit": failed.get("science_commit"),
        "source_manifest_file_sha256": failed.get(
            "source_manifest_file_sha256"
        ),
        "source_manifest_content_sha256": failed.get(
            "source_manifest_content_sha256"
        ),
        "run_ledger": _json_copy(failed.get("run_ledger")),
        "budget_ledger": _json_copy(failed.get("budget_ledger")),
        "stage_receipts": _json_copy(failed.get("stage_receipts")),
        "raw_inventory": _json_copy(failed.get("raw_inventory")),
        "complete_raw_inventory": _json_copy(
            failed.get("complete_raw_inventory")
        ),
        "scientific_dispatch_acceptance": _json_copy(
            failed.get("scientific_dispatch_acceptance")
        ),
        "release_attestation": _json_copy(failed.get("release_attestation")),
        "scientific_launch_input": _json_copy(
            failed.get("scientific_launch_input")
        ),
        "failure_profile": _json_copy(failed.get("failure_profile")),
        "provider_construction": False,
        "provider_calls": 0,
        "hosted_completions": 0,
        "registered_rows": 87,
        "operational_complete_rows": 1,
        "scientific_failed_rows": 86,
        "failed_rows_imported_into_logical_denominator": 0,
        "failed_effects_imported": 0,
        "failed_statuses_preserved": True,
        "resume_forbidden": True,
        "failure_reclassification_forbidden": True,
        "terminal_verification_pass": True,
        "scientific_evidence": False,
    }


def _validate_failed_lineage_audit(
    value: Mapping[str, Any],
    *,
    contract: PilotContract,
) -> dict[str, Any]:
    expected = _expected_failed_lineage_audit(contract)
    if _json_copy(value) != expected:
        raise PilotEvidenceError("V2.11.9 terminal no-go lineage audit drifted")
    return expected


def assemble_v21110_terminal_evidence(
    *,
    contract: PilotContract,
    parent_contract: PilotContract,
    parent_terminal_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    parent_import_receipt: Mapping[str, Any],
    parent_capability_by_model: Mapping[str, Any],
    parent_preflight_authority: Mapping[str, Any],
    current_authority: Mapping[str, Any],
    external_parent_gates: Mapping[str, Mapping[str, Any]],
    failed_release_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure terminal assembler used by publication and provider-free fixtures."""

    _require_contract_layout(contract, parent_contract, require_frozen=False)
    failed_lineage = _validate_failed_lineage_audit(
        failed_release_audit,
        contract=contract,
    )
    parent_by_id = _validate_rows(
        parent_terminal_rows,
        parent_contract,
        stage_ids=PARENT_TERMINAL_STAGE_IDS,
        require_all_terminal=True,
    )
    current_by_id = _validate_rows(
        current_rows,
        contract,
        stage_ids=CURRENT_STAGE_IDS,
        require_all_terminal=True,
    )
    mapping = _expected_mapping(contract, parent_contract)
    imported_mapping = parent_import_receipt.get("canonical_remaining_cell_mapping")
    if imported_mapping != mapping:
        raise PilotEvidenceError("verified parent receipt continuation mapping drifted")
    if set(external_parent_gates) != {"experiment_a", "experiment_c"}:
        raise PilotEvidenceError("external parent gate denominator drifted")
    experiment_a = _json_copy(external_parent_gates["experiment_a"])
    experiment_c = _json_copy(external_parent_gates["experiment_c"])
    frozen_receipts = _mapping(
        _mapping(
            contract.to_dict()["v21110_recovery_boundary"],
            "V2.11.10 frozen recovery boundary",
        ).get("parent_stage_receipts"),
        "frozen parent stage receipts",
    )
    if (
        experiment_a.get("status") != "no-go"
        or experiment_a.get("support_retrieval_effect") is not False
        or experiment_c.get("status") != "no-go"
        or experiment_c.get("support_rule_reliability") is not False
        or experiment_a.get("authority_binding")
        != _json_copy(frozen_receipts.get("experiment-a"))
        or experiment_c.get("authority_binding")
        != _json_copy(frozen_receipts.get("experiment-c"))
    ):
        raise PilotEvidenceError("parent A/C must remain external no-go evidence")

    current_parent_ids = {
        spec.run_id for spec in contract.expand(stage="parent-import")
    }
    current_science = [
        current_by_id[spec.run_id]
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in contract.expand(stage=stage_id)
    ]
    if any(row["run_id"] in current_parent_ids for row in current_science):
        raise PilotEvidenceError("current parent row leaked into logical denominator")
    logical_rows = [
        parent_by_id[spec.run_id]
        for stage_id in PARENT_TERMINAL_STAGE_IDS
        for spec in parent_contract.expand(stage=stage_id)
    ] + current_science
    if len(logical_rows) != LOGICAL_REGISTERED_DENOMINATOR:
        raise PilotEvidenceError("logical denominator is not 136 rows")
    if len({row["run_id"] for row in logical_rows}) != len(logical_rows):
        raise PilotEvidenceError("logical denominator contains duplicate run ids")
    if any(
        row.get("contract_id") not in {V2115_CONTRACT_ID, V21110_CONTRACT_ID}
        for row in logical_rows
    ):
        raise PilotEvidenceError(
            "V2.11.9 failure-lineage row leaked into the logical denominator"
        )

    current_audit = _status_audit(list(current_by_id.values()))
    logical_audit = _status_audit(logical_rows)
    parent_audit = _status_audit(list(parent_by_id.values()))
    denominator = {
        "schema_version": V21110_DENOMINATOR_SCHEMA_VERSION,
        "failed_v2119_lineage": {
            "registered_rows": failed_lineage["registered_rows"],
            "operational_complete_rows": failed_lineage[
                "operational_complete_rows"
            ],
            "scientific_failed_rows": failed_lineage["scientific_failed_rows"],
            "logical_rows_imported": failed_lineage[
                "failed_rows_imported_into_logical_denominator"
            ],
            "effects_imported": failed_lineage["failed_effects_imported"],
            "audit_only": True,
            "pass": True,
        },
        "current_release": {
            **current_audit,
            "expected_count": CURRENT_LEDGER_DENOMINATOR,
            "scientific_row_count": CURRENT_SCIENTIFIC_DENOMINATOR,
            "operational_parent_row_count": 1,
            "pass": bool(
                current_audit["row_count"] == CURRENT_LEDGER_DENOMINATOR
                and current_audit["all_rows_terminal"]
                and current_audit["all_completed_artifacts_validated"]
            ),
        },
        "parent_terminal_prefix": {
            **parent_audit,
            "expected_count": PARENT_TERMINAL_DENOMINATOR,
            "pass": bool(
                parent_audit["row_count"] == PARENT_TERMINAL_DENOMINATOR
                and parent_audit["all_rows_terminal"]
                and parent_audit["all_completed_artifacts_validated"]
            ),
        },
        "logical_v2115_matrix": {
            **logical_audit,
            "expected_count": LOGICAL_REGISTERED_DENOMINATOR,
            "scientific_row_count": LOGICAL_SCIENTIFIC_DENOMINATOR,
            "operational_row_count": 5,
            "parent_terminal_rows": PARENT_TERMINAL_DENOMINATOR,
            "continuation_rows": CURRENT_SCIENTIFIC_DENOMINATOR,
            "current_parent_meta_row_excluded": True,
            "mapping_sha256": mapping["mapping_sha256"],
            "pass": bool(
                logical_audit["row_count"] == LOGICAL_REGISTERED_DENOMINATOR
                and logical_audit["all_rows_terminal"]
                and logical_audit["all_completed_artifacts_validated"]
            ),
        },
    }
    capability = inherited_capability_by_model(
        parent_import_receipt,
        parent_capability_by_model=parent_capability_by_model,
        parent_preflight_authority=parent_preflight_authority,
        current_authority=current_authority,
    )
    experiment_b = _experiment_b_summary(current_science)
    experiment_d = _experiment_d_gate(contract, current_science)
    narrative = _narrative_gate(contract, current_science)
    cross_model = _cross_model_summary_v21110(contract, current_science, capability)
    gates = {
        "experiment_a": experiment_a,
        "experiment_c": experiment_c,
        "experiment_d": experiment_d,
        "narrative": narrative,
    }
    all_claims_supported = bool(
        experiment_a.get("status") == "supported"
        and experiment_c.get("status") == "supported"
        and experiment_d.get("status") == "supported"
        and narrative.get("status") == "supported"
    )
    return {
        "schema_version": V21110_EVIDENCE_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "authority_contract_id": parent_contract.contract_id,
        "authority_contract_sha256": parent_contract.canonical_hash,
        "failed_release_lineage": failed_lineage,
        "denominator": denominator,
        "claim_gates": gates,
        "experiment_b": experiment_b,
        "cross_model": cross_model,
        "inherited_capability": capability,
        "publication_status": {
            "terminal_evidence_complete": bool(
                denominator["failed_v2119_lineage"]["pass"]
                and denominator["current_release"]["pass"]
                and denominator["parent_terminal_prefix"]["pass"]
                and denominator["logical_v2115_matrix"]["pass"]
            ),
            "all_preregistered_claims_supported": all_claims_supported,
            "classification": (
                "terminal-complete-with-preregistered-no-go"
                if not all_claims_supported
                else "terminal-complete-with-supported-gates"
            ),
            "negative_results_retained": True,
        },
        "logical_rows": _json_copy(logical_rows),
    }


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotEvidenceError(
            f"git provenance check failed: {' '.join(args)}"
        ) from exc


def _resolve_current_paths(
    *,
    source_repo_root: str | Path | None,
    contract_path: str | Path,
    raw_root: str | Path,
    run_ledger_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    if source_repo_root is None:
        raise PilotEvidenceError("V2.11.10 publication requires --source-repo-root")
    supplied_source = Path(os.path.abspath(Path(source_repo_root)))
    if supplied_source.is_symlink():
        raise PilotEvidenceError("V2.11.10 source repository cannot be a symlink")
    source = supplied_source.resolve(strict=True)
    if supplied_source != source:
        raise PilotEvidenceError(
            "V2.11.10 source repository cannot cross a symlink component"
        )
    expected_contract = source / "experiments/pilot_v2_11_10.yaml"
    expected_raw = source.joinpath(*V21110_RAW_ROOT.parts)
    expected_ledger = expected_raw / "run_ledger.json"
    supplied = tuple(
        Path(path).absolute() for path in (contract_path, raw_root, run_ledger_path)
    )
    if supplied != (expected_contract, expected_raw, expected_ledger):
        raise PilotEvidenceError("V2.11.10 publication paths must be in-place and exact")
    for path in (expected_contract, expected_raw, expected_ledger):
        if path.is_symlink() or not path.exists() or path.resolve() != path.absolute():
            raise PilotEvidenceError("V2.11.10 publication path is missing or unsafe")
    return source, expected_contract, expected_raw, expected_ledger


def _validate_current_git(
    source: Path,
    contract: PilotContract,
    commit: str,
    *,
    release_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    tag = str(contract.implementation["required_git_tag"])
    tag_ref = f"refs/tags/{tag}"
    tag_object = _git(source, "rev-parse", tag_ref)
    local_tag = release_attestation.get("local_tag")
    if (
        Path(_git(source, "rev-parse", "--show-toplevel")).resolve() != source
        or tag != V21110_SCIENCE_TAG
        or _git(source, "rev-parse", "HEAD") != commit
        or _git(source, "cat-file", "-t", tag_ref) != "tag"
        or _git(source, "rev-parse", f"{tag_ref}^{{commit}}") != commit
        or _git(source, "status", "--porcelain", "--untracked-files=all")
        or not isinstance(local_tag, Mapping)
        or set(local_tag) != {"name", "object_id", "peeled_commit", "kind"}
        or local_tag.get("name") != tag
        or local_tag.get("object_id") != tag_object
        or local_tag.get("peeled_commit") != commit
        or local_tag.get("kind") != "annotated"
    ):
        raise PilotEvidenceError(
            "V2.11.10 source is not its tracked-clean annotated tag"
        )
    return {
        "source_repo_root": str(source),
        "git_tag": tag,
        "annotated_tag_object_id": tag_object,
        "resolved_git_commit": commit,
        "annotated_tag": True,
        "tracked_worktree_clean": True,
    }


def _artifact_path(raw_root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise PilotEvidenceError("terminal row lacks an artifact path")
    path = Path(value)
    if not path.is_absolute():
        path = raw_root / path
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(raw_root.resolve())
    except (FileNotFoundError, ValueError) as exc:
        raise PilotEvidenceError("artifact is missing or escapes raw root") from exc
    if path.is_symlink() or not resolved.is_file():
        raise PilotEvidenceError("artifact is not a regular in-root file")
    return resolved


def _failure_artifact_evidence(raw_root: Path, artifact: Any) -> dict[str, Any]:
    """Represent a replayed terminal failure without inventing an artifact."""

    if artifact is None:
        return {
            "artifact_kind": "terminal-failure-without-artifact",
            "artifact_sha256": None,
        }
    path = _artifact_path(raw_root, artifact)
    return {
        "artifact_kind": "failure-audit-artifact",
        "artifact_sha256": _sha256_file(path),
    }


def _expected_parent_payload(
    parent: Mapping[str, Any],
    authority: Mapping[str, Any],
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "metrics": {},
        "gate_evidence": {
            "parent_import_content_sha256": parent["integrity"]["content_sha256"],
            "current_authority_content_sha256": authority["integrity"][
                "content_sha256"
            ],
            "projection_content_sha256_by_model": {
                model_id: projections[model_id]["integrity"]["content_sha256"]
                for model_id in CAPABILITY_MODELS
            },
            "failed_v2119_terminal_rows_bound": 87,
            "mapped_v2115_scheduled_rows": 86,
        },
        "provider_construction": False,
        "provider_calls": 0,
        "imported_effect_cells": 0,
        "failed_terminal_rows_imported_as_child_rows": 0,
        "authority_terminal_rows_imported_as_child_rows": 0,
        "claim_boundary": parent["claim_boundary"],
    }


def _normalize_current_ledger(
    contract: PilotContract,
    *,
    raw_root: Path,
    source_repo_root: Path,
    ledger: PilotRunLedger,
    paid: Any,
    parent: Mapping[str, Any],
    authority: Mapping[str, Any],
    projections: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terminal_replay = verify_v21110_terminal_scientific_artifacts(
        contract,
        repo_root=source_repo_root,
        raw_root=raw_root,
        run_ledger=ledger,
        paid=paid,
    )
    snapshot = ledger.snapshot()
    observed = _mapping(snapshot.get("runs"), "V2.11.10 run-ledger rows")
    expected = {spec.run_id: spec for spec in contract.expand()}
    if set(observed) != set(expected):
        raise PilotEvidenceError("V2.11.10 run ledger is not the exact 87-cell matrix")
    rows: list[dict[str, Any]] = []
    for run_id, spec in expected.items():
        source = _mapping(observed[run_id], f"run row {run_id}")
        if (
            source.get("spec") != spec.to_dict()
            or source.get("status") not in TERMINAL_STATUSES
        ):
            raise PilotEvidenceError(f"V2.11.10 ledger spec/status drifted: {run_id}")
        row: dict[str, Any] = {
            **spec.to_dict(),
            "status": source["status"],
            "failure": _json_copy(source.get("failure")),
            "artifact_kind": None,
            "artifact_sha256": None,
            "scientific_eligible": False,
            "metrics": {},
            "gate_evidence": {},
            "capability": {},
            "narrative": {},
        }
        if source["status"] == "complete" and spec.stage_id in SCIENCE_STAGES:
            loaded = _load_completed_artifact(
                contract,
                spec.to_dict(),
                raw_root=raw_root,
                artifact=source.get("artifact"),
                source_repo_root=source_repo_root,
            )
            if loaded.get("scientific_eligible") is not True:
                raise PilotEvidenceError(
                    f"completed science row is ineligible: {run_id}"
                )
            row.update(loaded)
        elif source["status"] == "complete":
            ledger.verify_terminal_artifact_binding(run_id)
            path = _artifact_path(raw_root, source.get("artifact"))
            try:
                terminal = _load_v2_terminal_summary(
                    contract, spec, path, raw_root=raw_root, paid=paid
                )
            except PilotOrchestrationError as exc:
                raise PilotEvidenceError(
                    f"parent terminal replay failed: {exc}"
                ) from exc
            if (
                terminal.get("schema_version") != PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION
                or terminal.get("evidence_scope")
                != "preregistered_dual_root_authority_import"
                or terminal.get("scientific_evidence") is not False
                or terminal.get("diagnostic_only") is not False
                or terminal.get("payload")
                != _expected_parent_payload(parent, authority, projections)
            ):
                raise PilotEvidenceError("V2.11.10 parent terminal payload drifted")
            row.update(
                {
                    "artifact_kind": "terminal-summary",
                    "artifact_sha256": _sha256_file(path),
                    "gate_evidence": _json_copy(terminal["payload"]["gate_evidence"]),
                }
            )
        else:
            row.update(_failure_artifact_evidence(raw_root, source.get("artifact")))
        rows.append(row)
    _validate_rows(
        rows, contract, stage_ids=CURRENT_STAGE_IDS, require_all_terminal=True
    )
    events = snapshot.get("events")
    if not isinstance(events, list) or not events:
        raise PilotEvidenceError("V2.11.10 run-ledger events are absent")
    audit = {
        "schema_version": V21110_RUN_LEDGER_AUDIT_SCHEMA_VERSION,
        "path": str(raw_root / "run_ledger.json"),
        "file_sha256": _sha256_file(raw_root / "run_ledger.json"),
        "ledger_sha256": snapshot.get("ledger_sha256"),
        "event_count": len(events),
        "event_chain_head": events[-1].get("event_sha256"),
        "event_type_counts": dict(
            sorted(Counter(event.get("event_type") for event in events).items())
        ),
        "terminal_artifact_replay": _json_copy(terminal_replay),
    }
    return rows, audit


def _budget_owner_mapping(contract: PilotContract) -> dict[str, tuple[str, ...]]:
    """Map the 37 terminal budget units onto all 87 current ITT rows."""

    parent = tuple(contract.expand(stage="parent-import"))
    if len(parent) != 1:
        raise PilotEvidenceError("V2.11.10 parent budget owner drifted")
    owners: dict[str, tuple[str, ...]] = {parent[0].run_id: (parent[0].run_id,)}
    for stage_id in ("experiment-b", "cross-model"):
        for spec in contract.expand(stage=stage_id):
            owners[spec.run_id] = (spec.run_id,)
    d_specs = tuple(contract.expand(stage="experiment-d"))
    for model_id, seed in sorted(
        {(spec.model_id, spec.environment_seed) for spec in d_specs}
    ):
        group_id = (
            f"{contract.contract_id}--experiment-d--{model_id}--"
            f"checkpoint-group--s{seed}"
        )
        linked = tuple(
            spec.run_id
            for spec in d_specs
            if spec.model_id == model_id and spec.environment_seed == seed
        )
        if len(linked) != 11:
            raise PilotEvidenceError("Experiment D budget group is not 11 ITT cells")
        owners[group_id] = linked
    linked_ids = [run_id for values in owners.values() for run_id in values]
    expected_ids = {spec.run_id for spec in contract.expand()}
    if (
        len(owners) != 37
        or len(linked_ids) != 87
        or len(set(linked_ids)) != 87
        or set(linked_ids) != expected_ids
    ):
        raise PilotEvidenceError("V2.11.10 budget-to-ITT ownership drifted")
    return owners


def _exact_single_run_recovery(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("status") == "integrity-stopped"
        and row.get("artifact") is None
        and row.get("failure")
        == {
            "error_type": "BudgetFinalizedBeforeITT",
            "message": (
                "a prior process created budget state without a terminal ITT "
                "cell; the cell is retained and is not redispatched"
            ),
        }
        and ("artifact_binding" not in row or row.get("artifact_binding") is None)
    )


def _exact_d_recovery(
    row: Mapping[str, Any],
    *,
    model_id: str,
    environment_seed: int,
) -> bool:
    return bool(
        row.get("status") == "integrity-stopped"
        and row.get("artifact") is None
        and row.get("failure")
        == {
            "error_type": "BudgetFinalizedBeforeITT",
            "message": (
                "a prior process created shared Experiment D budget state "
                "without an exact terminal ITT group; no redispatch is permitted"
            ),
            "model_id": model_id,
            "environment_seed": environment_seed,
            "provider_dispatch_started": False,
            "stop_origin": "pre-catalog-interrupted-reservation-recovery",
        }
        and ("artifact_binding" not in row or row.get("artifact_binding") is None)
    )


def _exact_interrupted_d_budget_failure(
    failure: Any,
    *,
    model_id: str,
    environment_seed: int,
) -> bool:
    return bool(
        failure
        == {
            "error_type": "InterruptedReservation",
            "message": (
                "a prior process created shared Experiment D budget state "
                "without an exact terminal ITT group; no redispatch is permitted"
            ),
            "model_id": model_id,
            "environment_seed": environment_seed,
            "provider_dispatch_started": False,
            "stop_origin": "pre-catalog-interrupted-reservation-recovery",
            "accounting_basis": "unreconciled-conservative-reservation",
        }
    )


def _observed_owner_linkage(
    contract: PilotContract,
    *,
    budget_id: str,
    budget_row: Mapping[str, Any],
    linked_ids: Sequence[str],
    run_rows: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify linkage after continuation's strict budget/ITT replay passed."""

    linked = tuple(
        _mapping(run_rows[run_id], f"ITT row {run_id}") for run_id in linked_ids
    )
    budget_status = str(budget_row.get("status"))
    budget_failure = budget_row.get("failure")
    failure_type = (
        budget_failure.get("error_type")
        if isinstance(budget_failure, Mapping)
        else None
    )
    after_itt = bool(
        "--experiment-d--" not in budget_id
        and failure_type == "InterruptedReservationAfterITT"
    )
    direct = tuple(
        row
        for row in linked
        if not after_itt
        and row.get("status") == budget_status
        and row.get("failure") == budget_failure
    )
    is_d = "--experiment-d--" in budget_id
    if is_d:
        matching_specs = [
            spec
            for spec in contract.expand(stage="experiment-d")
            if spec.run_id in linked_ids
        ]
        identities = {
            (str(spec.model_id), int(spec.environment_seed)) for spec in matching_specs
        }
        if len(identities) != 1:
            raise PilotEvidenceError(f"D budget owner identity drifted: {budget_id}")
        model_id, seed = next(iter(identities))
        recovered = tuple(
            row
            for row in linked
            if _exact_d_recovery(
                row,
                model_id=model_id,
                environment_seed=seed,
            )
        )
        uncovered = tuple(
            row for row in linked if row not in direct and row not in recovered
        )
        interrupted_original = tuple()
        if _exact_interrupted_d_budget_failure(
            budget_failure,
            model_id=model_id,
            environment_seed=seed,
        ):
            interrupted_original = tuple(
                row
                for row in uncovered
                if row.get("status") in TERMINAL_STATUSES
                and row.get("status") != "scheduled"
                and (
                    (
                        row.get("status") == "complete"
                        and row.get("failure") is None
                        and row.get("artifact") is not None
                    )
                    or (
                        row.get("status") != "complete"
                        and isinstance(row.get("failure"), Mapping)
                    )
                )
            )
            if len(interrupted_original) > 1:
                raise PilotEvidenceError(
                    f"D interrupted owner has multiple original terminals: {budget_id}"
                )
    else:
        recovered = tuple(row for row in linked if _exact_single_run_recovery(row))
        interrupted_original = tuple()
    original_after_itt = tuple(
        row for row in linked if after_itt and row.get("status") in TERMINAL_STATUSES
    )
    if len(direct) + len(recovered) + len(interrupted_original) + len(
        original_after_itt
    ) != len(linked):
        raise PilotEvidenceError(
            f"budget owner/ITT terminal linkage drifted: {budget_id}"
        )
    return {
        "classification": (
            "original-terminal-after-budget-recovery"
            if original_after_itt
            else (
                "interrupted-reservation-recovery"
                if failure_type == "InterruptedReservation"
                else (
                    "finalized-before-itt-recovery" if recovered else "direct-terminal"
                )
            )
        ),
        "direct_terminal_count": len(direct),
        "exact_recovery_count": len(recovered),
        "pre_reservation_original_terminal_count": len(interrupted_original),
        "original_terminal_after_budget_recovery_count": len(original_after_itt),
        "linked_status_counts": dict(
            sorted(Counter(str(row.get("status")) for row in linked).items())
        ),
    }


def _absent_owner_stop(
    contract: PilotContract,
    *,
    budget_id: str,
    linked_ids: Sequence[str],
    run_rows: Mapping[str, Any],
    observed_budget_rows: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify that an unreserved owner is represented by exact ITT no-go rows."""

    linked = tuple(
        _mapping(run_rows[run_id], f"ITT row {run_id}") for run_id in linked_ids
    )
    stages = {str(row.get("stage_id")) for row in linked}
    models = {str(row.get("model_id")) for row in linked}
    if len(stages) != 1 or len(models) != 1:
        raise PilotEvidenceError(f"absent budget owner identity drifted: {budget_id}")
    stage_id = next(iter(stages))
    model_id = next(iter(models))
    stage_order = {value: index for index, value in enumerate(contract.stage_ids)}
    known_run_ids_by_stage = {
        candidate: {spec.run_id for spec in contract.expand(stage=candidate)}
        for candidate in contract.stage_ids
    }
    d_identities = {
        (spec.model_id, spec.environment_seed)
        for spec in contract.expand(stage="experiment-d")
    }

    def text(value: Any) -> bool:
        return isinstance(value, str) and bool(value)

    def exact_base(
        failure: Mapping[str, Any],
        *,
        source_stage: str,
        status: Any,
    ) -> str | None:
        keys = set(failure)
        if (
            failure
            == {
                "error_type": "StageExecutionNoGo",
                "message": "Experiment D contains a budget or integrity hard stop",
            }
            and source_stage == "experiment-d"
        ):
            return "ancestor-stage-execution-no-go"
        if (
            failure
            == {
                "error_type": "StageExecutionNoGo",
                "message": (
                    "experiment-b contains a budget/integrity hard stop or "
                    "lacks its mandatory pre-science selection"
                ),
            }
            and source_stage == "experiment-b"
        ):
            return "ancestor-stage-execution-no-go"
        if (
            keys == {"error_type", "cause_type", "message"}
            and failure.get("error_type") == "PrerequisiteNoGo"
            and text(failure.get("cause_type"))
            and text(failure.get("message"))
            and status == "integrity-stopped"
        ):
            return "prerequisite-no-go"
        if (
            failure.get("error_type") == "PilotBudgetError"
            and keys == {"error_type", "message"}
            and text(failure.get("message"))
            and source_stage == stage_id
            and status == "budget-stopped"
        ):
            return "per-spec-pre-dispatch-budget-stop"
        if (
            status == "integrity-stopped"
            and failure
            in (
                {
                    "error_type": "BudgetFinalizedBeforeITT",
                    "message": (
                        "a prior process created budget state without a terminal "
                        "ITT cell; the cell is retained and is not redispatched"
                    ),
                },
                {
                    "error_type": "InterruptedReservation",
                    "message": (
                        "a prior process created budget state without a terminal "
                        "ITT cell; the cell is retained and is not redispatched"
                    ),
                    "accounting_basis": "unreconciled-conservative-reservation",
                },
            )
            and source_stage == stage_id
        ):
            return "single-run-budget-recovery"
        if (
            keys == {"error_type", "message"}
            and failure.get("error_type") == "ProjectionNoGo"
            and text(failure.get("message"))
        ):
            return "ancestor-stage-projection-no-go"
        if (
            keys
            == {
                "error_type",
                "cause_type",
                "message",
                "model_id",
                "environment_seed",
                "provider_dispatch_started",
                "stop_origin",
            }
            and failure.get("error_type") == "PreDispatchIntegrityStop"
            and text(failure.get("cause_type"))
            and text(failure.get("message"))
            and failure.get("provider_dispatch_started") is False
            and failure.get("stop_origin") == "experiment-d-pre-provider-revalidation"
            and (failure.get("model_id"), failure.get("environment_seed"))
            in d_identities
            and source_stage == "experiment-d"
            and status == "integrity-stopped"
        ):
            return "experiment-d-pre-dispatch-integrity-stop"
        if (
            keys
            == {
                "error_type",
                "message",
                "model_id",
                "environment_seed",
                "provider_dispatch_started",
                "projection_scope",
                "stop_origin",
            }
            and text(failure.get("error_type"))
            and text(failure.get("message"))
            and failure.get("provider_dispatch_started") is False
            and failure.get("projection_scope")
            == "current-and-remaining-experiment-d-stage"
            and failure.get("stop_origin")
            == "experiment-d-group-pre-dispatch-budget-rejection"
            and (failure.get("model_id"), failure.get("environment_seed"))
            in d_identities
            and source_stage == "experiment-d"
            and status == "budget-stopped"
        ):
            return "experiment-d-pre-dispatch-budget-stop"
        if (
            keys
            == {
                "error_type",
                "cause_type",
                "message",
                "run_id",
                "provider_dispatch_started",
                "stop_origin",
            }
            and failure.get("error_type") == "PreDispatchIntegrityStop"
            and text(failure.get("cause_type"))
            and text(failure.get("message"))
            and failure.get("run_id") in known_run_ids_by_stage[source_stage]
            and failure.get("provider_dispatch_started") is False
            and failure.get("stop_origin") == "actor-pre-provider-revalidation"
            and status == "integrity-stopped"
        ):
            return "actor-pre-dispatch-integrity-stop"
        return None

    def exact_observed_budget_initiator(
        row: Mapping[str, Any],
        itt_row: Mapping[str, Any],
    ) -> bool:
        failure = row.get("failure")
        after_itt = bool(
            row.get("status") == "integrity-stopped"
            and isinstance(failure, Mapping)
            and failure
            == {
                "error_type": "InterruptedReservationAfterITT",
                "message": (
                    "a terminal ITT row retained an unreconciled reservation; "
                    "the conservative reservation was charged before stopping"
                ),
                "accounting_basis": "unreconciled-conservative-reservation",
            }
        )
        replayed_same_terminal = bool(
            row.get("status") in {"budget-stopped", "integrity-stopped"}
            and row.get("status") == itt_row.get("status")
            and isinstance(failure, Mapping)
            and failure == itt_row.get("failure")
            and failure.get("error_type") != "StageStopped"
        )
        return bool(after_itt or replayed_same_terminal)

    def earlier_stage_initiator(run_id: str) -> bool:
        ordered = tuple(spec.run_id for spec in contract.expand(stage=stage_id))
        if run_id not in ordered:
            return False
        for prior_id in ordered[: ordered.index(run_id)]:
            prior = run_rows.get(prior_id)
            if not isinstance(prior, Mapping):
                continue
            prior_disposition = disposition(
                prior_id,
                prior,
                allow_stage_tail=False,
            )
            if prior.get("status") in {
                "budget-stopped",
                "integrity-stopped",
            } and prior_disposition in {
                "per-spec-pre-dispatch-budget-stop",
                "prerequisite-no-go",
                "single-run-budget-recovery",
            }:
                return True
            budget_row = (
                observed_budget_rows.get(prior_id)
                if isinstance(observed_budget_rows, Mapping)
                else None
            )
            if isinstance(budget_row, Mapping) and exact_observed_budget_initiator(
                budget_row,
                prior,
            ):
                return True
        return False

    def disposition(
        run_id: str,
        row: Mapping[str, Any],
        *,
        allow_stage_tail: bool = True,
    ) -> str | None:
        status = row.get("status")
        artifact_kind = row.get("artifact_kind")
        artifact_sha256 = row.get("artifact_sha256")
        failure = row.get("failure")
        if (
            status not in TERMINAL_STATUSES
            or status == "complete"
            or not isinstance(failure, Mapping)
        ):
            return None
        # Catalog no-go happens before a reservation.  Its zero-completion
        # receipt is separately checked by terminal-artifact replay.
        if (
            artifact_kind == "failure-audit-artifact"
            and _is_sha256(artifact_sha256)
            and set(failure)
            == {"error_type", "message", "model_id", "paid_completions"}
            and text(failure.get("error_type"))
            and text(failure.get("message"))
            and failure.get("paid_completions") == 0
            and failure.get("model_id") == model_id
            and status
            == (
                "capability-no-go" if stage_id == "cross-model" else "integrity-stopped"
            )
        ):
            return "provider-catalog-no-go"
        if (
            artifact_kind != "terminal-failure-without-artifact"
            or artifact_sha256 is not None
        ):
            return None
        source_stage = failure.get("source_stage")
        blocked_stage = failure.get("blocked_stage")
        if source_stage is not None or blocked_stage is not None:
            own_prerequisite = bool(
                source_stage == stage_id
                and blocked_stage is None
                and failure.get("error_type") == "PrerequisiteNoGo"
            )
            ancestor = bool(
                isinstance(source_stage, str)
                and source_stage in stage_order
                and blocked_stage == stage_id
                and stage_order[source_stage] < stage_order[stage_id]
            )
            if not (own_prerequisite or ancestor):
                return None
            base = {
                key: value
                for key, value in failure.items()
                if key not in {"source_stage", "blocked_stage"}
            }
            return exact_base(base, source_stage=source_stage, status=status)
        direct = exact_base(failure, source_stage=stage_id, status=status)
        if direct is not None:
            return direct
        if (
            allow_stage_tail
            and earlier_stage_initiator(run_id)
            and failure
            == {
                "error_type": "StageStopped",
                "message": "an earlier budget/integrity failure stopped this stage",
            }
            and status == "budget-stopped"
        ):
            return "own-stage-tail-stop"
        if (
            set(failure) == {"error_type", "message", "projection_scope"}
            and text(failure.get("error_type"))
            and text(failure.get("message"))
            and failure.get("projection_scope") == "current-stage"
            and status in {"budget-stopped", "integrity-stopped"}
        ):
            return "own-stage-projection-stop"
        return None

    classifications = [
        disposition(run_id, row) for run_id, row in zip(linked_ids, linked)
    ]
    if any(value is None for value in classifications):
        raise PilotEvidenceError(
            f"absent budget owner lacks an exact undispatched stop: {budget_id}"
        )
    return {
        "classification": "absent-unreserved-terminal-no-go",
        "stage_id": stage_id,
        "model_id": model_id,
        "linked_itt_count": len(linked_ids),
        "disposition_counts": dict(sorted(Counter(classifications).items())),
    }


def _audit_current_budget(
    contract: PilotContract,
    *,
    raw_root: Path,
    budget: PilotBudgetLedger,
    current_rows: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    run_ledger: Any,
) -> dict[str, Any]:
    snapshot = budget.snapshot()
    rows = _mapping(snapshot.get("runs"), "V2.11.10 budget rows")
    run_snapshot = run_ledger.snapshot()
    run_rows = _mapping(run_snapshot.get("runs"), "V2.11.10 ITT rows")
    current_by_id = {str(row["run_id"]): row for row in current_rows}
    if set(current_by_id) != set(run_rows) or any(
        current_by_id[run_id].get("status") != row.get("status")
        or current_by_id[run_id].get("failure") != row.get("failure")
        for run_id, row in run_rows.items()
    ):
        raise PilotEvidenceError("budget audit received a different ITT denominator")
    owners = _budget_owner_mapping(contract)
    parent_id = tuple(contract.expand(stage="parent-import"))[0].run_id
    if parent_id not in rows or not set(rows).issubset(owners):
        raise PilotEvidenceError("V2.11.10 budget rows are outside the 37-unit universe")
    accepted = _mapping(
        _mapping(
            acceptance.get("budget_projection"), "acceptance budget projection"
        ).get("projection_sha256_by_run_id"),
        "accepted projections",
    )
    science_owner_ids = set(owners) - {parent_id}
    if set(accepted) != science_owner_ids or any(
        not _is_sha256(value) for value in accepted.values()
    ):
        raise PilotEvidenceError("accepted projection universe is not 36 science units")
    try:
        _verify_current_accepted_budget_rows(
            contract,
            acceptance,
            snapshot,
            run_snapshot,
        )
    except Exception as exc:
        raise PilotEvidenceError(f"V2.11.10 budget/ITT replay failed: {exc}") from exc

    try:
        _verified_parent_import_budget_actual(contract, rows[parent_id])
    except Exception as exc:
        raise PilotEvidenceError(f"parent budget replay failed: {exc}") from exc
    reservation_events: dict[str, list[Mapping[str, Any]]] = {}
    finalization_events: dict[str, list[Mapping[str, Any]]] = {}
    events = snapshot.get("events")
    if not isinstance(events, list) or not events:
        raise PilotEvidenceError("V2.11.10 budget events are absent")
    for event in events:
        if not isinstance(event, Mapping):
            raise PilotEvidenceError("V2.11.10 budget event is malformed")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise PilotEvidenceError("V2.11.10 budget event payload is malformed")
        if event.get("event_type") == "run_reserved":
            reservation_events.setdefault(str(payload.get("run_id")), []).append(
                payload
            )
        elif event.get("event_type") == "run_finalized":
            finalization_events.setdefault(str(payload.get("run_id")), []).append(
                payload
            )
    event_type_counts = Counter(str(event.get("event_type")) for event in events)
    expected_event_type_counts = Counter(
        {
            "genesis": 1,
            "parent_debit_imported": 1,
            "acceptance_receipt_bound": 1,
            "run_reserved": len(rows),
            "run_finalized": len(rows),
        }
    )
    if event_type_counts != expected_event_type_counts:
        raise PilotEvidenceError("V2.11.10 budget event type/count inventory drifted")
    if set(reservation_events) != set(rows) or set(finalization_events) != set(rows):
        raise PilotEvidenceError("budget events differ from observed budget units")

    owner_rows: dict[str, Any] = {}
    for budget_id in sorted(rows):
        linked_ids = owners[budget_id]
        row = _mapping(rows[budget_id], f"budget row {budget_id}")
        reservation = _mapping(row.get("reservation"), "budget reservation")
        actual = _mapping(row.get("actual"), "budget actual")
        failure = row.get("failure")
        reserved = reservation_events[budget_id]
        finalized = finalization_events[budget_id]
        if (
            row.get("status")
            not in {"complete", "failed", "budget-stopped", "integrity-stopped"}
            or reservation.get("run_id") != budget_id
            or row.get("stage_bucket") != reservation.get("stage_bucket")
            or len(reserved) != 1
            or reserved[0].get("run_id") != budget_id
            or reserved[0].get("projection_sha256") != canonical_sha256(reservation)
            or len(finalized) != 1
            or finalized[0].get("run_id") != budget_id
            or finalized[0].get("status") != row.get("status")
            or finalized[0].get("actual_sha256") != canonical_sha256(actual)
            or finalized[0].get("failure_sha256")
            != (None if failure is None else canonical_sha256(failure))
        ):
            raise PilotEvidenceError(f"budget owner/status/event drifted: {budget_id}")
        linkage = _observed_owner_linkage(
            contract,
            budget_id=budget_id,
            budget_row=row,
            linked_ids=linked_ids,
            run_rows=run_rows,
        )
        owner_rows[budget_id] = {
            "stage_bucket": row["stage_bucket"],
            "status": row["status"],
            "linked_itt_count": len(linked_ids),
            "linked_itt_run_ids": list(linked_ids),
            "reservation_sha256": canonical_sha256(reservation),
            "actual_sha256": canonical_sha256(actual),
            "failure_sha256": None if failure is None else canonical_sha256(failure),
            "linkage": linkage,
        }
    absent_rows = {
        budget_id: _absent_owner_stop(
            contract,
            budget_id=budget_id,
            linked_ids=owners[budget_id],
            run_rows=current_by_id,
            observed_budget_rows=rows,
        )
        for budget_id in sorted(set(owners) - set(rows))
    }
    committed = _mapping(snapshot.get("committed"), "committed budget")
    caps = _mapping(snapshot.get("caps"), "budget caps")
    if (
        float(committed["cost_usd"]) > float(caps["dispatchable_usd"]) + 1e-12
        or int(committed["completions"]) > int(caps["max_completions"])
        or int(committed["storage_bytes"]) > int(caps["max_storage_bytes"])
    ):
        raise PilotEvidenceError("V2.11.10 committed budget exceeds frozen caps")
    return {
        "schema_version": V21110_BUDGET_AUDIT_SCHEMA_VERSION,
        "path": str(raw_root / "budget_ledger.json"),
        "file_sha256": _sha256_file(raw_root / "budget_ledger.json"),
        "ledger_sha256": snapshot.get("ledger_sha256"),
        "parent_debit": _json_copy(snapshot.get("parent_debit")),
        "committed": _json_copy(committed),
        "caps": _json_copy(caps),
        "event_count": len(events),
        "event_chain_head": events[-1].get("event_sha256"),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "budget_owner_universe_count": len(owners),
        "observed_budget_unit_count": len(rows),
        "absent_unreserved_owner_count": len(absent_rows),
        "linked_itt_row_count": sum(len(value) for value in owners.values()),
        "experiment_d_group_count": sum(
            "--experiment-d--" in run_id for run_id in owners
        ),
        "owner_mapping_sha256": canonical_sha256(
            {key: list(value) for key, value in sorted(owners.items())}
        ),
        "owner_rows": owner_rows,
        "absent_owner_rows": absent_rows,
        "raw_root_storage_bytes": sum(
            path.stat().st_size for path in raw_root.rglob("*") if path.is_file()
        ),
        "pass": True,
    }


def _external_parent_gates(
    contract: PilotContract,
    parent_contract: PilotContract,
    parent_rows: Sequence[Mapping[str, Any]],
    stage_receipts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(stage_receipts.get("receipts"), "V2.11.5 stage receipts")
    a_receipt = _mapping(receipts.get("experiment-a"), "Experiment A receipt")
    c_receipt = _mapping(receipts.get("experiment-c"), "Experiment C receipt")
    expected_receipts = _mapping(
        _mapping(
            contract.to_dict()["v21110_recovery_boundary"],
            "V2.11.10 frozen recovery boundary",
        ).get("parent_stage_receipts"),
        "frozen parent stage receipt bindings",
    )
    for stage_id, observed in (
        ("experiment-a", a_receipt),
        ("experiment-c", c_receipt),
    ):
        expected = _mapping(expected_receipts.get(stage_id), f"frozen {stage_id}")
        observed_path = Path(str(observed.get("path"))).as_posix()
        expected_path = str(expected.get("path"))
        if not observed_path.endswith("/" + expected_path) or any(
            observed.get(key) != value
            for key, value in expected.items()
            if key != "path"
        ):
            raise PilotEvidenceError(
                f"V2.11.5 {stage_id} receipt differs from frozen V2.11.10 binding"
            )
    if (
        a_receipt.get("status") != "complete-with-no-go"
        or a_receipt.get("go") is not False
        or c_receipt.get("status") != "complete-with-no-go"
        or c_receipt.get("go") is not False
    ):
        raise PilotEvidenceError("V2.11.5 A/C no-go receipts drifted")
    a_gate = _experiment_a_gate(parent_contract, parent_rows)
    c_gate = _experiment_c_gate(parent_contract, parent_rows)
    if a_gate.get("support_retrieval_effect") is not False:
        raise PilotEvidenceError("V2.11.5 A outcome no longer matches its no-go")
    a_gate.update(
        {
            "status": "no-go",
            "scientific_evidence_complete": False,
            "support_retrieval_effect": False,
            "authority_release": V2115_CONTRACT_ID,
            "formal_stage_status": a_receipt["status"],
            "formal_stage_go": False,
            "authority_binding": _json_copy(expected_receipts["experiment-a"]),
        }
    )
    c_gate.update(
        {
            "status": "no-go",
            "scientific_evidence_complete": False,
            "support_rule_reliability": False,
            "authority_release": V2115_CONTRACT_ID,
            "formal_stage_status": c_receipt["status"],
            "formal_stage_go": False,
            "authority_binding": _json_copy(expected_receipts["experiment-c"]),
            "claim_action": "withdraw or narrow the rule-reliability claim",
            "reasons": list(c_gate.get("reasons", []))
            + [
                "the immutable parent stage did not seal its preregistered zero-API sensitivity artifact"
            ],
        }
    )
    return {"experiment_a": a_gate, "experiment_c": c_gate}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_package(
    target: Path,
    *,
    aggregate: Mapping[str, Any],
    controls: Mapping[str, Any],
    contract: PilotContract,
    contract_source: Path,
    failed_contract: PilotContract,
    failed_contract_source: Path,
    authority_contract: PilotContract,
    authority_contract_source: Path,
) -> tuple[Path, Path]:
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise PilotEvidenceError("temporary evidence package path is not empty")
    else:
        target.mkdir(parents=True, exist_ok=False)
    rows = aggregate["logical_rows"]
    logical_denominator = aggregate["denominator"]["logical_v2115_matrix"]
    claims = _claims(
        aggregate["claim_gates"],
        denominator=logical_denominator,
    )
    aggregate_payload = {**_json_copy(aggregate), "claims": claims}
    _write_json(target / "aggregate.json", aggregate_payload)
    (target / "aggregate.csv").write_bytes(_aggregate_csv(rows))
    _write_json(target / "release_controls.json", controls)
    _write_json(
        target / "failed_release_lineage.json",
        _mapping(
            aggregate.get("failed_release_lineage"),
            "failed-release lineage audit",
        ),
    )
    contract_dir = target / "contract"
    contract_dir.mkdir()
    contract_copy = contract_dir / contract_source.name
    contract_copy.write_bytes(contract_source.read_bytes())
    if _sha256_file(contract_copy) != _sha256_file(contract_source):
        raise PilotEvidenceError("contract changed while copying into package")
    failed_contract_copy = contract_dir / failed_contract_source.name
    failed_contract_copy.write_bytes(failed_contract_source.read_bytes())
    if _sha256_file(failed_contract_copy) != _sha256_file(failed_contract_source):
        raise PilotEvidenceError(
            "failed-lineage contract changed while copying into package"
        )
    authority_contract_copy = contract_dir / authority_contract_source.name
    authority_contract_copy.write_bytes(authority_contract_source.read_bytes())
    if _sha256_file(authority_contract_copy) != _sha256_file(authority_contract_source):
        raise PilotEvidenceError(
            "authority contract changed while copying into package"
        )
    source_manifests = (
        contract_source.with_name("pilot_v2_11_10_source_manifest.json"),
        failed_contract_source.with_name("pilot_v2_11_9_source_manifest.json"),
        authority_contract_source.with_name("pilot_v2_11_5_source_manifest.json"),
    )
    for source_manifest in source_manifests:
        copied_manifest = contract_dir / source_manifest.name
        copied_manifest.write_bytes(source_manifest.read_bytes())
        if _sha256_file(copied_manifest) != _sha256_file(source_manifest):
            raise PilotEvidenceError("source manifest changed while copying")
    if (
        load_pilot_contract(contract_copy).canonical_hash != contract.canonical_hash
        or load_pilot_contract(failed_contract_copy).canonical_hash
        != failed_contract.canonical_hash
        or load_pilot_contract(authority_contract_copy).canonical_hash
        != authority_contract.canonical_hash
    ):
        raise PilotEvidenceError("copied contract/source-manifest binding drifted")
    failures = [
        {
            "run_id": row["run_id"],
            "source_contract_id": row["contract_id"],
            "stage_id": row["stage_id"],
            "model_id": row["model_id"],
            "arm_id": row["arm_id"],
            "environment_seed": row["environment_seed"],
            "status": row["status"],
            "failure": _json_copy(row.get("failure")),
        }
        for row in rows
        if row["status"] != "complete"
    ]
    _write_json(
        target / "failure_ledger.json",
        {
            "schema_version": PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "denominator": _json_copy(aggregate["denominator"]),
            "rows": failures,
        },
    )
    _write_json(
        target / "claim_metric_artifact.json",
        {
            "schema_version": V21110_EVIDENCE_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "logical_denominator": _json_copy(logical_denominator),
            "claims": claims,
        },
    )
    _write_json(
        target / "method_differences_scaffold.json",
        _method_scaffold(contract_source.name),
    )
    _write_json(
        target / "model_capability_failures.json",
        {
            "schema_version": V21110_EVIDENCE_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "capability": _json_copy(aggregate["inherited_capability"]),
            "cross_model": _json_copy(aggregate["cross_model"]),
            "failure_count": len(failures),
            "failures": failures,
            "claim_boundary": (
                "capability, interface, parsing, provider, and ITT failures are "
                "reported separately; proposer failures are not relabeled as "
                "actor reasoning failures"
            ),
        },
    )
    _write_json(
        target / "narrative_results.json",
        {
            "schema_version": V21110_EVIDENCE_SCHEMA_VERSION,
            "contract_sha256": contract.canonical_hash,
            "gate": _json_copy(aggregate["claim_gates"]["narrative"]),
            "claim_boundary": (
                "controlled semantic response only; no real-news understanding claim"
            ),
        },
    )
    report = (
        "# FinEvo V2.11.10 terminal mechanism pilot\n\n"
        f"- Current release denominator: {CURRENT_LEDGER_DENOMINATOR}\n"
        f"- Cross-release logical denominator: {LOGICAL_REGISTERED_DENOMINATOR}\n"
        f"- Logical scientific denominator: {LOGICAL_SCIENTIFIC_DENOMINATOR}\n"
        f"- Classification: {aggregate['publication_status']['classification']}\n"
        "- Experiment A and C remain external V2.11.5 no-go evidence.\n"
        "- All 87 V2.11.9 rows are immutable failure-lineage audit only; zero are imported as effects.\n"
        "- B, D, narrative, and cross-model summaries use only sealed V2.11.10 continuations.\n"
        "- Capability is inherited dispatch authority, not a fresh V2.11.10 sample.\n"
    )
    (target / "reviewer_report.md").write_text(report, encoding="utf-8")
    published_files = sorted(
        {
            "aggregate.csv",
            "aggregate.json",
            "claim_metric_artifact.json",
            f"contract/{contract_source.name}",
            f"contract/{failed_contract_source.name}",
            f"contract/{authority_contract_source.name}",
            "contract/pilot_v2_11_5_source_manifest.json",
            "contract/pilot_v2_11_9_source_manifest.json",
            "contract/pilot_v2_11_10_source_manifest.json",
            "failed_release_lineage.json",
            "failure_ledger.json",
            "method_differences_scaffold.json",
            "model_capability_failures.json",
            "narrative_results.json",
            "release_controls.json",
            "reviewer_report.md",
        }
    )
    manifest = {
        "schema_version": V21110_EVIDENCE_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "terminal_evidence_complete": aggregate["publication_status"][
            "terminal_evidence_complete"
        ],
        "all_preregistered_claims_supported": aggregate["publication_status"][
            "all_preregistered_claims_supported"
        ],
        "logical_denominator": LOGICAL_REGISTERED_DENOMINATOR,
        "failed_lineage_rows": 87,
        "failed_lineage_rows_imported": 0,
        "raw_provider_calls_during_publication": 0,
        "published_files": published_files,
    }
    _write_json(target / "package_manifest.json", manifest)
    checksum_paths = sorted(path for path in target.rglob("*") if path.is_file())
    checksums = {
        "schema_version": PILOT_CHECKSUM_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "files": [
            {
                "path": path.relative_to(target).as_posix(),
                "sha256": _sha256_file(path),
                "byte_size": path.stat().st_size,
            }
            for path in checksum_paths
        ],
    }
    _write_json(target / "checksums.json", checksums)
    return target / "package_manifest.json", target / "checksums.json"


def _package_target(
    build_root: str | Path,
    *,
    source_roots: Sequence[Path],
) -> Path:
    supplied_build = Path(os.path.abspath(Path(build_root)))
    resolved_build = supplied_build.resolve()
    if supplied_build != resolved_build:
        raise PilotEvidenceError("evidence build root cannot cross a symlink alias")
    target = resolved_build / "current_v2" / "pilot-v2.11.10"
    for source_root in source_roots:
        root = source_root.resolve(strict=True)
        if target == root or target.is_relative_to(root):
            raise PilotEvidenceError(
                "evidence target cannot be inside an immutable source/raw root"
            )
    if target.exists() or target.is_symlink():
        raise PilotEvidenceError(f"refusing to overwrite evidence package: {target}")
    return target


def _verify_package_tree(
    target: Path,
    *,
    contract: PilotContract,
    failed_contract: PilotContract,
    authority_contract: PilotContract,
) -> dict[str, Any]:
    files: dict[str, Path] = {}
    for path in target.rglob("*"):
        relative = path.relative_to(target).as_posix()
        if path.is_symlink():
            raise PilotEvidenceError(f"package crosses a symlink: {relative}")
        if path.is_file():
            files[relative] = path
        elif not path.is_dir():
            raise PilotEvidenceError(f"package contains a special entry: {relative}")
    checksums = _mapping(
        _strict_json_load(target / "checksums.json"), "package checksums"
    )
    checksum_rows = checksums.get("files")
    if not isinstance(checksum_rows, list):
        raise PilotEvidenceError("package checksum rows are malformed")
    by_path = {
        str(row.get("path")): row for row in checksum_rows if isinstance(row, Mapping)
    }
    expected_checksum_paths = set(files) - {"checksums.json"}
    if (
        len(by_path) != len(checksum_rows)
        or set(by_path) != expected_checksum_paths
        or checksums.get("schema_version") != PILOT_CHECKSUM_SCHEMA_VERSION
        or checksums.get("contract_sha256") != contract.canonical_hash
    ):
        raise PilotEvidenceError("package checksum inventory drifted")
    for relative, row in by_path.items():
        path = files[relative]
        if (
            row.get("sha256") != _sha256_file(path)
            or row.get("byte_size") != path.stat().st_size
        ):
            raise PilotEvidenceError(f"package checksum mismatch: {relative}")
    manifest = _mapping(
        _strict_json_load(target / "package_manifest.json"), "package manifest"
    )
    payload_paths = set(files) - {"checksums.json", "package_manifest.json"}
    if (
        manifest.get("schema_version") != V21110_EVIDENCE_SCHEMA_VERSION
        or manifest.get("contract_id") != contract.contract_id
        or manifest.get("contract_sha256") != contract.canonical_hash
        or manifest.get("failed_lineage_rows") != 87
        or manifest.get("failed_lineage_rows_imported") != 0
        or set(manifest.get("published_files", [])) != payload_paths
        or load_pilot_contract(
            target / "contract" / "pilot_v2_11_10.yaml"
        ).canonical_hash
        != contract.canonical_hash
        or load_pilot_contract(
            target / "contract" / "pilot_v2_11_9.yaml"
        ).canonical_hash
        != failed_contract.canonical_hash
        or load_pilot_contract(
            target / "contract" / "pilot_v2_11_5.yaml"
        ).canonical_hash
        != authority_contract.canonical_hash
    ):
        raise PilotEvidenceError("package manifest/contract binding drifted")
    return {
        "file_count": len(files),
        "storage_bytes": sum(path.stat().st_size for path in files.values()),
        "inventory_sha256": canonical_sha256(
            [
                {
                    "path": relative,
                    "sha256": _sha256_file(files[relative]),
                    "byte_size": files[relative].stat().st_size,
                }
                for relative in sorted(files)
            ]
        ),
    }


def _raw_tree_inventory(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PilotEvidenceError(f"raw inventory crosses a symlink: {relative}")
        if path.is_file():
            entries.append(
                {
                    "path": relative,
                    "byte_size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        elif not path.is_dir():
            raise PilotEvidenceError(f"raw inventory has a special entry: {relative}")
    return {
        "file_count": len(entries),
        "storage_bytes": sum(item["byte_size"] for item in entries),
        "inventory_sha256": canonical_sha256(entries),
    }


def _build_pilot_v21110_evidence_package_guarded(
    *,
    contract_path: str | Path,
    run_ledger_path: str | Path,
    raw_root: str | Path,
    build_root: str | Path,
    source_repo_root: str | Path | None = None,
    failed_repo_root: str | Path | None = None,
    authority_repo_root: str | Path | None = None,
) -> PilotEvidencePackage:
    """Build a zero-provider package from three independent release roots."""

    source, current_contract_path, current_raw, current_ledger_path = (
        _resolve_current_paths(
            source_repo_root=source_repo_root,
            contract_path=contract_path,
            raw_root=raw_root,
            run_ledger_path=run_ledger_path,
        )
    )
    current_contract = load_pilot_contract(current_contract_path)
    if authority_repo_root is None:
        raise PilotEvidenceError("V2.11.10 publication requires --authority-repo-root")
    supplied_authority_root = Path(authority_repo_root)
    parent_contract_path = supplied_authority_root / V2115_CONTRACT_RELATIVE
    parent_raw = supplied_authority_root / V2115_RAW_RELATIVE
    parent_ledger_path = parent_raw / "run_ledger.json"
    authority_root, parent_contract_path, parent_raw, parent_ledger_path = (
        _resolve_v2115_source_paths(
            source_repo_root=supplied_authority_root,
            contract_path=parent_contract_path,
            raw_root=parent_raw,
            run_ledger_path=parent_ledger_path,
        )
    )
    parent_contract = load_pilot_contract(parent_contract_path)
    _require_contract_layout(current_contract, parent_contract, require_frozen=True)
    if failed_repo_root is None:
        raise PilotEvidenceError("V2.11.10 publication requires --failed-repo-root")
    try:
        failed_root = _real_root(
            failed_repo_root,
            name="V2.11.9 failed repository",
        )
        _require_distinct_roots(
            current=source,
            failed=failed_root,
            authority=authority_root,
        )
        failed_state = verify_v2119_terminal_no_go(
            failed_repo_root=failed_root,
            authority_repo_root=authority_root,
        )
    except Exception as exc:
        raise PilotEvidenceError(
            f"V2.11.9 immutable terminal no-go validation failed: {exc}"
        ) from exc
    failed_contract = failed_state["failed_contract"]
    failed_contract_path = failed_root.joinpath(*V2119_CONTRACT_PATH.parts)
    failed_raw = failed_root.joinpath(*V2119_FAILED_RAW_ROOT.parts)
    failed_lineage_audit = _expected_failed_lineage_audit(current_contract)

    current_release, current_commit, paid = _validate_release(
        current_contract, raw_root=current_raw
    )
    _assert_v21110_local_release_guard(
        current_contract,
        repo_root=source,
        paid=paid,
    )
    current_source = _validate_current_git(
        source,
        current_contract,
        current_commit,
        release_attestation=paid.release_attestation,
    )
    try:
        current_source_manifest = validate_v21110_source_manifest(
            contract=current_contract,
            repo_root=source,
            failed_repo_root=failed_root,
            authority_repo_root=authority_root,
        )
    except Exception as exc:
        raise PilotEvidenceError(
            f"V2.11.10 three-root source manifest validation failed: {exc}"
        ) from exc
    current_ledger = PilotRunLedger(
        current_ledger_path,
        contract_hash=current_contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    current_budget = PilotBudgetLedger(
        current_raw / "budget_ledger.json",
        contract_hash=current_contract.canonical_hash,
        caps=_budget_caps(current_contract),
        tamper_evident=True,
        parent_debit=parent_budget_debit_for_v21110(current_contract),
    )
    parent_import = verify_v21110_parent_import_receipt(
        current_raw / "parent-import/parent_import_receipt.json",
        contract=current_contract,
        repo_root=source,
        raw_root=current_raw,
        paid=paid,
    )
    authority = verify_v21110_current_authority(
        contract=current_contract,
        repo_root=source,
        raw_root=current_raw,
        paid=paid,
    )
    projections: dict[str, Mapping[str, Any]] = {}
    projection_controls: dict[str, Any] = {}
    for model_id in CAPABILITY_MODELS:
        projection, path = verified_v21110_projection(
            current_contract,
            model_id,
            repo_root=source,
            raw_root=current_raw,
            paid=paid,
        )
        projections[model_id] = projection
        projection_controls[model_id] = {
            "path": str(path),
            "file_sha256": _sha256_file(path),
            "content_sha256": projection["integrity"]["content_sha256"],
            "provider_calls": 0,
            "scientific_evidence": False,
        }
    acceptance = verify_v21110_scientific_dispatch_acceptance(
        current_raw / V21110_ACCEPTANCE_FILENAME,
        contract=current_contract,
        repo_root=source,
        raw_root=current_raw,
        paid=paid,
        run_ledger=current_ledger,
        budget_ledger=current_budget,
    )
    current_rows, run_audit = _normalize_current_ledger(
        current_contract,
        raw_root=current_raw,
        source_repo_root=source,
        ledger=current_ledger,
        paid=paid,
        parent=parent_import,
        authority=authority,
        projections=projections,
    )
    current_stage_receipts = _validate_stage_receipts(
        current_contract,
        raw_root=current_raw,
        ledger=current_ledger,
        paid=paid,
        authority_repo_root=source,
    )
    terminal_namespace = audit_v21110_scientific_stage_namespace(
        current_contract,
        raw_root=current_raw,
        stage_id="cross-model",
        run_ledger=current_ledger,
    )
    current_raw_inventory = _raw_tree_inventory(current_raw)
    failed_raw_inventory = _raw_tree_inventory(failed_raw)
    budget_audit = _audit_current_budget(
        current_contract,
        raw_root=current_raw,
        budget=current_budget,
        current_rows=current_rows,
        acceptance=acceptance,
        run_ledger=current_ledger,
    )

    parent_release, parent_commit, parent_paid = _validate_release(
        parent_contract, raw_root=parent_raw
    )
    parent_source = _validate_v2115_source_git(
        authority_root, parent_contract, expected_commit=parent_commit
    )
    parent_ledger_json = _strict_json_load(parent_ledger_path)
    all_parent_rows, parent_denominator = _normalize_v2115_partial_ledger(
        parent_contract,
        parent_ledger_json,
        raw_root=parent_raw,
        expected_commit=parent_commit,
        source_repo_root=authority_root,
    )
    parent_rows = [
        row for row in all_parent_rows if row["stage_id"] in PARENT_TERMINAL_STAGE_IDS
    ]
    parent_capability = _v2115_capability_by_model(
        all_parent_rows,
        parent_contract,
    )
    parent_ledger = PilotRunLedger(
        parent_ledger_path,
        contract_hash=parent_contract.canonical_hash,
        tamper_evident=True,
    )
    parent_acceptance, parent_budget = _validated_v2115_acceptance_and_budget(
        parent_contract,
        raw_root=parent_raw,
        source_repo_root=authority_root,
        paid=parent_paid,
        run_ledger=parent_ledger,
        rows=all_parent_rows,
    )
    parent_stage_receipts = _normalized_stage_receipts(
        parent_contract,
        raw_root=parent_raw,
        ledger=parent_ledger,
        paid=parent_paid,
        source_repo_root=authority_root,
    )
    parent_post_gate = _validated_post_gate(
        parent_contract,
        source_repo_root=authority_root,
        raw_root=parent_raw,
        commit=parent_commit,
        rows=all_parent_rows,
    )
    parent_raw_inventory = _raw_tree_inventory(parent_raw)
    offline_admission, _ = _validate_offline_candidate_admission(
        parent_contract,
        raw_root=parent_raw,
        rows=all_parent_rows,
        ledger=parent_ledger_json,
    )
    external_gates = _external_parent_gates(
        current_contract,
        parent_contract,
        parent_rows,
        parent_stage_receipts,
    )
    aggregate = assemble_v21110_terminal_evidence(
        contract=current_contract,
        parent_contract=parent_contract,
        parent_terminal_rows=parent_rows,
        current_rows=current_rows,
        parent_import_receipt=parent_import,
        parent_capability_by_model=parent_capability,
        parent_preflight_authority=parent_post_gate,
        current_authority=authority,
        external_parent_gates=external_gates,
        failed_release_audit=failed_lineage_audit,
    )
    controls = {
        "schema_version": V21110_CONTROLS_SCHEMA_VERSION,
        "pass": True,
        "provider_construction": False,
        "provider_calls": 0,
        "current_release": current_release,
        "current_source": current_source,
        "current_source_manifest": {
            "path": V21110_SOURCE_MANIFEST_PATH.as_posix(),
            "file_sha256": _sha256_file(
                source.joinpath(*V21110_SOURCE_MANIFEST_PATH.parts)
            ),
            "content_sha256": current_source_manifest["integrity"][
                "content_sha256"
            ],
            "three_root_replay_pass": True,
            "provider_construction": False,
            "provider_calls": 0,
        },
        "failed_v2119_lineage": failed_lineage_audit,
        "failed_v2119_raw_inventory": failed_raw_inventory,
        "current_parent_import": {
            "content_sha256": parent_import["integrity"]["content_sha256"],
            "scientific_evidence": False,
        },
        "current_authority": {
            "path": str(current_authority_path(current_raw)),
            "file_sha256": _sha256_file(current_authority_path(current_raw)),
            "content_sha256": authority["integrity"]["content_sha256"],
            "provider_calls": 0,
            "scientific_evidence": False,
        },
        "current_projections": projection_controls,
        "scientific_dispatch_acceptance": {
            "path": str(current_raw / V21110_ACCEPTANCE_FILENAME),
            "file_sha256": _sha256_file(current_raw / V21110_ACCEPTANCE_FILENAME),
            "content_sha256": acceptance["integrity"]["content_sha256"],
            "go": acceptance["go"],
            "provider_boundary": _json_copy(acceptance["provider_boundary"]),
        },
        "current_run_ledger": run_audit,
        "current_budget": budget_audit,
        "current_stage_receipts": current_stage_receipts,
        "terminal_namespace": terminal_namespace,
        "current_raw_inventory": current_raw_inventory,
        "parent_release": parent_release,
        "parent_source": parent_source,
        "parent_partial_denominator": parent_denominator,
        "parent_acceptance": parent_acceptance,
        "parent_budget": parent_budget,
        "parent_stage_receipts": parent_stage_receipts,
        "parent_post_gate": parent_post_gate,
        "parent_capability": parent_capability,
        "parent_raw_inventory": parent_raw_inventory,
        "parent_offline_candidate_admission": offline_admission,
    }

    target = _package_target(
        build_root,
        source_roots=(
            current_raw,
            failed_root,
            failed_raw,
            authority_root,
            parent_raw,
        ),
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=".pilot-v2.11.10-build-", dir=source.parent)
    )
    try:
        manifest, checksums = _write_package(
            temporary,
            aggregate=aggregate,
            controls=controls,
            contract=current_contract,
            contract_source=current_contract_path,
            failed_contract=failed_contract,
            failed_contract_source=failed_contract_path,
            authority_contract=parent_contract,
            authority_contract_source=parent_contract_path,
        )
        package_inventory = _verify_package_tree(
            temporary,
            contract=current_contract,
            failed_contract=failed_contract,
            authority_contract=parent_contract,
        )
        total_bytes = sum(
            path.stat().st_size for path in temporary.rglob("*") if path.is_file()
        )
        if total_bytes + budget_audit["raw_root_storage_bytes"] > int(
            current_contract.budgets["max_storage_bytes"]
        ):
            raise PilotEvidenceError("raw plus V2.11.10 package exceeds storage cap")
        _assert_v21110_local_release_guard(
            current_contract,
            repo_root=source,
            paid=paid,
        )
        _validate_current_git(
            source,
            current_contract,
            current_commit,
            release_attestation=paid.release_attestation,
        )
        _validate_v2115_source_git(
            authority_root,
            parent_contract,
            expected_commit=parent_commit,
        )
        try:
            final_failed_state = verify_v2119_terminal_no_go(
                failed_repo_root=failed_root,
                authority_repo_root=authority_root,
            )
            final_source_manifest = validate_v21110_source_manifest(
                contract=current_contract,
                repo_root=source,
                failed_repo_root=failed_root,
                authority_repo_root=authority_root,
            )
        except Exception as exc:
            raise PilotEvidenceError(
                f"three-root lineage drifted before installation: {exc}"
            ) from exc
        final_namespace = audit_v21110_scientific_stage_namespace(
            current_contract,
            raw_root=current_raw,
            stage_id="cross-model",
            run_ledger=current_ledger,
        )
        if (
            final_namespace != terminal_namespace
            or _raw_tree_inventory(current_raw) != current_raw_inventory
            or _raw_tree_inventory(failed_raw) != failed_raw_inventory
            or _raw_tree_inventory(parent_raw) != parent_raw_inventory
            or final_failed_state["failed_contract"].canonical_hash
            != failed_contract.canonical_hash
            or final_source_manifest != current_source_manifest
            or _sha256_file(current_contract_path)
            != _sha256_file(temporary / "contract" / current_contract_path.name)
            or _sha256_file(parent_contract_path)
            != _sha256_file(temporary / "contract" / parent_contract_path.name)
            or _sha256_file(current_ledger_path) != run_audit["file_sha256"]
            or _sha256_file(current_raw / "budget_ledger.json")
            != budget_audit["file_sha256"]
        ):
            raise PilotEvidenceError(
                "source provenance or raw inventory drifted before installation"
            )
        if (
            _verify_package_tree(
                temporary,
                contract=current_contract,
                failed_contract=failed_contract,
                authority_contract=parent_contract,
            )
            != package_inventory
        ):
            raise PilotEvidenceError("package inventory drifted before installation")
        target.parent.mkdir(parents=True, exist_ok=True)
        _install_package_no_overwrite(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PilotEvidencePackage(
        package_dir=target,
        manifest_path=target / manifest.name,
        checksums_path=target / checksums.name,
        contract_hash=current_contract.canonical_hash,
        scientific_complete=bool(
            aggregate["publication_status"]["all_preregistered_claims_supported"]
        ),
        claim_gates=_json_copy(aggregate["claim_gates"]),
    )


def build_pilot_v21110_evidence_package(
    *,
    contract_path: str | Path,
    run_ledger_path: str | Path,
    raw_root: str | Path,
    build_root: str | Path,
    source_repo_root: str | Path | None = None,
    failed_repo_root: str | Path | None = None,
    authority_repo_root: str | Path | None = None,
) -> PilotEvidencePackage:
    """Build evidence under an explicit zero-credential/provider sentinel."""

    require_v21110_provider_keys_absent()
    with v2117._acceptance_provider_sentinels():
        return _build_pilot_v21110_evidence_package_guarded(
            contract_path=contract_path,
            run_ledger_path=run_ledger_path,
            raw_root=raw_root,
            build_root=build_root,
            source_repo_root=source_repo_root,
            failed_repo_root=failed_repo_root,
            authority_repo_root=authority_repo_root,
        )


__all__ = [
    "CURRENT_LEDGER_DENOMINATOR",
    "LOGICAL_REGISTERED_DENOMINATOR",
    "LOGICAL_SCIENTIFIC_DENOMINATOR",
    "V21110_EVIDENCE_SCHEMA_VERSION",
    "V21110_FAILED_LINEAGE_AUDIT_SCHEMA_VERSION",
    "assemble_v21110_terminal_evidence",
    "build_pilot_v21110_evidence_package",
    "inherited_capability_by_model",
]
