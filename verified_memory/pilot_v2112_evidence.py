"""Fail-closed evidence publication for the FinEvo V2.11.2 pilot.

V2.11.2 is not a lane-compatible continuation of V2.4--V2.10.2.  It has a
fresh 136-cell hosted denominator, imports only calibration and capability
wrappers from V2.11.1, and obtains a new long-context preflight/P95 authority.
This adapter therefore keeps the inherited prerequisite audit separate from
fresh preflight and scientific A--D evidence.

The publisher is provider-free.  It validates existing ledgers, release and
stage receipts, artifacts, the post-gate authority, and budget accounting; it
never repairs or dispatches a cell.  A draft contract or a nonterminal ITT
denominator is rejected before a package is written.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .pilot_budget import PilotBudgetLedger
from .pilot_contract import PilotContract, canonical_sha256, load_pilot_contract
from .pilot_evidence import (
    CURRENT_SCIENTIFIC_SCOPE,
    HISTORICAL_SCOPE,
    PILOT_CHECKSUM_SCHEMA_VERSION,
    PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
    PILOT_RUN_LEDGER_SCHEMA_VERSION_V2,
    PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION,
    PilotEvidenceError,
    PilotEvidencePackage,
    TERMINAL_STATUSES,
    V211_NON_SCIENTIFIC_STAGES,
    V211_SCIENTIFIC_STAGES,
    _aggregate_csv,
    _atomic_bytes,
    _capability_by_model,
    _claims,
    _cross_model_summary,
    _evidence_namespace,
    _experiment_a_gate,
    _experiment_b_summary,
    _experiment_c_gate,
    _experiment_d_gate,
    _json_copy,
    _load_completed_artifact,
    _method_scaffold,
    _narrative_gate,
    _pretty_bytes,
    _resolve_artifact,
    _sha256_file,
    _strict_json_load,
    _validate_v2_release_attestation,
    _validated_experiment_c_sensitivity,
)
from .pilot_orchestrator import (
    GitProvenance,
    PilotOrchestrationError,
    PilotRunLedger,
    _budget_caps,
    _parent_budget_debit,
    _verify_v2_stage_receipt,
)
from .pilot_v2112_gate import verify_v2112_gate_receipt


V2112_CONTRACT_ID = "finevo-pilot-v2.11.2"
V2112_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.11.2-evidence-package-v1"
V2112_STAGE_RECEIPTS_SCHEMA_VERSION = "finevo-pilot-v2.11.2-stage-receipts-audit-v1"
V2112_RUN_LEDGER_RECEIPT_SCHEMA_VERSION = "finevo-pilot-v2.11.2-run-ledger-audit-v1"
V2112_BUDGET_RECEIPT_SCHEMA_VERSION = "finevo-pilot-v2.11.2-budget-audit-v1"

_EXPECTED_STAGE_IDS = (
    "parent-import",
    "capability-gate",
    "long-context-preflight",
    "experiment-c",
    "experiment-a",
    "experiment-d",
    "experiment-b",
    "cross-model",
)
_EXPECTED_STAGE_COUNTS = {
    "parent-import": 1,
    "capability-gate": 2,
    "long-context-preflight": 2,
    "experiment-c": 25,
    "experiment-a": 20,
    "experiment-d": 55,
    "experiment-b": 25,
    "cross-model": 6,
}
_NONTERMINAL_STATUSES = frozenset({"scheduled", "running", "reserved"})


def _frozen_contract(contract: PilotContract) -> None:
    if contract.contract_id != V2112_CONTRACT_ID:
        raise PilotEvidenceError(
            "V2.11.2 evidence adapter received a different contract"
        )
    if contract.status != "frozen":
        raise PilotEvidenceError(
            "V2.11.2 publish-evidence requires the frozen contract; draft "
            "contracts are execution and publication no-go"
        )
    if tuple(contract.stage_ids) != _EXPECTED_STAGE_IDS:
        raise PilotEvidenceError("V2.11.2 evidence stage order drifted")
    counts = {
        stage_id: len(contract.expand(stage=stage_id))
        for stage_id in contract.stage_ids
    }
    if counts != _EXPECTED_STAGE_COUNTS or sum(counts.values()) != 136:
        raise PilotEvidenceError("V2.11.2 evidence denominator is not 136 cells")
    if contract.v2112_forward_boundary is None:
        raise PilotEvidenceError("V2.11.2 forward boundary is absent")
    if contract.v2112_recovery_amendment is None:
        raise PilotEvidenceError("V2.11.2 lifecycle recovery amendment is absent")


def _terminal_summary_header(
    contract: PilotContract,
    spec: Mapping[str, Any],
    path: Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    value = _strict_json_load(path)
    unsigned = _json_copy(value)
    integrity = unsigned.pop("integrity", None)
    if (
        value.get("schema_version") != PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION
        or not isinstance(integrity, Mapping)
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or integrity.get("content_sha256") != canonical_sha256(unsigned)
    ):
        raise PilotEvidenceError(f"terminal summary integrity drifted: {path}")
    if (
        value.get("contract_id") != contract.contract_id
        or value.get("contract_sha256") != contract.canonical_hash
        or value.get("run_spec") != dict(spec)
    ):
        raise PilotEvidenceError(f"terminal summary contract/spec drifted: {path}")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PilotEvidenceError(f"terminal summary provenance is absent: {path}")
    expected_binding = contract.validate_provenance(
        expected_commit,
        str(contract.implementation["required_git_tag"]),
    )
    observed_binding = {
        key: provenance.get(key)
        for key in (
            "git_tag",
            "resolved_git_commit",
            "commit_resolution",
            "p0_base_commit",
            "contract_id",
            "contract_sha256",
        )
    }
    if (
        observed_binding != expected_binding
        or provenance.get("tag_object_type") != "tag"
        or provenance.get("worktree_clean") is not True
    ):
        raise PilotEvidenceError(f"terminal summary release binding drifted: {path}")
    stage_id = str(spec["stage_id"])
    expected_scope = {
        "parent-import": "preregistered_parent_authority_import",
        "capability-gate": "preregistered_task_capability_gate",
        "long-context-preflight": "preregistered_task_capability_gate",
    }[stage_id]
    if (
        value.get("diagnostic_only") is not False
        or value.get("scientific_evidence") is not False
        or value.get("evidence_scope") != expected_scope
    ):
        raise PilotEvidenceError(
            f"{stage_id} terminal summary falsely claims scientific evidence"
        )
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise PilotEvidenceError(f"terminal summary payload is malformed: {path}")
    return value


def _normalize_v2112_ledger(
    contract: PilotContract,
    ledger: Mapping[str, Any],
    *,
    raw_root: Path,
    expected_commit: str,
    source_repo_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a terminal 136-cell ITT ledger and every completed artifact."""

    if ledger.get("schema_version") != PILOT_RUN_LEDGER_SCHEMA_VERSION_V2:
        raise PilotEvidenceError("V2.11.2 run ledger schema drifted")
    if ledger.get("contract_hash") != contract.canonical_hash:
        raise PilotEvidenceError("V2.11.2 run ledger contract hash drifted")
    # The constructor replays the event chain and checks the self-hash.  It is
    # read-only because the file already exists.
    PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    observed = ledger.get("runs")
    if not isinstance(observed, Mapping):
        raise PilotEvidenceError("V2.11.2 run ledger rows are malformed")
    expected_specs = {spec.run_id: spec.to_dict() for spec in contract.expand()}
    if set(observed) != set(expected_specs):
        raise PilotEvidenceError(
            "V2.11.2 publication requires all 136 registered ledger rows"
        )

    rows: list[dict[str, Any]] = []
    for run_id, spec in expected_specs.items():
        source = observed[run_id]
        if not isinstance(source, Mapping) or source.get("spec") != spec:
            raise PilotEvidenceError(f"ledger spec drifted for {run_id}")
        status = source.get("status")
        if status in _NONTERMINAL_STATUSES or status not in TERMINAL_STATUSES:
            raise PilotEvidenceError(
                "V2.11.2 publication requires an all-terminal ITT denominator; "
                f"{run_id} is {status!r}"
            )
        row: dict[str, Any] = {
            **spec,
            "status": status,
            "failure": _json_copy(source.get("failure")),
            "artifact_kind": None,
            "artifact_sha256": None,
            "scientific_eligible": False,
            "metrics": {},
            "gate_evidence": {},
            "capability": {},
            "narrative": {},
        }
        artifact = source.get("artifact")
        if status == "complete":
            if spec["stage_id"] in V211_SCIENTIFIC_STAGES:
                evidence = _load_completed_artifact(
                    contract,
                    spec,
                    raw_root=raw_root,
                    artifact=artifact,
                    source_repo_root=source_repo_root,
                )
                if evidence.get("scientific_eligible") is not True:
                    raise PilotEvidenceError(
                        f"completed scientific cell is ineligible: {run_id}"
                    )
            else:
                path = _resolve_artifact(raw_root, artifact)
                terminal = _terminal_summary_header(
                    contract,
                    spec,
                    path,
                    expected_commit=expected_commit,
                )
                payload = terminal["payload"]
                evidence = {
                    "artifact_kind": "terminal-summary",
                    "artifact_sha256": _sha256_file(path),
                    "scientific_eligible": False,
                    "metrics": _json_copy(payload.get("metrics", {})),
                    "gate_evidence": _json_copy(payload.get("gate_evidence", {})),
                    "capability": _json_copy(payload.get("capability", {})),
                    "narrative": _json_copy(payload.get("narrative", {})),
                }
            row.update(evidence)
        elif status == "capability-no-go" and artifact is not None:
            path = _resolve_artifact(raw_root, artifact)
            terminal = _terminal_summary_header(
                contract,
                spec,
                path,
                expected_commit=expected_commit,
            )
            payload = terminal["payload"]
            row.update(
                {
                    "artifact_kind": "terminal-summary",
                    "artifact_sha256": _sha256_file(path),
                    "capability": _json_copy(payload.get("capability", {})),
                }
            )
        elif artifact is not None:
            # Failure artifacts remain audit-only.  Bind their bytes, but do
            # not expose metrics or reinterpret them as effect evidence.
            path = _resolve_artifact(raw_root, artifact)
            row["artifact_kind"] = "failure-audit-artifact"
            row["artifact_sha256"] = _sha256_file(path)
        rows.append(row)

    counts = Counter(str(row["status"]) for row in rows)
    stage_counts = {
        stage_id: dict(
            sorted(
                Counter(
                    str(row["status"]) for row in rows if row["stage_id"] == stage_id
                ).items()
            )
        )
        for stage_id in contract.stage_ids
    }
    denominator = {
        "expected_count": 136,
        "observed_ledger_count": len(observed),
        "all_rows_present": True,
        "all_rows_terminal": True,
        "status_counts": dict(sorted(counts.items())),
        "stage_status_counts": stage_counts,
        "all_completed_artifacts_validated": all(
            row["artifact_kind"] is not None
            for row in rows
            if row["status"] == "complete"
        ),
        "itt_failures_retained": sum(
            count for status, count in counts.items() if status != "complete"
        ),
    }
    denominator["pass"] = bool(denominator["all_completed_artifacts_validated"])
    return rows, denominator


def _paid_provenance(
    contract: PilotContract,
    release: Mapping[str, Any],
    *,
    commit: str,
) -> GitProvenance:
    return GitProvenance(
        git_tag=str(contract.implementation["required_git_tag"]),
        head_commit=commit,
        tag_commit=commit,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding=contract.validate_provenance(
            commit,
            str(contract.implementation["required_git_tag"]),
        ),
        release_attestation=_json_copy(release),
    )


def _validate_release(
    contract: PilotContract,
    *,
    raw_root: Path,
) -> tuple[dict[str, Any], str, GitProvenance]:
    path = raw_root / "release_attestation.json"
    release = _strict_json_load(path)
    commit = release.get("head_commit")
    if not isinstance(commit, str):
        raise PilotEvidenceError("V2.11.2 release attestation lacks HEAD commit")
    try:
        checks = _validate_v2_release_attestation(
            contract,
            release,
            common_commit=commit,
        )
    except (PilotEvidenceError, TypeError, ValueError, KeyError) as exc:
        raise PilotEvidenceError(
            f"V2.11.2 release attestation failed replay: {exc}"
        ) from exc
    if not checks or not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise PilotEvidenceError(f"V2.11.2 release attestation is no-go: {failed}")
    return (
        {
            "pass": True,
            "path": str(path),
            "file_sha256": _sha256_file(path),
            "attestation_sha256": release.get("attestation_sha256"),
            "checks": checks,
        },
        commit,
        _paid_provenance(contract, release, commit=commit),
    )


def _validate_stage_receipts(
    contract: PilotContract,
    *,
    raw_root: Path,
    ledger: PilotRunLedger,
    paid: GitProvenance,
) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    ledger_rows = ledger.snapshot()["runs"]
    for stage_id in contract.stage_ids:
        path = raw_root / stage_id / "stage_receipt.json"
        specs = contract.expand(stage=stage_id)
        stage_rows = [ledger_rows[spec.run_id] for spec in specs]
        if not path.exists():
            residual_files = (
                [item for item in (raw_root / stage_id).rglob("*") if item.is_file()]
                if (raw_root / stage_id).exists()
                else []
            )
            safe_ancestor_stop = bool(
                stage_rows
                and all(
                    row.get("status") in TERMINAL_STATUSES
                    and row.get("status") != "complete"
                    and row.get("artifact") is None
                    and isinstance(row.get("failure"), Mapping)
                    and isinstance(row["failure"].get("source_stage"), str)
                    and row["failure"].get("blocked_stage") == stage_id
                    for row in stage_rows
                )
                and not residual_files
            )
            if not safe_ancestor_stop:
                raise PilotEvidenceError(
                    f"V2.11.2 {stage_id} terminal rows lack their stage receipt"
                )
            receipts[stage_id] = {
                "path": None,
                "file_sha256": None,
                "content_sha256": None,
                "status": "ancestor-no-go-propagated",
                "go": False,
                "execution_progression_go": False,
                "go_models": [],
                "status_counts": dict(
                    sorted(Counter(str(row["status"]) for row in stage_rows).items())
                ),
                "registered_run_count": len(specs),
                "complete_cell_count": 0,
                "receipt_absence_validated_from_itt_ancestor_stop": True,
            }
            continue
        try:
            value = _verify_v2_stage_receipt(
                contract,
                stage_id,
                _strict_json_load(path),
                raw_root=raw_root,
                ledger=ledger,
                paid=paid,
            )
        except (PilotOrchestrationError, PilotEvidenceError) as exc:
            raise PilotEvidenceError(
                f"V2.11.2 {stage_id} receipt failed replay: {exc}"
            ) from exc
        if value.get("terminal") is not True:
            raise PilotEvidenceError(f"V2.11.2 {stage_id} receipt is nonterminal")
        receipts[stage_id] = {
            "path": str(path),
            "file_sha256": _sha256_file(path),
            "content_sha256": value["integrity"]["content_sha256"],
            "status": value.get("status"),
            "go": value.get("go"),
            "execution_progression_go": value.get("execution_progression_go"),
            "go_models": _json_copy(value.get("go_models", [])),
            "status_counts": _json_copy(value.get("status_counts", {})),
            "registered_run_count": value.get("registered_run_count"),
            "complete_cell_count": value.get("complete_cell_count"),
        }
    return {
        "schema_version": V2112_STAGE_RECEIPTS_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "all_terminal": True,
        "receipts": receipts,
    }


def _validate_post_gate(
    contract: PilotContract,
    *,
    raw_root: Path,
    commit: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = raw_root / "long-context-preflight" / "post_gate_authority.json"
    complete_science = any(
        row["stage_id"] in V211_SCIENTIFIC_STAGES and row["status"] == "complete"
        for row in rows
    )
    if not path.exists():
        if complete_science:
            raise PilotEvidenceError(
                "scientific cells exist without a V2.11.2 post-gate authority"
            )
        return {
            "available": False,
            "go": False,
            "reason": "fresh preflight was terminal no-go before authority sealing",
        }
    try:
        receipt = verify_v2112_gate_receipt(
            _strict_json_load(path),
            expected_contract_sha256=contract.canonical_hash,
            expected_git_commit=commit,
        )
    except Exception as exc:
        raise PilotEvidenceError(
            f"V2.11.2 post-gate authority failed replay: {exc}"
        ) from exc
    if complete_science and receipt.get("go") is not True:
        raise PilotEvidenceError(
            "scientific cells exist behind a V2.11.2 global preflight no-go"
        )
    return {
        "available": True,
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "content_sha256": receipt["receipt_sha256"],
        "go": receipt["go"],
        "eligible_model_ids": _json_copy(receipt["denominator"]["eligible_model_ids"]),
        "model_decisions": _json_copy(receipt["model_decisions"]),
        "evidence_actuals": _json_copy(receipt["evidence_actuals"]),
        "projection": _json_copy(receipt["projection"]),
        "provider_calls_during_authority": 0,
        "scientific_evidence": False,
    }


def _allowed_budget_ids(contract: PilotContract) -> set[str]:
    allowed = {
        spec.run_id
        for spec in contract.expand()
        if spec.execution_mode != "checkpoint_continuation"
    }
    allowed.update(
        f"{contract.contract_id}--experiment-d--gpt52_main--"
        f"checkpoint-group--s{seed}"
        for seed in contract.seeds["sets"]["main"]
    )
    return allowed


def _validate_budget(
    contract: PilotContract,
    *,
    raw_root: Path,
    repo_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = raw_root / "budget_ledger.json"
    try:
        parent_debit = _parent_budget_debit(contract, repo_root=repo_root)
        ledger = PilotBudgetLedger(
            path,
            contract_hash=contract.canonical_hash,
            caps=_budget_caps(contract),
            tamper_evident=True,
            parent_debit=parent_debit,
        )
        snapshot = ledger.snapshot()
    except Exception as exc:
        raise PilotEvidenceError(f"V2.11.2 budget ledger failed replay: {exc}") from exc
    budget_rows = snapshot["runs"]
    if not set(budget_rows).issubset(_allowed_budget_ids(contract)):
        raise PilotEvidenceError("V2.11.2 budget ledger has foreign run IDs")
    reserve_events: dict[str, int] = {}
    finalize_events: dict[str, int] = {}
    for event in snapshot["events"]:
        event_type = event.get("event_type")
        if event_type in {"genesis", "parent_debit_imported"}:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise PilotEvidenceError("V2.11.2 budget event payload is malformed")
        run_id = payload.get("run_id")
        row = budget_rows.get(run_id)
        if not isinstance(run_id, str) or not isinstance(row, Mapping):
            raise PilotEvidenceError("V2.11.2 budget event references a foreign run")
        if event_type == "run_reserved":
            if run_id in reserve_events or payload.get(
                "projection_sha256"
            ) != canonical_sha256(row.get("reservation")):
                raise PilotEvidenceError(
                    f"V2.11.2 budget reservation event drifted: {run_id}"
                )
            reserve_events[run_id] = int(event["event_index"])
        elif event_type == "run_finalized":
            failure = row.get("failure")
            if (
                run_id in finalize_events
                or payload.get("status") != row.get("status")
                or payload.get("actual_sha256") != canonical_sha256(row.get("actual"))
                or payload.get("failure_sha256")
                != (None if failure is None else canonical_sha256(failure))
            ):
                raise PilotEvidenceError(
                    f"V2.11.2 budget finalization event drifted: {run_id}"
                )
            finalize_events[run_id] = int(event["event_index"])
        else:
            raise PilotEvidenceError(
                f"V2.11.2 budget event type is unsupported: {event_type!r}"
            )
    if (
        set(reserve_events) != set(budget_rows)
        or set(finalize_events) != set(budget_rows)
        or any(
            reserve_events[run_id] >= finalize_events[run_id] for run_id in budget_rows
        )
    ):
        raise PilotEvidenceError(
            "V2.11.2 budget rows lack one ordered reserve/finalize event pair"
        )
    for run_id, row in budget_rows.items():
        reservation = row.get("reservation")
        actual = row.get("actual")
        status = row.get("status")
        if (
            not isinstance(reservation, Mapping)
            or not isinstance(actual, Mapping)
            or reservation.get("run_id") != run_id
            or reservation.get("stage_bucket") != row.get("stage_bucket")
            or status
            not in {"complete", "failed", "budget-stopped", "integrity-stopped"}
        ):
            raise PilotEvidenceError(
                f"V2.11.2 budget row is malformed or nonterminal: {run_id}"
            )
        for field in ("cost_usd", "completions", "storage_bytes"):
            observed = actual.get(field)
            reserved = reservation.get(field)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or float(observed) < 0
                or isinstance(reserved, bool)
                or not isinstance(reserved, (int, float))
                or float(reserved) < 0
            ):
                raise PilotEvidenceError(
                    f"V2.11.2 budget row has invalid {field}: {run_id}"
                )
            if float(observed) > float(reserved) + 1e-12 and (
                status != "integrity-stopped"
                or not isinstance(row.get("failure"), Mapping)
            ):
                raise PilotEvidenceError(
                    f"V2.11.2 budget actual exceeds reservation: {run_id}"
                )
    nonfinal = sorted(
        run_id
        for run_id, row in budget_rows.items()
        if row.get("actual") is None or row.get("status") == "reserved"
    )
    if nonfinal:
        raise PilotEvidenceError(
            f"V2.11.2 budget ledger has unfinalized reservations: {nonfinal[:3]}"
        )
    required_ids: set[str] = set()
    for row in rows:
        if row.get("artifact_kind") is None:
            continue
        if row["execution_mode"] == "checkpoint_continuation":
            required_ids.add(
                f"{contract.contract_id}--experiment-d--gpt52_main--"
                f"checkpoint-group--s{row['environment_seed']}"
            )
        else:
            required_ids.add(str(row["run_id"]))
    if not required_ids.issubset(set(budget_rows)):
        missing = sorted(required_ids - set(budget_rows))
        raise PilotEvidenceError(
            f"artifact-backed dispatch lacks budget rows: {missing[:3]}"
        )
    totals = snapshot["committed"]
    caps = snapshot["caps"]
    if (
        float(totals["cost_usd"]) > float(caps["dispatchable_usd"]) + 1e-12
        or int(totals["completions"]) > int(caps["max_completions"])
        or int(totals["storage_bytes"]) > int(caps["max_storage_bytes"])
        or any(
            float(cost) > float(caps["stage_usd_caps"][stage_id]) + 1e-12
            for stage_id, cost in totals["stage_cost_usd"].items()
        )
    ):
        raise PilotEvidenceError("V2.11.2 committed budget exceeds frozen caps")
    raw_storage = sum(
        path.stat().st_size for path in raw_root.rglob("*") if path.is_file()
    )
    if raw_storage > int(caps["max_storage_bytes"]):
        raise PilotEvidenceError("V2.11.2 raw tree exceeds the frozen storage cap")
    return {
        "schema_version": V2112_BUDGET_RECEIPT_SCHEMA_VERSION,
        "pass": True,
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "ledger_sha256": snapshot["ledger_sha256"],
        "event_chain_head": snapshot["event_chain_head"],
        "parent_debit": _json_copy(snapshot["parent_debit"]),
        "committed": _json_copy(totals),
        "caps": _json_copy(caps),
        "raw_root_storage_bytes": raw_storage,
        "finalized_budget_unit_count": len(budget_rows),
    }


def _run_ledger_receipt(
    contract: PilotContract,
    ledger: Mapping[str, Any],
    denominator: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    events = ledger["events"]
    return {
        "schema_version": V2112_RUN_LEDGER_RECEIPT_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "ledger_sha256": ledger["ledger_sha256"],
        "event_count": len(events),
        "event_chain_head": events[-1]["event_sha256"],
        "denominator": _json_copy(denominator),
    }


def _report(
    contract: PilotContract,
    *,
    denominator: Mapping[str, Any],
    gates: Mapping[str, Any],
    capability: Mapping[str, Any],
    cross_model: Mapping[str, Any],
    release_controls: Mapping[str, Any],
) -> str:
    lines = [
        "# FinEvo V2.11.2 preregistered mechanism micro-pilot",
        "",
        f"- Contract: `{contract.contract_id}` / `{contract.canonical_hash}`",
        "- Scale: 4 agents x 12 months; not the 10x24x5 confirmatory pilot or 100x240 run.",
        f"- ITT denominator: 136/136 terminal; statuses `{json.dumps(denominator['status_counts'], sort_keys=True)}`.",
        "- V2.11.1 failed-preflight journals are budget/failure audit only and are not scientific evidence.",
        "",
        "## Claim decisions",
        "",
    ]
    for name in ("experiment_a", "experiment_c", "experiment_d", "narrative"):
        gate = gates[name]
        boundary = gate.get("claim_action") or gate.get("claim_boundary")
        lines.append(f"- `{name}`: `{gate['status']}` — {boundary}")
    lines.extend(
        [
            "",
            "Experiment B is reported descriptively; no arm is selected by wealth alone.",
            "",
            "## Capability, fresh preflight, and cross-model boundary",
            "",
            f"- Capability/preflight: `{json.dumps(capability, sort_keys=True)}`",
            f"- Cross-model: `{json.dumps(cross_model, sort_keys=True)}`",
            "- No result supports backbone-independent wording.",
            "",
            "## Provenance and controls",
            "",
            f"- Release: `{json.dumps(release_controls['release'], sort_keys=True)}`",
            f"- Fresh post-gate authority: `{json.dumps(release_controls['post_gate'], sort_keys=True)}`",
            f"- Budget: `{json.dumps(release_controls['budget'], sort_keys=True)}`",
            "",
            "All failures remain in `failure_ledger.json`; raw prompts and provider outputs are not copied.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_package(
    target: Path,
    *,
    contract_path: Path,
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    denominator: Mapping[str, Any],
    gates: Mapping[str, Any],
    capability: Mapping[str, Any],
    cross_model: Mapping[str, Any],
    release_controls: Mapping[str, Any],
    experiment_b: Mapping[str, Any],
    rule_sensitivity: Mapping[str, Any] | None,
    run_ledger: Mapping[str, Any],
    budget_ledger: Mapping[str, Any],
) -> tuple[Path, Path, bool]:
    target.mkdir(parents=True, exist_ok=True)
    contract_target = target / "contract" / contract_path.name
    contract_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract_path, contract_target)
    if load_pilot_contract(contract_target).canonical_hash != contract.canonical_hash:
        raise PilotEvidenceError("copied V2.11.2 contract failed revalidation")

    sanitized_rows = [
        {
            key: _json_copy(row[key])
            for key in (
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
        }
        for row in rows
    ]
    claims = _claims(gates, denominator=denominator)
    aggregate = {
        "schema_version": V2112_EVIDENCE_SCHEMA_VERSION,
        "evidence_namespace": _evidence_namespace(contract),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "resolved_git_commit": release_controls["resolved_git_commit"],
        "denominator": _json_copy(denominator),
        "claim_gates": _json_copy(gates),
        "claims": claims,
        "experiment_b": _json_copy(experiment_b),
        "model_capability": _json_copy(capability),
        "cross_model": _json_copy(cross_model),
        "release_controls": _json_copy(release_controls),
        "experiment_c_rule_sensitivity": _json_copy(rule_sensitivity),
        "rows": sanitized_rows,
    }
    _atomic_bytes(target / "aggregate.json", _pretty_bytes(aggregate))
    _atomic_bytes(target / "aggregate.csv", _aggregate_csv(sanitized_rows))
    failures = [
        {
            "run_id": row["run_id"],
            "stage_id": row["stage_id"],
            "model_id": row["model_id"],
            "arm_id": row["arm_id"],
            "narrative_id": row["narrative_id"],
            "environment_seed": row["environment_seed"],
            "status": row["status"],
            "failure": _json_copy(row["failure"]),
        }
        for row in rows
        if row["status"] != "complete"
    ]
    _atomic_bytes(
        target / "failure_ledger.json",
        _pretty_bytes(
            {
                "schema_version": PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
                "contract_sha256": contract.canonical_hash,
                "denominator": _json_copy(denominator),
                "rows": failures,
            }
        ),
    )
    _atomic_bytes(
        target / "stage_receipts.json",
        _pretty_bytes(release_controls["stage_receipts"]),
    )
    _atomic_bytes(
        target / "run_ledger_receipt.json",
        _pretty_bytes(release_controls["run_ledger"]),
    )
    _atomic_bytes(
        target / "budget_receipt.json",
        _pretty_bytes(release_controls["budget"]),
    )
    _atomic_bytes(
        target / "audit" / "run_ledger.json",
        _pretty_bytes(run_ledger),
    )
    _atomic_bytes(
        target / "audit" / "budget_ledger.json",
        _pretty_bytes(budget_ledger),
    )
    if release_controls["post_gate"].get("available") is True:
        _atomic_bytes(
            target / "post_gate_authority.json",
            _pretty_bytes(
                _strict_json_load(Path(release_controls["post_gate"]["path"]))
            ),
        )
    if rule_sensitivity is not None:
        _atomic_bytes(
            target / "experiment_c_rule_sensitivity.json",
            _pretty_bytes(rule_sensitivity),
        )
    _atomic_bytes(
        target / "method_differences_scaffold.json",
        _pretty_bytes(_method_scaffold(contract_path.name)),
    )
    _atomic_bytes(
        target / "reviewer_report.md",
        _report(
            contract,
            denominator=denominator,
            gates=gates,
            capability=capability,
            cross_model=cross_model,
            release_controls=release_controls,
        ).encode("utf-8"),
    )

    scientific_matrix_complete = bool(
        denominator["pass"]
        and all(
            row["status"] == "complete" and row["scientific_eligible"] is True
            for row in rows
            if row["stage_id"] in V211_SCIENTIFIC_STAGES
        )
        and release_controls["post_gate"].get("go") is True
        and release_controls["budget"].get("pass") is True
    )
    claim_gates_supported = all(
        gates[name].get("status") == "supported"
        for name in ("experiment_a", "experiment_c", "experiment_d", "narrative")
    )
    scientific_complete = bool(scientific_matrix_complete and claim_gates_supported)
    published_files = sorted(
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    )
    manifest = {
        "schema_version": V2112_EVIDENCE_SCHEMA_VERSION,
        "evidence_namespace": _evidence_namespace(contract),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "resolved_git_commit": release_controls["resolved_git_commit"],
        "scientific_matrix_complete": scientific_matrix_complete,
        "scientific_claim_gates_supported": claim_gates_supported,
        "scientific_complete": scientific_complete,
        "claim_gates": _json_copy(gates),
        "published_files": published_files,
        "excluded_sources": [
            HISTORICAL_SCOPE,
            "V2.11.1 failed-preflight samples and provider journals",
            "V2.11.1 checkpoints, exactness receipts, and P95 authority",
            "diagnostic artifacts as scientific evidence",
            "raw prompts and raw provider outputs",
        ],
    }
    manifest_path = target / "package_manifest.json"
    _atomic_bytes(manifest_path, _pretty_bytes(manifest))
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
    checksums_path = target / "checksums.json"
    _atomic_bytes(checksums_path, _pretty_bytes(checksums))
    for row in checksums["files"]:
        path = target / row["path"]
        if _sha256_file(path) != row["sha256"]:
            raise PilotEvidenceError("V2.11.2 package checksum replay failed")
    package_bytes = sum(
        path.stat().st_size for path in target.rglob("*") if path.is_file()
    )
    if package_bytes + int(release_controls["budget"]["raw_root_storage_bytes"]) > int(
        contract.budgets["max_storage_bytes"]
    ):
        raise PilotEvidenceError("raw plus V2.11.2 package exceeds storage cap")
    return manifest_path, checksums_path, scientific_complete


def build_pilot_v2112_evidence_package(
    *,
    contract_path: str | Path,
    run_ledger_path: str | Path,
    raw_root: str | Path,
    build_root: str | Path,
    source_repo_root: str | Path | None = None,
) -> PilotEvidencePackage:
    """Build a provider-free V2.11.2 reviewer package from terminal evidence."""

    contract_source = Path(contract_path).resolve()
    contract = load_pilot_contract(contract_source)
    _frozen_contract(contract)
    raw = Path(raw_root).resolve()
    if not raw.is_dir():
        raise PilotEvidenceError(f"V2.11.2 raw root does not exist: {raw}")
    ledger_path = Path(run_ledger_path).resolve()
    if ledger_path != raw / "run_ledger.json":
        raise PilotEvidenceError("V2.11.2 publisher requires the raw-root ledger")

    code_root = Path(__file__).resolve().parents[1]
    repo_root = code_root
    source_context = nullcontext(None)
    if source_repo_root is not None:
        candidate = Path(source_repo_root)
        if candidate.is_symlink():
            raise PilotEvidenceError("V2.11.2 source repository cannot be a symlink")
        candidate = candidate.resolve(strict=True)
        if not candidate.is_dir():
            raise PilotEvidenceError("V2.11.2 source repository is not a directory")
        if not raw.is_relative_to(candidate):
            raise PilotEvidenceError("V2.11.2 raw root escaped its source checkout")
        repo_root = candidate
        source_context = nullcontext(candidate)

    release_control, commit, paid = _validate_release(contract, raw_root=raw)
    ledger = _strict_json_load(ledger_path)
    with source_context as source_root:
        rows, denominator = _normalize_v2112_ledger(
            contract,
            ledger,
            raw_root=raw,
            expected_commit=commit,
            source_repo_root=source_root,
        )
    ledger_object = PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    stage_receipts = _validate_stage_receipts(
        contract,
        raw_root=raw,
        ledger=ledger_object,
        paid=paid,
    )
    post_gate = _validate_post_gate(
        contract,
        raw_root=raw,
        commit=commit,
        rows=rows,
    )
    budget = _validate_budget(
        contract,
        raw_root=raw,
        repo_root=repo_root,
        rows=rows,
    )
    run_receipt = _run_ledger_receipt(
        contract,
        ledger,
        denominator,
        path=ledger_path,
    )

    gates = {
        "experiment_a": _experiment_a_gate(contract, rows),
        "experiment_c": _experiment_c_gate(contract, rows),
        "experiment_d": _experiment_d_gate(contract, rows),
        "narrative": _narrative_gate(contract, rows),
    }
    capability = _capability_by_model(rows, contract)
    cross_model = _cross_model_summary(contract, rows, capability)
    experiment_b = _experiment_b_summary(rows)
    rule_sensitivity, sensitivity_control = _validated_experiment_c_sensitivity(
        contract,
        raw_root=raw,
        rows=rows,
        common_commit=commit,
    )
    release_controls = {
        "pass": True,
        "resolved_git_commit": commit,
        "release": release_control,
        "run_ledger": run_receipt,
        "stage_receipts": stage_receipts,
        "post_gate": post_gate,
        "budget": budget,
        "experiment_c_sensitivity": sensitivity_control,
        "historical_import_boundary": {
            "source_contract": "finevo-pilot-v2.11.1",
            "failed_preflight_calls_retained_for_budget_audit": 64,
            "failed_preflight_samples_admitted": 0,
            "failed_preflight_checkpoints_admitted": 0,
            "failed_preflight_p95_authorities_admitted": 0,
            "treatment_effect_cells_imported": 0,
            "scientific_evidence": False,
        },
    }

    base = Path(build_root).resolve()
    target = base / _evidence_namespace(contract)
    if target.exists():
        raise PilotEvidenceError(f"refusing to overwrite evidence package: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-build-", dir=target.parent)
    )
    try:
        manifest, checksums, scientific_complete = _write_package(
            temporary,
            contract_path=contract_source,
            contract=contract,
            rows=rows,
            denominator=denominator,
            gates=gates,
            capability=capability,
            cross_model=cross_model,
            release_controls=release_controls,
            experiment_b=experiment_b,
            rule_sensitivity=rule_sensitivity,
            run_ledger=ledger,
            budget_ledger=_strict_json_load(raw / "budget_ledger.json"),
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PilotEvidencePackage(
        package_dir=target,
        manifest_path=target / manifest.name,
        checksums_path=target / checksums.name,
        contract_hash=contract.canonical_hash,
        scientific_complete=scientific_complete,
        claim_gates=_json_copy(gates),
    )


__all__ = [
    "V2112_CONTRACT_ID",
    "V2112_EVIDENCE_SCHEMA_VERSION",
    "build_pilot_v2112_evidence_package",
]
