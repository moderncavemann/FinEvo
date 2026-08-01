"""Deterministic, zero-provider reviewer trace for the sealed V2.11.5 run.

This module is a publication-time consumer.  It selects one coordinate from
the frozen contract before reading any result stream, verifies the complete
source hash chain, and emits a descriptive state-to-memory-to-action-to-state
trace.  It never imports a provider implementation, reads a provider key,
dispatches a request, retries a run, or searches for a replacement example.

The trace is deliberately not stage-authoritative.  Experiment A remains an
immutable ``complete-with-no-go`` stage with three retained ITT failures.
``publication_provider_calls=0`` refers only to this offline build; fields
named ``frozen_source_provider_call`` are copied historical observations from
the sealed run and are never newly dispatched by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator

from .pilot_contract import PilotContract, canonical_sha256, load_pilot_contract


TRACE_SCHEMA_VERSION = "finevo-reviewer-closed-loop-trace-v1"
TRACE_SCHEMA_RELATIVE = Path("schemas/reviewer_closed_loop_trace_v1.schema.json")
TRACE_FILENAME = "reviewer_closed_loop_trace.json"
TRACE_CHECKSUMS_FILENAME = "checksums.json"
TRACE_CANONICALIZATION = "json-sort-keys-utf8-v1"
TRACE_POLICY_ID = "finevo-contract-indexed-first-recovery-trace-v1"
TRACE_SOURCE_ID = "sealed-science:pilot-v2.11.5-science"
TRACE_PUBLISHER_ID = "publication-consumer:reviewer-trace-v1"

V2115_CONTRACT_ID = "finevo-pilot-v2.11.5"
V2115_CONTRACT_RELATIVE = Path("experiments/pilot_v2_11_5.yaml")
V2115_RAW_RELATIVE = Path("experiment_results/pilot-v2.11.5/raw")
V2115_SOURCE_TAG = "pilot-v2.11.5-science"
V2115_SOURCE_COMMIT = "2351ac2283f9fedb9dce70067174020be56ed9cc"
V2115_SOURCE_TAG_OBJECT = "bccfb13cee7d592470d1873cfacc3b12bed38be4"
V2115_CONTRACT_FILE_SHA256 = (
    "b96438430231f0c46fd6c5f15ba749713534feb15f964c496aa02606cf11103b"
)
V2115_CONTRACT_CANONICAL_SHA256 = (
    "e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef"
)
V2115_EXPERIMENT_A_RECEIPT_FILE_SHA256 = (
    "8193f3449663f63c9cf0c881ee5e7759d2682f320f214c4941040489c81734f9"
)
V2115_EXPERIMENT_A_RECEIPT_CONTENT_SHA256 = (
    "177dc8ce4d1957eac0734bb1716279676f77931e30b3a1d10dd2c138a43a5457"
)
V2115_SELECTED_RUN_MANIFEST_SHA256 = (
    "b0589053dbdbee7900050c135266809481e7c0307e96b8434e2a16e5b33ad35f"
)
PUBLICATION_BASE_COMMIT = "34134f2624833e45f0e1f559332b0d11ea1942d6"

_RUN_STREAMS = (
    "actions.jsonl",
    "api_usage.jsonl",
    "context_trace.jsonl",
    "decision_snapshots.jsonl",
    "episodes.jsonl",
    "macro_steps.jsonl",
    "semantic_proposals.jsonl",
    "semantic_rule_events.jsonl",
    "semantic_rules.jsonl",
    "shock_events.jsonl",
    "utility_ledger.jsonl",
)
_RUN_METADATA = ("config.json", "provenance.json", "schemas.json")
_REQUIRED_PUBLISHER_FILES = (
    TRACE_SCHEMA_RELATIVE.as_posix(),
    "docs/reviewer_closed_loop_trace_v1.md",
    "requirements.txt",
    "scripts/build_v2115_reviewer_trace.py",
    "tests/test_reviewer_closed_loop_trace.py",
    "verified_memory/reviewer_closed_loop_trace.py",
)


class ReviewerTraceError(RuntimeError):
    """Raised when the sealed source or a cross-stream binding is invalid."""


class ReviewerTraceUnavailable(ReviewerTraceError):
    """Raised when the fixed coordinate cannot produce a complete trace."""


@dataclass(frozen=True, slots=True)
class JsonlRecord:
    line_number: int
    raw_line_sha256: str
    value: Mapping[str, Any]


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewerTraceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReviewerTraceError(f"non-finite JSON constant is forbidden: {value}")


def _loads_strict(raw: bytes, *, name: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewerTraceError(f"invalid UTF-8 JSON in {name}") from exc


def _load_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReviewerTraceError(f"JSON source is missing or unsafe: {path}")
    value = _loads_strict(path.read_bytes(), name=str(path))
    if not isinstance(value, Mapping):
        raise ReviewerTraceError(f"JSON source must contain an object: {path}")
    return value


def _load_jsonl(path: Path) -> tuple[JsonlRecord, ...]:
    if path.is_symlink() or not path.is_file():
        raise ReviewerTraceError(f"JSONL source is missing or unsafe: {path}")
    records: list[JsonlRecord] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line.strip():
            raise ReviewerTraceError(f"blank JSONL row at {path}:{line_number}")
        value = _loads_strict(raw_line, name=f"{path}:{line_number}")
        if not isinstance(value, Mapping):
            raise ReviewerTraceError(
                f"JSONL row must contain an object: {path}:{line_number}"
            )
        records.append(
            JsonlRecord(
                line_number=line_number,
                raw_line_sha256=_sha256_bytes(raw_line),
                value=value,
            )
        )
    return tuple(records)


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReviewerTraceError(
            f"git provenance check failed: {' '.join(args)}"
        ) from exc
    return result.stdout.strip()


def _git_blob_oid(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _absolute_without_symlink(path: str | Path, *, name: str) -> Path:
    expanded = Path(path).expanduser()
    candidate = Path(os.path.abspath(os.fspath(expanded)))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ReviewerTraceError(f"{name} crosses a symlink: {current}")
        if not current.exists():
            break
    return candidate


_CENTRAL_LOADER_PROGRAM = """
import json
from pathlib import Path
import sys
from verified_memory.pilot_contract import canonical_sha256
from verified_memory.runner_artifacts import load_verified_run_artifacts

run_dir = Path(sys.argv[1])
authority = Path(sys.argv[2])
result = load_verified_run_artifacts(run_dir, authority_repo_root=authority)
print(json.dumps({
    "budget_snapshot_sha256": canonical_sha256(result.budget_snapshot),
    "config_sha256": canonical_sha256(result.config),
    "record_counts": {
        name: len(rows) for name, rows in sorted(result.records.items())
    },
    "result_complete": bool(result.summary.get("result_complete")),
    "records_sha256": canonical_sha256({
        name: list(rows) for name, rows in sorted(result.records.items())
    }),
    "run_id": result.config.get("run_id"),
    "runner_schema_version": result.config.get("schema_version"),
    "summary_sha256": canonical_sha256(result.summary),
    "validation_status_sha256": canonical_sha256(result.validation_status),
}, sort_keys=True, separators=(",", ":")))
""".strip()


def _central_loader_receipt(
    *,
    run_dir: Path,
    authority_repo_root: Path,
    publisher_repo_root: Path,
) -> Mapping[str, Any]:
    environment = {
        "PATH": os.defpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _CENTRAL_LOADER_PROGRAM,
                str(run_dir),
                str(authority_repo_root),
            ],
            cwd=publisher_repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ReviewerTraceError("central verified-run loader rejected the source") from exc
    try:
        receipt = _loads_strict(
            completed.stdout.strip().encode("utf-8"),
            name="central verified-run loader receipt",
        )
    except ReviewerTraceError as exc:
        raise ReviewerTraceError("central verified-run loader returned invalid output") from exc
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("result_complete") is not True
        or receipt.get("runner_schema_version") != "verified-simulation-runner-v3"
        or not isinstance(receipt.get("record_counts"), Mapping)
    ):
        raise ReviewerTraceError("central verified-run loader receipt is incomplete")
    return receipt


def _has_symlink_component(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _verify_source_checkout(source_repo_root: Path) -> dict[str, Any]:
    if source_repo_root.is_symlink():
        raise ReviewerTraceError("science source repository cannot be a symlink")
    try:
        source = source_repo_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ReviewerTraceError("science source repository does not exist") from exc
    if not source.is_dir():
        raise ReviewerTraceError("science source repository is not a directory")
    if Path(_git(source, "rev-parse", "--show-toplevel")).resolve() != source:
        raise ReviewerTraceError("science source must be the exact git top-level")
    head = _git(source, "rev-parse", "HEAD")
    branch = _git(source, "rev-parse", "--abbrev-ref", "HEAD")
    tag_type = _git(source, "cat-file", "-t", f"refs/tags/{V2115_SOURCE_TAG}")
    tag_object = _git(
        source, "rev-parse", f"refs/tags/{V2115_SOURCE_TAG}^{{object}}"
    )
    tag_commit = _git(
        source, "rev-parse", f"refs/tags/{V2115_SOURCE_TAG}^{{commit}}"
    )
    tracked_status = _git(source, "status", "--porcelain", "--untracked-files=no")
    if (
        head != V2115_SOURCE_COMMIT
        or branch != "HEAD"
        or tag_type != "tag"
        or tag_object != V2115_SOURCE_TAG_OBJECT
        or tag_commit != V2115_SOURCE_COMMIT
        or tracked_status
    ):
        raise ReviewerTraceError(
            "science source is not the detached, tracked-clean V2.11.5 tag"
        )
    contract_path = source / V2115_CONTRACT_RELATIVE
    raw_root = source / V2115_RAW_RELATIVE
    for path in (contract_path, raw_root):
        if _has_symlink_component(source, path):
            raise ReviewerTraceError(f"science source crosses a symlink: {path}")
    if not contract_path.is_file() or not raw_root.is_dir():
        raise ReviewerTraceError("science contract or raw root is missing")
    return {
        "source_id": TRACE_SOURCE_ID,
        "git_tag": V2115_SOURCE_TAG,
        "tag_object": V2115_SOURCE_TAG_OBJECT,
        "resolved_git_commit": V2115_SOURCE_COMMIT,
        "detached_head": True,
        "tracked_worktree_clean": True,
    }


def _verify_publisher_checkout(code_root: Path) -> dict[str, Any]:
    if code_root.is_symlink():
        raise ReviewerTraceError("publisher repository cannot be a symlink")
    root = code_root.resolve(strict=True)
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ReviewerTraceError("publisher path must be the exact git top-level")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ReviewerTraceError("reviewer-trace publisher must be tracked-clean")
    head = _git(root, "rev-parse", "HEAD")
    _git(root, "merge-base", "--is-ancestor", PUBLICATION_BASE_COMMIT, head)
    blobs: dict[str, str] = {}
    for relative in _REQUIRED_PUBLISHER_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ReviewerTraceError(f"publisher file is not tracked: {relative}")
        _git(root, "diff", "--quiet", "HEAD", "--", relative)
        blobs[relative] = _git(root, "rev-parse", f"HEAD:{relative}")
    return {
        "publisher_id": TRACE_PUBLISHER_ID,
        "git_commit": head,
        "tracked_worktree_clean": True,
        "implementation_base_commit": PUBLICATION_BASE_COMMIT,
        "required_tracked_head_blobs": blobs,
        "publication_provider_calls": 0,
    }


def _verify_content_hash(value: Mapping[str, Any], *, expected: str, name: str) -> None:
    candidate = _json_copy(value)
    integrity = candidate.pop("integrity", None)
    if not isinstance(integrity, Mapping):
        raise ReviewerTraceError(f"{name} has no integrity object")
    if (
        integrity
        != {
            "canonicalization": TRACE_CANONICALIZATION,
            "content_sha256": expected,
        }
        or canonical_sha256(candidate) != expected
    ):
        raise ReviewerTraceError(f"{name} content hash drifted")


def derive_selection(contract: PilotContract) -> tuple[Any, dict[str, Any]]:
    """Select a coordinate from contract structure without reading results."""

    if (
        contract.contract_id != V2115_CONTRACT_ID
        or contract.canonical_hash != V2115_CONTRACT_CANONICAL_SHA256
        or contract.status != "frozen"
    ):
        raise ReviewerTraceError("reviewer trace requires the frozen V2.11.5 contract")
    stage = contract.stage("experiment-a")
    if (
        stage.seed_set != "main"
        or stage.shock_id != "registered-rate-shock"
        or stage.num_agents < 1
        or tuple(contract.models_for_stage("experiment-a")) != ("gpt52_main",)
        or "full" not in contract.arms_for_stage("experiment-a")
        or "none" not in {item for cell in stage.cells for item in cell.narratives}
        or "stage0-selected" not in stage.utility_profiles
    ):
        raise ReviewerTraceError("Experiment A selection inputs drifted")
    main_seeds = tuple(int(item) for item in contract.seeds["sets"]["main"])
    if not main_seeds:
        raise ReviewerTraceError("main seed set is empty")
    recovery = sorted(
        (
            row
            for row in contract.shocks[stage.shock_id]["schedule"]
            if row.get("phase") == "recovery"
        ),
        key=lambda row: int(row["start"]),
    )
    if not recovery:
        raise ReviewerTraceError("registered shock has no recovery interval")
    decision_t = int(recovery[0]["start"])
    if decision_t + 1 > int(recovery[0]["end"]):
        raise ReviewerTraceError("first recovery interval has no continuation step")
    specs = tuple(
        spec
        for spec in contract.expand(stage="experiment-a", model="gpt52_main", arm="full")
        if spec.narrative_id == "none"
        and spec.utility_profile_id == "stage0-selected"
        and spec.environment_seed == main_seeds[0]
    )
    if len(specs) != 1:
        raise ReviewerTraceError("contract-indexed trace coordinate is not unique")
    spec = specs[0]
    agent_id = min(range(stage.num_agents))
    selection = {
        "policy_id": TRACE_POLICY_ID,
        "selection_timing": "publication-time-post-seal",
        "preregistered": False,
        "human_prior_case_awareness": True,
        "outcome_fields_used_by_selector": False,
        "stage_rule": "experiment-a",
        "model_rule": "sole-primary-model-in-experiment-a",
        "arm_rule": "full-mechanism-arm",
        "narrative_rule": "none",
        "utility_rule": "stage0-selected",
        "seed_rule": "first-contract-main-seed",
        "agent_rule": "minimum-agent-id",
        "decision_t_rule": "start-of-first-registered-recovery-interval",
        "next_t_rule": "decision-t-plus-one",
        "fallback_policy": "none-emit-unavailable",
        "contract_pointers": {
            "stage": "/stages/4",
            "seed": "/seeds/sets/main/0",
            "recovery": "/shocks/registered-rate-shock/schedule/2/start",
        },
        "excluded_selection_fields": [
            "actions",
            "clipping",
            "macro_deltas",
            "parse_or_provider_status",
            "retrieval_scores_or_order",
            "rule_events_or_status",
            "selected_rule_ids",
            "utility_or_utility_advantage",
            "wealth",
        ],
        "selected_coordinates": {
            "stage_id": spec.stage_id,
            "model_id": spec.model_id,
            "arm_id": spec.arm_id,
            "narrative_id": spec.narrative_id,
            "utility_profile_id": spec.utility_profile_id,
            "seed": spec.environment_seed,
            "agent_id": agent_id,
            "decision_t": decision_t,
            "outcome_t": decision_t + 1,
            "next_decision_t": decision_t + 1,
            "run_id": spec.run_id,
        },
    }
    selection["policy_sha256"] = canonical_sha256(selection)
    return spec, selection


def _select_record(
    records: Sequence[JsonlRecord],
    *,
    source_id: str,
    selector: Mapping[str, Any],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> JsonlRecord:
    matches = [record for record in records if predicate(record.value)]
    if not matches:
        raise ReviewerTraceUnavailable(
            f"fixed coordinate has no {source_id} row for {dict(selector)}"
        )
    if len(matches) != 1:
        raise ReviewerTraceError(
            f"fixed coordinate has duplicate {source_id} rows for {dict(selector)}"
        )
    return matches[0]


def _source_record_binding(
    source_id: str,
    record: JsonlRecord,
    *,
    selector: Mapping[str, Any],
    copied_to: Sequence[str],
    native_record_id: str | None = None,
    native_record_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "line_number": record.line_number,
        "selector": _json_copy(selector),
        "raw_line_sha256": record.raw_line_sha256,
        "native_record_id": native_record_id,
        "native_record_hash": native_record_hash,
        "copied_to": list(copied_to),
    }


def _retrieval_items(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids = context.get("retrieved_episode_ids")
    scores = context.get("retrieval_scores")
    components = context.get("score_components")
    if not isinstance(ids, list) or not isinstance(scores, list) or not isinstance(
        components, list
    ):
        raise ReviewerTraceError("retrieval arrays are missing")
    if not (len(ids) == len(scores) == len(components)):
        raise ReviewerTraceError("retrieval arrays have different lengths")
    result: list[dict[str, Any]] = []
    for rank, (episode_id, score, component) in enumerate(
        zip(ids, scores, components), start=1
    ):
        if not isinstance(component, Mapping):
            raise ReviewerTraceError("retrieval score component is not an object")
        result.append(
            {
                "rank": rank,
                "episode_id": str(episode_id),
                "score": float(score),
                "context_similarity": float(component["context_similarity"]),
                "state_similarity": float(component["state_similarity"]),
                "recency": float(component["recency"]),
                "importance": float(component["importance"]),
            }
        )
    return result


def _macro_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": int(value["outcome_t"]),
        "price": float(value["price"]),
        "monthly_inflation": float(value["monthly_inflation"]),
        "average_wealth": float(value["average_wealth"]),
        "low_labor_rate": float(value["low_labor_rate"]),
    }


def _context(value: Mapping[str, Any]) -> dict[str, Any]:
    packet = value.get("context_packet")
    if not isinstance(packet, Mapping):
        raise ReviewerTraceError("context packet is missing")
    return {
        "context_id": value["context_id"],
        "context_hash": value["context_hash"],
        "source_hash": packet["source_hash"],
        "feature_schema_version": packet["feature_schema_version"],
        "history_start": packet["history_start"],
        "observation_count": packet["observation_count"],
        "observed_through": packet["observed_through"],
        "prompt_summary": packet["prompt_summary"],
        "context_to_prompt": value["context_to_prompt"],
        "context_to_retrieval": value["context_to_retrieval"],
    }


def _action(value: Mapping[str, Any]) -> dict[str, Any]:
    decision = value.get("decision")
    if not isinstance(decision, Mapping):
        raise ReviewerTraceError("action decision is missing")
    return {
        "proposed_work_fraction": float(decision["proposed_work_fraction"]),
        "proposed_consumption_fraction": float(
            decision["proposed_consumption_fraction"]
        ),
        "executed_labor_hours": float(decision["executed_labor_hours"]),
        "executed_consumption_rate": float(decision["executed_consumption_rate"]),
        "labor_action_index": int(decision["labor_action_index"]),
        "consumption_action_index": int(decision["consumption_action_index"]),
        "clipped": bool(decision["clipped"]),
        "repair_attempts": int(decision["repair_attempts"]),
        "reflection": str(decision["reflection"]),
    }


def _provider_call(
    value: Mapping[str, Any], *, parse_mode: str | None = None
) -> dict[str, Any]:
    identity = value.get("request_artifact_identity")
    dispatch = value.get("parameter_dispatch")
    if not isinstance(identity, Mapping) or not isinstance(dispatch, Mapping):
        raise ReviewerTraceError("provider-call identity is missing")
    return {
        "provider": value["provider"],
        "request_profile_id": value["request_profile_id"],
        "requested_model": value["model"],
        "served_snapshot": identity["served_snapshot"],
        "response_model": value["response_model"],
        "response_provider": value["response_provider"],
        "response_route": value["response_route"],
        "attempts": int(value["attempts"]),
        "response_completed": bool(value["response_completed"]),
        "output_disposition": value["output_disposition"],
        "parse_mode": value.get("action_parse_mode", parse_mode),
        "prompt_hash": value["prompt_hash"],
        "raw_output_hash": value["raw_output_hash"],
        "seed_dispatch": dispatch["seed"],
        "request_seed": value["request_seed"],
    }


def _rule_event(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": value["event_id"],
        "timestamp": value["timestamp"],
        "event_type": value["event_type"],
        "rule_id": value["rule_id"],
        "candidate_id": value["candidate_id"],
        "episode_ids": _json_copy(value["episode_ids"]),
        "from_status": value["from_status"],
        "to_status": value["to_status"],
        "reason": value["reason"],
        "metrics": _json_copy(value["metrics"]),
        "provenance": _json_copy(value["provenance"]),
    }


def _numeric_predicate_matches(
    actual: Any,
    *,
    operator: Any,
    expected: Any,
    tolerance: Any,
) -> bool:
    try:
        observed = float(actual)
        target = float(expected)
        slack = float(tolerance)
    except (TypeError, ValueError):
        return False
    if operator == ">":
        return observed > target
    if operator == ">=":
        return observed >= target
    if operator == "<":
        return observed < target
    if operator == "<=":
        return observed <= target
    if operator == "==":
        return abs(observed - target) <= slack
    return False


def _recompute_candidate_identity(
    *,
    proposal: Mapping[str, Any],
    candidate_event: Mapping[str, Any],
    outcome_criterion: Mapping[str, Any],
) -> dict[str, Any]:
    raw_output = proposal.get("raw_output")
    if not isinstance(raw_output, str):
        raise ReviewerTraceError("semantic proposal raw output is missing")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ReviewerTraceError("semantic proposal raw output is not exact JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ReviewerTraceError("semantic proposal raw output is not an object")
    required = {
        "action_guidance",
        "condition",
        "context_scope",
        "rationale",
        "supporting_episode_ids",
    }
    if set(parsed) != required:
        raise ReviewerTraceError("semantic proposal keys drifted")
    provenance = candidate_event.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ReviewerTraceError("candidate verification provenance is missing")
    raw_response_hash = canonical_sha256({"raw_response": raw_output})
    generator_id = provenance.get("generator_id")
    migration_notes = provenance.get("migration_notes")
    content = {
        "context_scope": _json_copy(parsed["context_scope"]),
        "condition": _json_copy(parsed["condition"]),
        "action_guidance": _json_copy(parsed["action_guidance"]),
        "outcome_criterion": _json_copy(outcome_criterion),
        "rationale": parsed["rationale"],
        "supporting_episode_ids": _json_copy(parsed["supporting_episode_ids"]),
        "generator_id": generator_id,
        "raw_response_hash": raw_response_hash,
        "migration_notes": _json_copy(migration_notes),
    }
    return {
        "candidate_id": f"cand-{canonical_sha256(content)[:20]}",
        "generator_id": generator_id,
        "migration_notes": _json_copy(migration_notes),
        "provider_raw_output_sha256": hashlib.sha256(
            raw_output.encode("utf-8")
        ).hexdigest(),
        "raw_response_hash": raw_response_hash,
        "supporting_episode_ids": _json_copy(parsed["supporting_episode_ids"]),
    }


def _recompute_rule_classification(
    rule: Mapping[str, Any], episode: Mapping[str, Any]
) -> dict[str, Any]:
    pre_state = episode.get("pre_state")
    executed_action = episode.get("executed_action")
    if not isinstance(pre_state, Mapping) or not isinstance(executed_action, Mapping):
        raise ReviewerTraceError("episode is missing verifier inputs")
    context_scope = rule.get("context_scope")
    condition = rule.get("condition")
    guidance = rule.get("action_guidance")
    criterion = rule.get("outcome_criterion")
    if not all(
        isinstance(item, Mapping)
        for item in (context_scope, condition, guidance, criterion)
    ):
        raise ReviewerTraceError("selected rule verifier predicates are malformed")
    scope_matches = all(
        isinstance(predicate, Mapping)
        and _numeric_predicate_matches(
            pre_state.get(predicate.get("field")),
            operator=predicate.get("operator"),
            expected=predicate.get("value"),
            tolerance=predicate.get("tolerance"),
        )
        for predicate in context_scope.get("predicates", [])
    )
    condition_matches = _numeric_predicate_matches(
        pre_state.get(condition.get("field")),
        operator=condition.get("operator"),
        expected=condition.get("value"),
        tolerance=condition.get("tolerance"),
    )
    target = guidance.get("target")
    direction = guidance.get("direction")
    actual_action = executed_action.get(target)
    if direction == "at_least":
        guidance_operator = ">="
    elif direction == "at_most":
        guidance_operator = "<="
    elif direction == "approximately":
        guidance_operator = "=="
    else:
        raise ReviewerTraceError("selected rule guidance direction is unsupported")
    guidance_compliant = _numeric_predicate_matches(
        actual_action,
        operator=guidance_operator,
        expected=guidance.get("threshold"),
        tolerance=guidance.get("tolerance"),
    )
    metric = criterion.get("metric")
    observed_outcome = (
        episode.get("outcome", {}).get("wealth_change")
        if metric == "wealth_change"
        else episode.get(metric)
    )
    outcome_passed = _numeric_predicate_matches(
        observed_outcome,
        operator=criterion.get("operator"),
        expected=criterion.get("value"),
        tolerance=criterion.get("tolerance"),
    )
    if not scope_matches or not condition_matches:
        classification = "irrelevant"
    elif guidance_compliant and outcome_passed:
        classification = "support"
    elif guidance_compliant:
        classification = "harmful_compliance"
    elif outcome_passed:
        classification = "alternative_success"
    else:
        classification = "alternative_failure"
    return {
        "scope_matches": scope_matches,
        "condition_matches": condition_matches,
        "guidance_compliant": guidance_compliant,
        "guidance_observed_value": actual_action,
        "outcome_metric": metric,
        "outcome_observed_value": observed_outcome,
        "outcome_passed": outcome_passed,
        "classification": classification,
    }


def _artifact_content_sha256(value: Mapping[str, Any]) -> str:
    candidate = _json_copy(value)
    integrity = candidate.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("content_sha256", None)
    return canonical_sha256(candidate)


def _contains_host_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in {"source_repo_root", "repository_root", "worktree_root"}
            or _contains_host_absolute_path(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_host_absolute_path(item) for item in value)
    if isinstance(value, str):
        return value.startswith(("/Users/", "/home/", "/var/", "/tmp/"))
    return False


def validate_trace_artifact(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "artifact_id",
        "status",
        "publication_provider_calls",
        "selection_policy",
        "evidence_scope",
        "provenance",
        "source_files",
        "source_records",
        "trace",
        "failure_reason",
        "link_checks",
        "claim_boundary",
        "integrity",
    }
    if set(value) != required:
        raise ReviewerTraceError("reviewer trace top-level schema drifted")
    selection = value.get("selection_policy")
    scope = value.get("evidence_scope")
    boundary = value.get("claim_boundary")
    integrity = value.get("integrity")
    provenance = value.get("provenance")
    selection_body = _json_copy(selection) if isinstance(selection, Mapping) else {}
    selection_hash = selection_body.pop("policy_sha256", None)
    if (
        value.get("schema_version") != TRACE_SCHEMA_VERSION
        or value.get("publication_provider_calls") != 0
        or not isinstance(selection, Mapping)
        or selection.get("policy_id") != TRACE_POLICY_ID
        or selection.get("selection_timing") != "publication-time-post-seal"
        or selection.get("preregistered") is not False
        or selection.get("outcome_fields_used_by_selector") is not False
        or selection.get("fallback_policy") != "none-emit-unavailable"
        or selection_hash != canonical_sha256(selection_body)
        or not isinstance(scope, Mapping)
        or scope
        != {
            "diagnostic_only": True,
            "descriptive_only": True,
            "effectiveness_evidence": False,
            "frozen_source_provider_call_scope": (
                "historical-observations-read-from-sealed-logs"
            ),
            "publication_provider_calls": 0,
            "stage_authoritative": False,
        }
        or not isinstance(boundary, Mapping)
        or boundary.get("no_causal_attribution") is not True
        or boundary.get("stage_decision_unchanged") is not True
        or not isinstance(provenance, Mapping)
        or _contains_host_absolute_path(provenance)
        or not isinstance(integrity, Mapping)
        or set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != TRACE_CANONICALIZATION
        or integrity.get("content_sha256") != _artifact_content_sha256(value)
    ):
        raise ReviewerTraceError("reviewer trace invariant failed")
    status = value.get("status")
    if status == "complete":
        checks = value.get("link_checks")
        trace = value.get("trace")
        if (
            not isinstance(trace, Mapping)
            or value.get("failure_reason") is not None
            or not isinstance(checks, Mapping)
            or checks.get("all_pass") is not True
            or len(checks.get("checks", {})) != 17
            or not all(checks.get("checks", {}).values())
            or len(value.get("source_files", [])) != 17
            or len(value.get("source_records", [])) != 23
            or provenance.get("stage_status") != "complete-with-no-go"
            or provenance.get("stage_go") is not False
            or provenance.get("stage_scientific_matrix_complete") is not False
            or provenance.get("stage_registered_runs") != 20
            or provenance.get("stage_complete_runs") != 17
            or provenance.get("stage_failed_runs") != 3
            or "frozen_source_provider_call" not in trace
            or "frozen_source_provider_call"
            not in trace.get("next_decision", {})
            or trace.get("memory_update", {})
            .get("verifier_recomputation", {})
            .get("classification")
            != "harmful_compliance"
        ):
            raise ReviewerTraceError("complete reviewer trace has failed links")
    elif status == "unavailable":
        if value.get("trace") is not None or not value.get("failure_reason"):
            raise ReviewerTraceError("unavailable reviewer trace is malformed")
    else:
        raise ReviewerTraceError("unknown reviewer trace status")


def _validate_against_runtime_schema(
    value: Mapping[str, Any], schema_bytes: bytes
) -> None:
    schema = _loads_strict(schema_bytes, name="reviewer trace JSON Schema")
    if not isinstance(schema, Mapping):
        raise ReviewerTraceError("reviewer trace JSON Schema root is not an object")
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    except Exception as exc:
        raise ReviewerTraceError("reviewer trace JSON Schema is invalid") from exc
    if errors:
        pointer = "/" + "/".join(str(part) for part in errors[0].absolute_path)
        raise ReviewerTraceError(
            f"reviewer trace failed runtime schema validation at {pointer}"
        )


def _seal_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(value)
    result["integrity"] = {"canonicalization": TRACE_CANONICALIZATION}
    result["integrity"]["content_sha256"] = _artifact_content_sha256(result)
    validate_trace_artifact(result)
    return result


def _claim_boundary() -> dict[str, Any]:
    return {
        "selection_not_preregistered": True,
        "human_prior_case_awareness_disclosed": True,
        "single_observational_trace": True,
        "no_causal_attribution": True,
        "not_representative": True,
        "stage_decision_unchanged": True,
        "allowed": [
            "The sealed logs expose an observed state-retrieval-decision-outcome-memory-update path at the fixed coordinate.",
            "The t=8 episode was ingested as harmful compliance, the selected rule was retired, and that episode was retrieved at t=9.",
            "The frozen source provider-call records describe historical execution; this publication build made zero provider calls.",
        ],
        "forbidden": [
            "The selected semantic rule caused the action or the macro-state change.",
            "The verifier is effective, robust, representative, or hallucination-free.",
            "The t=9 action change was caused by retirement of the rule.",
            "Experiment A passed or its immutable no-go was reversed.",
            "This publication-time selection was preregistered.",
        ],
    }


def _source_file(
    source: Path,
    *,
    source_id: str,
    path: Path,
    binding_kind: str,
    line_count: int | None = None,
) -> dict[str, Any]:
    result = {
        "source_id": source_id,
        "relative_path": path.relative_to(source).as_posix(),
        "file_sha256": _sha256_file(path),
        "binding_kind": binding_kind,
    }
    if line_count is not None:
        result["line_count"] = line_count
    return result


def _fixed_run_manifest_path(raw_root: Path, run_id: str) -> Path:
    manifest_path = raw_root / "experiment-a" / "runs" / run_id / "manifest.json"
    if not manifest_path.is_file():
        raise ReviewerTraceUnavailable(
            "the contract-indexed run manifest is absent; no fallback was attempted"
        )
    return manifest_path


def _extract_trace(
    *,
    source: Path,
    publisher_root: Path,
    raw_root: Path,
    spec: Any,
    selection: Mapping[str, Any],
    contract: PilotContract,
    source_provenance: Mapping[str, Any],
    publisher_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    coordinates = selection["selected_coordinates"]
    agent_id = int(coordinates["agent_id"])
    decision_t = int(coordinates["decision_t"])
    next_t = int(coordinates["next_decision_t"])
    run_id = str(coordinates["run_id"])
    receipt_path = raw_root / "experiment-a" / "stage_receipt.json"
    if _sha256_file(receipt_path) != V2115_EXPERIMENT_A_RECEIPT_FILE_SHA256:
        raise ReviewerTraceError("Experiment A stage receipt file hash drifted")
    receipt = _load_json(receipt_path)
    _verify_content_hash(
        receipt,
        expected=V2115_EXPERIMENT_A_RECEIPT_CONTENT_SHA256,
        name="Experiment A stage receipt",
    )
    if (
        receipt.get("stage_id") != "experiment-a"
        or receipt.get("status") != "complete-with-no-go"
        or receipt.get("terminal") is not True
        or receipt.get("denominator_terminal") is not True
        or receipt.get("go") is not False
        or receipt.get("scientific_matrix_complete") is not False
        or receipt.get("registered_run_count") != 20
        or receipt.get("complete_cell_count") != 17
        or receipt.get("status_counts") != {"complete": 17, "failed": 3}
    ):
        raise ReviewerTraceError("Experiment A no-go boundary drifted")

    manifest_path = _fixed_run_manifest_path(raw_root, run_id)
    run_dir = manifest_path.parent
    if _sha256_file(manifest_path) != V2115_SELECTED_RUN_MANIFEST_SHA256:
        raise ReviewerTraceError("selected run manifest hash drifted")
    central_loader_before = _central_loader_receipt(
        run_dir=run_dir,
        authority_repo_root=source,
        publisher_repo_root=publisher_root,
    )
    if central_loader_before.get("run_id") != run_id:
        raise ReviewerTraceError("central verified-run loader selected another run")
    manifest = _load_json(manifest_path)
    validation = manifest.get("validation_status")
    result = manifest.get("result")
    if (
        not isinstance(validation, Mapping)
        or validation.get("status") != "pass"
        or validation.get("diagnostic_only") is not False
        or validation.get("scientific_evidence") is not True
        or not isinstance(result, Mapping)
        or result.get("complete") is not True
        or result.get("required_streams_present") is not True
    ):
        raise ReviewerTraceError("selected run manifest is not complete and valid")
    receipt_sources = receipt.get("bindings", {}).get("source_files", [])
    bound = [
        row
        for row in receipt_sources
        if isinstance(row, Mapping)
        and row.get("file_sha256") == V2115_SELECTED_RUN_MANIFEST_SHA256
    ]
    if len(bound) != 1 or Path(str(bound[0].get("path"))).resolve() != manifest_path.resolve():
        raise ReviewerTraceError("stage receipt does not bind the selected manifest")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ReviewerTraceError("selected manifest artifact inventory is malformed")
    manifest_entries = {
        str(row["path"]): row
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    required_paths = set(_RUN_METADATA) | {f"streams/{name}" for name in _RUN_STREAMS}
    if not required_paths.issubset(manifest_entries):
        raise ReviewerTraceError("selected manifest omits a required trace source")
    for relative in sorted(required_paths):
        path = run_dir / relative
        entry = manifest_entries[relative]
        if (
            _has_symlink_component(source, path)
            or not path.is_file()
            or _sha256_file(path) != entry.get("sha256")
            or path.stat().st_size != entry.get("byte_size")
        ):
            raise ReviewerTraceError(f"manifest-bound source drifted: {relative}")

    config_path = run_dir / "config.json"
    provenance_path = run_dir / "provenance.json"
    schemas_path = run_dir / "schemas.json"
    config = _load_json(config_path)
    provenance = _load_json(provenance_path)
    run_spec = provenance.get("details", {}).get("run_spec")
    if (
        config.get("run_id") != run_id
        or config.get("pilot_contract_hash") != contract.canonical_hash
        or config.get("seed") != spec.environment_seed
        or config.get("context_mode") != "full"
        or config.get("enable_episodic_retrieval") is not True
        or config.get("enable_semantic") is not True
        or config.get("semantic_policy") != "evidence-grounded"
        or config.get("retrieval_k") != 5
        or config.get("rule_budget") != 3
        or not isinstance(run_spec, Mapping)
        or run_spec.get("run_id") != run_id
        or run_spec.get("environment_seed") != spec.environment_seed
        or run_spec.get("stage_id") != "experiment-a"
        or run_spec.get("model_id") != "gpt52_main"
        or run_spec.get("arm_id") != "full"
        or provenance.get("git", {}).get("commit") != V2115_SOURCE_COMMIT
        or provenance.get("details", {}).get("scientific_evidence") is not True
    ):
        raise ReviewerTraceError("selected run config/provenance drifted")

    stream_paths = {name: run_dir / "streams" / name for name in _RUN_STREAMS}
    streams = {name: _load_jsonl(path) for name, path in stream_paths.items()}
    for name, records in streams.items():
        expected_lines = manifest_entries[f"streams/{name}"].get("line_count")
        if len(records) != expected_lines:
            raise ReviewerTraceError(f"stream line count drifted: {name}")

    a8 = _select_record(
        streams["actions.jsonl"],
        source_id="actions",
        selector={"agent_id": agent_id, "decision_t": decision_t},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("decision_t") == decision_t,
    )
    a9 = _select_record(
        streams["actions.jsonl"],
        source_id="actions",
        selector={"agent_id": agent_id, "decision_t": next_t},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("decision_t") == next_t,
    )
    c8 = _select_record(
        streams["context_trace.jsonl"],
        source_id="context_trace",
        selector={"agent_id": agent_id, "decision_t": decision_t},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("decision_t") == decision_t,
    )
    c9 = _select_record(
        streams["context_trace.jsonl"],
        source_id="context_trace",
        selector={"agent_id": agent_id, "decision_t": next_t},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("decision_t") == next_t,
    )
    d8 = _select_record(
        streams["decision_snapshots.jsonl"],
        source_id="decision_snapshots",
        selector={"agent_id": agent_id, "decision_t": decision_t},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("decision_t") == decision_t,
    )
    d9 = _select_record(
        streams["decision_snapshots.jsonl"],
        source_id="decision_snapshots",
        selector={"agent_id": agent_id, "decision_t": next_t},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("decision_t") == next_t,
    )
    e8 = _select_record(
        streams["episodes.jsonl"],
        source_id="episodes",
        selector={"agent_id": agent_id, "decision_t": decision_t},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("decision_t") == decision_t,
    )
    e9 = _select_record(
        streams["episodes.jsonl"],
        source_id="episodes",
        selector={"agent_id": agent_id, "decision_t": next_t},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("decision_t") == next_t,
    )
    u8 = _select_record(
        streams["utility_ledger.jsonl"],
        source_id="utility_ledger",
        selector={"agent_id": str(agent_id), "period": decision_t},
        predicate=lambda row: row.get("agent_id") == str(agent_id)
        and row.get("period") == decision_t,
    )
    m_before = _select_record(
        streams["macro_steps.jsonl"],
        source_id="macro_steps",
        selector={"outcome_t": decision_t},
        predicate=lambda row: row.get("outcome_t") == decision_t,
    )
    m_after = _select_record(
        streams["macro_steps.jsonl"],
        source_id="macro_steps",
        selector={"decision_t": decision_t, "outcome_t": decision_t + 1},
        predicate=lambda row: row.get("decision_t") == decision_t
        and row.get("outcome_t") == decision_t + 1,
    )
    shock = _select_record(
        streams["shock_events.jsonl"],
        source_id="shock_events",
        selector={"decision_t": decision_t},
        predicate=lambda row: row.get("decision_t") == decision_t,
    )
    api8 = _select_record(
        streams["api_usage.jsonl"],
        source_id="api_usage",
        selector={
            "call_kind": "action",
            "agent_id": agent_id,
            "decision_t": decision_t,
        },
        predicate=lambda row: row.get("call_kind") == "action"
        and row.get("agent_id") == agent_id
        and row.get("decision_t") == decision_t,
    )
    api9 = _select_record(
        streams["api_usage.jsonl"],
        source_id="api_usage",
        selector={
            "call_kind": "action",
            "agent_id": agent_id,
            "decision_t": next_t,
        },
        predicate=lambda row: row.get("call_kind") == "action"
        and row.get("agent_id") == agent_id
        and row.get("decision_t") == next_t,
    )

    selected_rule_ids = a8.value.get("selected_rule_ids")
    if not isinstance(selected_rule_ids, list) or len(selected_rule_ids) != 1:
        raise ReviewerTraceUnavailable(
            "the fixed coordinate does not contain exactly one selected rule; "
            "no fallback was attempted"
        )
    rule_id = str(selected_rule_ids[0])
    rule = _select_record(
        streams["semantic_rules.jsonl"],
        source_id="semantic_rules",
        selector={"agent_id": agent_id, "rule_id": rule_id},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("rule_id") == rule_id,
    )
    candidate_ids = rule.value.get("candidate_ids")
    if not isinstance(candidate_ids, list) or len(candidate_ids) != 1:
        raise ReviewerTraceError("selected rule candidate provenance is ambiguous")
    candidate_id = str(candidate_ids[0])
    proposal = _select_record(
        streams["semantic_proposals.jsonl"],
        source_id="semantic_proposals",
        selector={"agent_id": agent_id, "rule_id": rule_id},
        predicate=lambda row: row.get("agent_id") == agent_id
        and row.get("rule_id") == rule_id
        and row.get("current_t") == rule.value.get("created_at"),
    )
    proposal_api = _select_record(
        streams["api_usage.jsonl"],
        source_id="api_usage",
        selector={
            "call_kind": "semantic",
            "agent_id": agent_id,
            "decision_t": int(rule.value["created_at"]),
        },
        predicate=lambda row: row.get("call_kind") == "semantic"
        and row.get("agent_id") == agent_id
        and row.get("decision_t") == int(rule.value["created_at"]),
    )

    def event(event_type: str, timestamp: int) -> JsonlRecord:
        return _select_record(
            streams["semantic_rule_events.jsonl"],
            source_id="semantic_rule_events",
            selector={
                "agent_id": agent_id,
                "rule_id": rule_id,
                "timestamp": timestamp,
                "event_type": event_type,
            },
            predicate=lambda row: row.get("agent_id") == agent_id
            and row.get("rule_id") == rule_id
            and row.get("timestamp") == timestamp
            and row.get("event_type") == event_type,
        )

    candidate_verified = event("candidate_verified", int(rule.value["created_at"]))
    activation_episode = _select_record(
        streams["episodes.jsonl"],
        source_id="episodes",
        selector={"episode_id": rule.value["activation_episode_id"]},
        predicate=lambda row: row.get("episode_id")
        == rule.value["activation_episode_id"],
    )
    activation = event(
        "rule_activated", int(activation_episode.value["outcome_t"])
    )
    retrieved = event("active_rule_retrieved", decision_t)
    harmful = event("harmful_compliance_evidence_added", decision_t + 1)
    retired = event("rule_retired", decision_t + 1)

    v_a8 = a8.value
    v_a9 = a9.value
    v_c8 = c8.value
    v_c9 = c9.value
    v_d8 = d8.value
    v_d9 = d9.value
    v_e8 = e8.value
    v_e9 = e9.value
    v_u8 = u8.value
    v_mb = m_before.value
    v_ma = m_after.value
    v_shock = shock.value
    v_api8 = api8.value
    v_api9 = api9.value
    v_proposal_api = proposal_api.value
    episode_id = str(v_e8["episode_id"])
    candidate_identity = _recompute_candidate_identity(
        proposal=proposal.value,
        candidate_event=candidate_verified.value,
        outcome_criterion=rule.value["outcome_criterion"],
    )
    verifier_recomputation = _recompute_rule_classification(rule.value, v_e8)

    t8_episode_ids = [
        str(item) for item in v_c8.get("retrieved_episode_ids", [])
    ]
    t9_episode_ids = [
        str(item) for item in v_c9.get("retrieved_episode_ids", [])
    ]
    episode_records = streams["episodes.jsonl"]
    episode_ids = [str(record.value.get("episode_id")) for record in episode_records]
    episodes_by_id = {
        str(record.value.get("episode_id")): record.value for record in episode_records
    }

    def causal_same_agent_episode(item: str, *, upper_t: int) -> bool:
        episode = episodes_by_id.get(item)
        return (
            isinstance(episode, Mapping)
            and episode.get("agent_id") == agent_id
            and isinstance(episode.get("decision_t"), int)
            and isinstance(episode.get("outcome_t"), int)
            and int(episode["decision_t"]) < upper_t
            and int(episode["outcome_t"]) <= upper_t
        )

    checks = {
        "unique_rows": len(episode_ids) == len(set(episode_ids))
        and "None" not in episode_ids,
        "shock_bound": v_shock == v_d8.get("shock_event")
        and v_shock.get("interest_rate") == v_e8.get("pre_state", {}).get("interest_rate")
        and v_shock.get("interest_rate") == v_u8.get("interest_rate"),
        "context_bound": v_c8.get("context_hash") == v_d8.get("context_packet_hash")
        and v_c8.get("context_packet", {}).get("observed_through") == decision_t
        and v_c9.get("context_packet", {}).get("observed_through") == next_t
        and v_c8.get("protected_context_prompt_hash")
        == v_d8.get("protected_context_hash")
        and v_c9.get("protected_context_prompt_hash")
        == v_d9.get("protected_context_hash"),
        "prompt_bound": v_c8.get("memory_prompt_hash") == v_d8.get("memory_hash")
        and v_d8.get("full_prompt_hash") == v_a8.get("prompt_hash")
        and hashlib.sha256(
            str(v_d8.get("protected_context_text")).encode("utf-8")
        ).hexdigest()
        == v_d8.get("protected_context_hash")
        and canonical_sha256(v_e8.get("pre_state"))
        == v_d8.get("environment_state_hash"),
        "provider_bound": v_api8.get("prompt_hash") == v_a8.get("prompt_hash")
        and v_api8.get("raw_output_hash")
        == v_a8.get("decision", {}).get("raw_output_hash")
        and v_api9.get("prompt_hash") == v_a9.get("prompt_hash")
        and v_api9.get("raw_output_hash")
        == v_a9.get("decision", {}).get("raw_output_hash")
        and all(
            row.get("provider") == action.get("provider") == "openai"
            and row.get("model") == action.get("model")
            and snapshot.get("provider_model")
            == f"{row.get('provider')}/{row.get('model')}"
            and row.get("request_profile_id") == "gpt52_main"
            and row.get("request_artifact_identity", {}).get("served_snapshot")
            == action.get("model")
            and row.get("response_model") == action.get("model")
            and row.get("response_completed") is True
            and row.get("output_disposition") == "accepted"
            and row.get("action_parse_mode") == action.get("parse_mode")
            == "exact_json"
            and row.get("attempts") == 1
            for row, action, snapshot in (
                (v_api8, v_a8, v_d8),
                (v_api9, v_a9, v_d9),
            )
        ),
        "retrieval_bound": v_c8.get("retrieved_episode_ids")
        == v_a8.get("retrieved_episode_ids")
        == v_e8.get("retrieved_episode_ids")
        and len(t8_episode_ids) == config.get("retrieval_k")
        and len(t8_episode_ids) == len(set(t8_episode_ids))
        and all(causal_same_agent_episode(item, upper_t=decision_t) for item in t8_episode_ids),
        "rule_bound": v_c8.get("selected_rule_ids")
        == v_a8.get("selected_rule_ids")
        == v_e8.get("selected_rule_ids")
        and retrieved.value.get("rule_id") == rule_id
        and retrieved.value.get("from_status") == "active"
        and retrieved.value.get("to_status") == "active"
        and rule.value.get("candidate_ids") == [candidate_id]
        and candidate_verified.value.get("candidate_id") == candidate_id
        and candidate_identity["candidate_id"] == candidate_id
        and candidate_identity["raw_response_hash"]
        == candidate_verified.value.get("provenance", {}).get("raw_response_hash")
        and candidate_identity["supporting_episode_ids"]
        == candidate_verified.value.get("provenance", {}).get(
            "requested_support_ids"
        )
        and candidate_identity["provider_raw_output_sha256"]
        == proposal.value.get("raw_output_hash")
        == v_proposal_api.get("raw_output_hash")
        and proposal.value.get("prompt_hash") == v_proposal_api.get("prompt_hash")
        and v_proposal_api.get("response_completed") is True
        and v_proposal_api.get("output_disposition") == "accepted"
        and v_proposal_api.get("request_profile_id") == "gpt52_main"
        and v_proposal_api.get("model")
        == v_proposal_api.get("response_model")
        == v_proposal_api.get("request_artifact_identity", {}).get(
            "served_snapshot"
        )
        and v_proposal_api.get("provider") == "openai"
        and v_proposal_api.get("attempts") == 1
        and proposal.value.get("candidate_parse_status") == "success"
        and proposal.value.get("candidate_parse_mode") == "exact_json"
        and candidate_verified.value.get("from_status") is None
        and candidate_verified.value.get("to_status") == "provisional"
        and activation.value.get("episode_ids")
        == [rule.value.get("activation_episode_id")]
        and activation_episode.value.get("agent_id") == agent_id
        and activation_episode.value.get("outcome_t")
        == activation.value.get("timestamp")
        and activation.value.get("from_status") == "provisional"
        and activation.value.get("to_status") == "active",
        "action_bound": v_a8.get("decision", {}).get(
            "proposed_consumption_fraction"
        )
        == v_e8.get("proposed_action", {}).get("consumption_fraction")
        == v_u8.get("proposed_consumption_fraction")
        and v_a8.get("decision", {}).get("proposed_work_fraction")
        == v_e8.get("proposed_action", {}).get("work_propensity")
        == v_u8.get("proposed_work_propensity")
        and v_a8.get("decision", {}).get("executed_consumption_rate")
        == v_e8.get("executed_action", {}).get("consumption_fraction")
        == v_u8.get("executed_consumption_rate")
        and v_a8.get("decision", {}).get("executed_labor_hours")
        == v_e8.get("executed_action", {}).get("labor_hours")
        == v_u8.get("executed_labor_hours")
        and math.isclose(
            float(v_u8.get("proposed_labor_hours")),
            float(v_a8.get("decision", {}).get("proposed_work_fraction"))
            * float(config.get("max_labor_hours")),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "utility_bound": v_e8.get("flow_utility") == v_u8.get("flow_utility")
        and v_e8.get("outcome", {}).get("budget_residual")
        == v_u8.get("budget_residual")
        and v_e8.get("outcome", {}).get("supply_rationed")
        == v_u8.get("supply_rationed")
        and v_e8.get("next_state", {}).get("wealth") == v_u8.get("wealth_post"),
        "pre_macro_bound": v_e8.get("pre_state", {}).get("price") == v_mb.get("price")
        and v_e8.get("pre_state", {}).get("inflation")
        == v_mb.get("monthly_inflation")
        and v_e8.get("pre_state", {}).get("low_labor_rate")
        == v_mb.get("low_labor_rate"),
        "post_macro_bound": v_e8.get("next_state", {}).get("price")
        == v_ma.get("price")
        and v_e8.get("next_state", {}).get("inflation")
        == v_ma.get("monthly_inflation")
        and v_e8.get("next_state", {}).get("low_labor_rate")
        == v_ma.get("low_labor_rate")
        and v_e8.get("outcome_t") == v_ma.get("outcome_t"),
        "next_state_bound": v_e8.get("next_state") == v_e9.get("pre_state"),
        "verifier_ingested_outcome": harmful.value.get("episode_ids") == [episode_id]
        and harmful.value.get("rule_id") == rule_id
        and harmful.value.get("metrics", {}).get("consecutive_failures") == 2
        and harmful.value.get("provenance", {}).get(
            "registered_outcome_criterion"
        )
        == rule.value.get("outcome_criterion")
        and verifier_recomputation["classification"] == "harmful_compliance"
        and episode_id in rule.value.get("harmful_compliance_episode_ids", []),
        "retirement_bound": retired.value.get("episode_ids") == [episode_id]
        and retired.value.get("rule_id") == rule_id
        and retired.value.get("from_status") == "active"
        and retired.value.get("to_status") == "retired"
        and rule.value.get("status") == "retired"
        and rule.value.get("updated_at") == retired.value.get("timestamp"),
        "next_retrieval_bound": bool(t9_episode_ids)
        and t9_episode_ids[0] == episode_id
        and v_c9.get("retrieved_episode_ids")
        == v_a9.get("retrieved_episode_ids")
        == v_e9.get("retrieved_episode_ids")
        and len(t9_episode_ids) == config.get("retrieval_k")
        and len(t9_episode_ids) == len(set(t9_episode_ids))
        and all(causal_same_agent_episode(item, upper_t=next_t) for item in t9_episode_ids),
        "next_rule_absent": v_c9.get("selected_rule_ids")
        == v_a9.get("selected_rule_ids")
        == v_e9.get("selected_rule_ids")
        == [],
        "next_prompt_bound": v_c9.get("context_hash")
        == v_d9.get("context_packet_hash")
        and v_c9.get("memory_prompt_hash") == v_d9.get("memory_hash")
        and v_d9.get("full_prompt_hash") == v_a9.get("prompt_hash")
        and hashlib.sha256(
            str(v_d9.get("protected_context_text")).encode("utf-8")
        ).hexdigest()
        == v_d9.get("protected_context_hash")
        and canonical_sha256(v_e9.get("pre_state"))
        == v_d9.get("environment_state_hash"),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ReviewerTraceError(f"closed-loop bindings failed: {failed}")

    rule_payload = {
        "rule_id": rule_id,
        "candidate_id": candidate_id,
        "injected": rule.value["injected"],
        "condition": _json_copy(rule.value["condition"]),
        "context_scope": _json_copy(rule.value["context_scope"]),
        "action_guidance": _json_copy(rule.value["action_guidance"]),
        "outcome_criterion": _json_copy(rule.value["outcome_criterion"]),
        "proposal_t": proposal.value["current_t"],
        "proposal_parse_status": proposal.value["candidate_parse_status"],
        "proposal_parse_mode": proposal.value["candidate_parse_mode"],
        "proposal_raw_output_hash": proposal.value["raw_output_hash"],
        "proposal_prompt_hash": proposal.value["prompt_hash"],
        "proposal_frozen_source_provider_call": _provider_call(
            v_proposal_api,
            parse_mode=str(proposal.value["candidate_parse_mode"]),
        ),
        "candidate_identity_recomputation": candidate_identity,
        "candidate_verified_event_id": candidate_verified.value["event_id"],
        "candidate_verified_from_status": candidate_verified.value["from_status"],
        "candidate_verified_to_status": candidate_verified.value["to_status"],
        "searched_counterevidence": candidate_verified.value["provenance"][
            "searched_counterevidence"
        ],
        "activation_t": activation.value["timestamp"],
        "activation_event_id": activation.value["event_id"],
        "activation_episode_id": rule.value["activation_episode_id"],
        "activation_from_status": activation.value["from_status"],
        "activation_to_status": activation.value["to_status"],
        "status_at_retrieval": retrieved.value["to_status"],
        "confidence_at_retrieval": retrieved.value["metrics"]["confidence"],
        "margin_at_retrieval": retrieved.value["metrics"]["margin"],
        "final_status": rule.value["status"],
        "final_updated_at": rule.value["updated_at"],
        "retirement_t": retired.value["timestamp"],
        "retirement_from_status": retired.value["from_status"],
        "retirement_to_status": retired.value["to_status"],
    }
    trace = {
        "identity": {
            **_json_copy(coordinates),
            "requested_model": spec.requested_model,
            "shock_id": spec.shock_id,
            "num_agents": spec.num_agents,
            "episode_length": spec.episode_length,
        },
        "macro_before": _macro_state(v_mb),
        "shock": _json_copy(v_shock),
        "focal_state_before": _json_copy(v_e8["pre_state"]),
        "context": _context(v_c8),
        "retrieval": {
            "k": config["retrieval_k"],
            "items": _retrieval_items(v_c8),
            "selected_rules": [rule_payload],
        },
        "actor_visible": {
            "protected_context_text": v_d8["protected_context_text"],
            "protected_context_hash": v_d8["protected_context_hash"],
            "memory_text": v_d8["memory_text"],
            "context_packet_hash": v_d8["context_packet_hash"],
            "memory_hash": v_d8["memory_hash"],
            "environment_state_hash": v_d8["environment_state_hash"],
            "base_prompt_hash": v_d8["base_prompt_hash"],
            "full_prompt_hash": v_d8["full_prompt_hash"],
        },
        "frozen_source_provider_call": _provider_call(v_api8),
        "action": _action(v_a8),
        "outcome": {
            "episode_id": episode_id,
            "record_hash": v_e8["record_hash"],
            "outcome_t": v_e8["outcome_t"],
            "flow_utility": v_e8["flow_utility"],
            "discounted_flow_utility": v_u8["discounted_flow_utility"],
            "utility_advantage": v_e8["utility_advantage"],
            "wealth_pre": v_u8["wealth_pre"],
            "wealth_post": v_u8["wealth_post"],
            "wealth_change": v_e8["outcome"]["wealth_change"],
            "requested_consumption_spend": v_u8["requested_consumption_spend"],
            "proposed_consumption_fraction": v_u8[
                "proposed_consumption_fraction"
            ],
            "proposed_labor_hours": v_u8["proposed_labor_hours"],
            "proposed_work_propensity": v_u8["proposed_work_propensity"],
            "realized_consumption_spend": v_u8["realized_consumption_spend"],
            "realized_consumption_quantity": v_u8["realized_consumption_quantity"],
            "supply_rationed": v_u8["supply_rationed"],
            "budget_residual": v_u8["budget_residual"],
            "lump_sum_transfer": v_u8["lump_sum_transfer"],
            "tax_paid": v_u8["tax_paid"],
        },
        "focal_state_after": _json_copy(v_e8["next_state"]),
        "macro_after": _macro_state(v_ma),
        "memory_update": {
            "ingested_episode_id": episode_id,
            "verifier_recomputation": verifier_recomputation,
            "events": [_rule_event(harmful.value), _rule_event(retired.value)],
        },
        "next_decision": {
            "decision_t": next_t,
            "context": _context(v_c9),
            "state_continuity_verified": True,
            "retrieval_items": _retrieval_items(v_c9),
            "prior_episode_rank": t9_episode_ids.index(episode_id) + 1,
            "selected_rule_ids": _json_copy(v_c9["selected_rule_ids"]),
            "actor_visible": {
                "protected_context_text": v_d9["protected_context_text"],
                "protected_context_hash": v_d9["protected_context_hash"],
                "memory_text": v_d9["memory_text"],
                "context_packet_hash": v_d9["context_packet_hash"],
                "memory_hash": v_d9["memory_hash"],
                "environment_state_hash": v_d9["environment_state_hash"],
                "base_prompt_hash": v_d9["base_prompt_hash"],
                "full_prompt_hash": v_d9["full_prompt_hash"],
            },
            "frozen_source_provider_call": _provider_call(v_api9),
            "action": _action(v_a9),
        },
    }

    source_files = [
        _source_file(
            source,
            source_id="contract",
            path=source / V2115_CONTRACT_RELATIVE,
            binding_kind="frozen-hardcoded-file-hash",
        ),
        _source_file(
            source,
            source_id="experiment_a_stage_receipt",
            path=receipt_path,
            binding_kind="frozen-hardcoded-file-and-content-hash",
        ),
        _source_file(
            source,
            source_id="selected_run_manifest",
            path=manifest_path,
            binding_kind="stage-receipt-and-hardcoded-file-hash",
        ),
    ]
    for name in _RUN_METADATA:
        source_files.append(
            _source_file(
                source,
                source_id=name.removesuffix(".json"),
                path=run_dir / name,
                binding_kind="selected-run-manifest",
                line_count=int(manifest_entries[name]["line_count"]),
            )
        )
    for name in _RUN_STREAMS:
        source_files.append(
            _source_file(
                source,
                source_id=name.removesuffix(".jsonl"),
                path=stream_paths[name],
                binding_kind="selected-run-manifest",
                line_count=len(streams[name]),
            )
        )
    for binding in source_files:
        path = source / str(binding["relative_path"])
        if _sha256_file(path) != binding["file_sha256"]:
            raise ReviewerTraceError(
                f"source changed during extraction: {binding['source_id']}"
            )
    central_loader_after = _central_loader_receipt(
        run_dir=run_dir,
        authority_repo_root=source,
        publisher_repo_root=publisher_root,
    )
    if central_loader_after != central_loader_before:
        raise ReviewerTraceError("central verified-run receipt changed during extraction")
    for name, records in streams.items():
        central_name = name.removesuffix(".jsonl")
        if central_loader_after["record_counts"].get(central_name) != len(records):
            raise ReviewerTraceError(
                f"central verified-run stream count drifted: {central_name}"
            )

    source_records = [
        _source_record_binding(
            "semantic_proposals",
            proposal,
            selector={"agent_id": agent_id, "rule_id": rule_id},
            copied_to=["/trace/retrieval/selected_rules/0"],
            native_record_id=rule_id,
        ),
        _source_record_binding(
            "api_usage",
            proposal_api,
            selector={
                "call_kind": "semantic",
                "agent_id": agent_id,
                "decision_t": int(rule.value["created_at"]),
            },
            copied_to=[
                "/trace/retrieval/selected_rules/0/proposal_frozen_source_provider_call"
            ],
        ),
        _source_record_binding(
            "semantic_rule_events",
            candidate_verified,
            selector={"event_id": candidate_verified.value["event_id"]},
            copied_to=["/trace/retrieval/selected_rules/0"],
            native_record_id=candidate_verified.value["event_id"],
        ),
        _source_record_binding(
            "semantic_rule_events",
            activation,
            selector={"event_id": activation.value["event_id"]},
            copied_to=["/trace/retrieval/selected_rules/0"],
            native_record_id=activation.value["event_id"],
        ),
        _source_record_binding(
            "episodes",
            activation_episode,
            selector={"episode_id": rule.value["activation_episode_id"]},
            copied_to=[
                "/trace/retrieval/selected_rules/0/activation_episode_id"
            ],
            native_record_id=str(activation_episode.value["episode_id"]),
            native_record_hash=str(activation_episode.value["record_hash"]),
        ),
        _source_record_binding(
            "semantic_rule_events",
            retrieved,
            selector={"event_id": retrieved.value["event_id"]},
            copied_to=["/trace/retrieval/selected_rules/0"],
            native_record_id=retrieved.value["event_id"],
        ),
        _source_record_binding(
            "semantic_rule_events",
            harmful,
            selector={"event_id": harmful.value["event_id"]},
            copied_to=["/trace/memory_update/events/0"],
            native_record_id=harmful.value["event_id"],
        ),
        _source_record_binding(
            "semantic_rule_events",
            retired,
            selector={"event_id": retired.value["event_id"]},
            copied_to=["/trace/memory_update/events/1"],
            native_record_id=retired.value["event_id"],
        ),
        _source_record_binding(
            "semantic_rules",
            rule,
            selector={"agent_id": agent_id, "rule_id": rule_id},
            copied_to=["/trace/retrieval/selected_rules/0"],
            native_record_id=rule_id,
        ),
        _source_record_binding(
            "macro_steps",
            m_before,
            selector={"outcome_t": decision_t},
            copied_to=["/trace/macro_before"],
        ),
        _source_record_binding(
            "macro_steps",
            m_after,
            selector={"decision_t": decision_t, "outcome_t": decision_t + 1},
            copied_to=["/trace/macro_after"],
        ),
        _source_record_binding(
            "shock_events",
            shock,
            selector={"decision_t": decision_t},
            copied_to=["/trace/shock"],
        ),
    ]
    for source_id, current, following, current_paths, next_paths in (
        (
            "context_trace",
            c8,
            c9,
            ["/trace/context", "/trace/retrieval"],
            ["/trace/next_decision/context", "/trace/next_decision/retrieval_items"],
        ),
        (
            "decision_snapshots",
            d8,
            d9,
            ["/trace/actor_visible"],
            ["/trace/next_decision/actor_visible"],
        ),
        (
            "actions",
            a8,
            a9,
            ["/trace/action"],
            ["/trace/next_decision/action"],
        ),
        (
            "api_usage",
            api8,
            api9,
            ["/trace/frozen_source_provider_call"],
            ["/trace/next_decision/frozen_source_provider_call"],
        ),
        (
            "episodes",
            e8,
            e9,
            [
                "/trace/focal_state_before",
                "/trace/outcome",
                "/trace/focal_state_after",
            ],
            ["/trace/next_decision"],
        ),
    ):
        source_records.extend(
            [
                _source_record_binding(
                    source_id,
                    current,
                    selector={"agent_id": agent_id, "decision_t": decision_t},
                    copied_to=current_paths,
                    native_record_id=(
                        str(current.value.get("episode_id"))
                        if source_id == "episodes"
                        else None
                    ),
                    native_record_hash=(
                        str(current.value.get("record_hash"))
                        if source_id == "episodes"
                        else None
                    ),
                ),
                _source_record_binding(
                    source_id,
                    following,
                    selector={"agent_id": agent_id, "decision_t": next_t},
                    copied_to=next_paths,
                    native_record_id=(
                        str(following.value.get("episode_id"))
                        if source_id == "episodes"
                        else None
                    ),
                    native_record_hash=(
                        str(following.value.get("record_hash"))
                        if source_id == "episodes"
                        else None
                    ),
                ),
            ]
        )
    source_records.append(
        _source_record_binding(
            "utility_ledger",
            u8,
            selector={"agent_id": str(agent_id), "period": decision_t},
            copied_to=["/trace/outcome"],
        )
    )

    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "artifact_id": (
            f"{V2115_CONTRACT_ID}--reviewer-closed-loop--experiment-a--"
            f"gpt52_main--full--s{spec.environment_seed}--a{agent_id}--t{decision_t}"
        ),
        "status": "complete",
        "publication_provider_calls": 0,
        "selection_policy": _json_copy(selection),
        "evidence_scope": {
            "diagnostic_only": True,
            "descriptive_only": True,
            "effectiveness_evidence": False,
            "frozen_source_provider_call_scope": (
                "historical-observations-read-from-sealed-logs"
            ),
            "publication_provider_calls": 0,
            "stage_authoritative": False,
        },
        "provenance": {
            "contract_id": contract.contract_id,
            "contract_file_sha256": V2115_CONTRACT_FILE_SHA256,
            "contract_canonical_sha256": contract.canonical_hash,
            "science_source": _json_copy(source_provenance),
            "publisher": _json_copy(publisher_provenance),
            "run_id": run_id,
            "stage_status": receipt["status"],
            "stage_go": receipt["go"],
            "stage_scientific_matrix_complete": receipt[
                "scientific_matrix_complete"
            ],
            "stage_denominator_terminal": receipt["denominator_terminal"],
            "stage_registered_runs": receipt["registered_run_count"],
            "stage_complete_runs": receipt["complete_cell_count"],
            "stage_failed_runs": receipt["status_counts"]["failed"],
            "selected_run_complete": result["complete"],
            "selected_run_validation_status": validation["status"],
            "selected_run_scientific_evidence": validation["scientific_evidence"],
            "selected_run_manifest_sha256": V2115_SELECTED_RUN_MANIFEST_SHA256,
            "central_loader_validation": _json_copy(central_loader_after),
        },
        "source_files": source_files,
        "source_records": source_records,
        "trace": trace,
        "failure_reason": None,
        "link_checks": {"all_pass": True, "checks": checks},
        "claim_boundary": _claim_boundary(),
    }
    return _seal_artifact(payload)


def _build_trace_artifact_verified(
    *,
    source: Path,
    publisher_root: Path,
    source_provenance: Mapping[str, Any],
    publisher_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    contract_path = source / V2115_CONTRACT_RELATIVE
    if _sha256_file(contract_path) != V2115_CONTRACT_FILE_SHA256:
        raise ReviewerTraceError("V2.11.5 contract file hash drifted")
    contract = load_pilot_contract(contract_path)
    spec, selection = derive_selection(contract)
    raw_root = source / V2115_RAW_RELATIVE
    try:
        payload = _extract_trace(
            source=source,
            publisher_root=publisher_root,
            raw_root=raw_root,
            spec=spec,
            selection=selection,
            contract=contract,
            source_provenance=source_provenance,
            publisher_provenance=publisher_provenance,
        )
    except ReviewerTraceUnavailable as exc:
        unavailable = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "artifact_id": (
                f"{V2115_CONTRACT_ID}--reviewer-closed-loop--unavailable"
            ),
            "status": "unavailable",
            "publication_provider_calls": 0,
            "selection_policy": selection,
            "evidence_scope": {
                "diagnostic_only": True,
                "descriptive_only": True,
                "effectiveness_evidence": False,
                "frozen_source_provider_call_scope": (
                    "historical-observations-read-from-sealed-logs"
                ),
                "publication_provider_calls": 0,
                "stage_authoritative": False,
            },
            "provenance": {
                "contract_id": contract.contract_id,
                "contract_file_sha256": V2115_CONTRACT_FILE_SHA256,
                "contract_canonical_sha256": contract.canonical_hash,
                "science_source": _json_copy(source_provenance),
                "publisher": _json_copy(publisher_provenance),
            },
            "source_files": [],
            "source_records": [],
            "trace": None,
            "failure_reason": str(exc),
            "link_checks": {"all_pass": False, "checks": {}},
            "claim_boundary": _claim_boundary(),
        }
        payload = _seal_artifact(unavailable)
    schema_path = publisher_root / TRACE_SCHEMA_RELATIVE
    if schema_path.is_symlink() or not schema_path.is_file():
        raise ReviewerTraceError("reviewer trace schema is missing or unsafe")
    _validate_against_runtime_schema(payload, schema_path.read_bytes())
    return payload


def build_trace_artifact(
    *,
    source_repo_root: str | Path,
    publisher_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the in-memory trace after verifying both immutable checkouts."""

    source_input = _absolute_without_symlink(
        source_repo_root, name="science source repository"
    )
    code_input = (
        _absolute_without_symlink(
            publisher_repo_root, name="publisher repository"
        )
        if publisher_repo_root is not None
        else _absolute_without_symlink(
            Path(__file__).resolve().parents[1], name="publisher repository"
        )
    )
    source_provenance = _verify_source_checkout(source_input)
    publisher_provenance = _verify_publisher_checkout(code_input)
    source = source_input.resolve(strict=True)
    code_root = code_input.resolve(strict=True)
    return _build_trace_artifact_verified(
        source=source,
        publisher_root=code_root,
        source_provenance=source_provenance,
        publisher_provenance=publisher_provenance,
    )


_PUBLICATION_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{12,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bxox[a-z]-[A-Za-z0-9-]{12,}"),
    re.compile(
        r"(?i)\b(?:OPENAI|OPENROUTER|GEMINI|GOOGLE|ANTHROPIC)[_-]?API[_-]?KEY\s*[:=]"
    ),
)
_PUBLICATION_HOST_PATH_PATTERNS = (
    re.compile(r"/(?:Users|home|tmp|var|root|opt)/"),
    re.compile(r"file://", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]:[\\/]+"),
)


def _scan_publication_bytes(data: bytes, *, name: str) -> None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewerTraceError(f"publication file is not UTF-8: {name}") from exc
    if any(pattern.search(text) for pattern in _PUBLICATION_SECRET_PATTERNS):
        raise ReviewerTraceError(f"publication secret scan failed: {name}")
    if any(pattern.search(text) for pattern in _PUBLICATION_HOST_PATH_PATTERNS):
        raise ReviewerTraceError(f"publication host-path scan failed: {name}")


def _write_trace_package(
    *,
    payload: Mapping[str, Any],
    schema_source: Path,
    target: Path,
) -> Path:
    validate_trace_artifact(payload)
    if schema_source.is_symlink() or not schema_source.is_file():
        raise ReviewerTraceError("reviewer trace schema is missing or unsafe")
    target = _absolute_without_symlink(target, name="reviewer trace output")
    schema_bytes = schema_source.read_bytes()
    expected_schema_blob = (
        payload.get("provenance", {})
        .get("publisher", {})
        .get("required_tracked_head_blobs", {})
        .get(TRACE_SCHEMA_RELATIVE.as_posix())
    )
    if _git_blob_oid(schema_bytes) != expected_schema_blob:
        raise ReviewerTraceError("reviewer trace schema does not match publisher HEAD")
    _validate_against_runtime_schema(payload, schema_bytes)
    trace_bytes = _pretty_bytes(payload)
    checksums = {
        "schema_version": "finevo-reviewer-closed-loop-trace-checksums-v1",
        "publication_provider_calls": 0,
        "files": [
            {
                "path": TRACE_FILENAME,
                "sha256": _sha256_bytes(trace_bytes),
                "byte_size": len(trace_bytes),
            },
            {
                "path": TRACE_SCHEMA_RELATIVE.name,
                "sha256": _sha256_bytes(schema_bytes),
                "byte_size": len(schema_bytes),
            },
        ],
    }
    checksums_bytes = _pretty_bytes(checksums)
    package_bytes = {
        TRACE_FILENAME: trace_bytes,
        TRACE_SCHEMA_RELATIVE.name: schema_bytes,
        TRACE_CHECKSUMS_FILENAME: checksums_bytes,
    }
    for name, data in package_bytes.items():
        _scan_publication_bytes(data, name=name)
    target.parent.mkdir(parents=True, exist_ok=True)
    reserved = False
    try:
        target.mkdir(mode=0o700)
        reserved = True
    except FileExistsError as exc:
        raise ReviewerTraceError(f"refusing to overwrite reviewer trace: {target}") from exc
    try:
        for name, data in package_bytes.items():
            with (target / name).open("xb") as handle:
                handle.write(data)
        actual_paths = tuple(target.iterdir())
        if {path.name for path in actual_paths} != set(package_bytes) or any(
            path.is_symlink() or not path.is_file() for path in actual_paths
        ):
            raise ReviewerTraceError("publication package file set drifted")
        for name, expected in package_bytes.items():
            written = (target / name).read_bytes()
            if written != expected:
                raise ReviewerTraceError(f"publication write verification failed: {name}")
            _scan_publication_bytes(written, name=name)
        written_payload = _loads_strict(
            (target / TRACE_FILENAME).read_bytes(), name=TRACE_FILENAME
        )
        if not isinstance(written_payload, Mapping):
            raise ReviewerTraceError("written reviewer trace is not an object")
        validate_trace_artifact(written_payload)
        _validate_against_runtime_schema(written_payload, schema_bytes)
        if _git_blob_oid((target / TRACE_SCHEMA_RELATIVE.name).read_bytes()) != (
            expected_schema_blob
        ):
            raise ReviewerTraceError("copied reviewer trace schema blob drifted")
        final_paths = tuple(target.iterdir())
        if {path.name for path in final_paths} != set(package_bytes) or any(
            path.is_symlink() or not path.is_file() for path in final_paths
        ):
            raise ReviewerTraceError("publication package file set changed")
        for path in final_paths:
            path.chmod(0o444)
        target.chmod(0o755)
    except Exception:
        if reserved:
            shutil.rmtree(target, ignore_errors=True)
        raise
    return target / TRACE_FILENAME


def build_trace_package(
    *,
    source_repo_root: str | Path,
    output_dir: str | Path,
    publisher_repo_root: str | Path | None = None,
) -> Path:
    """Write a no-overwrite deterministic package and return the trace path."""

    code_input = (
        _absolute_without_symlink(
            publisher_repo_root, name="publisher repository"
        )
        if publisher_repo_root is not None
        else _absolute_without_symlink(
            Path(__file__).resolve().parents[1], name="publisher repository"
        )
    )
    target = _absolute_without_symlink(output_dir, name="reviewer trace output")
    if target.exists() or target.is_symlink():
        raise ReviewerTraceError(f"refusing to overwrite reviewer trace: {target}")
    publisher = _verify_publisher_checkout(code_input)
    source_input = _absolute_without_symlink(
        source_repo_root, name="science source repository"
    )
    source_provenance = _verify_source_checkout(source_input)
    code_root = code_input.resolve(strict=True)
    source = source_input.resolve(strict=True)
    payload = _build_trace_artifact_verified(
        source=source,
        publisher_root=code_root,
        source_provenance=source_provenance,
        publisher_provenance=publisher,
    )
    schema_source = code_root / TRACE_SCHEMA_RELATIVE
    return _write_trace_package(
        payload=payload,
        schema_source=schema_source,
        target=target,
    )


__all__ = [
    "ReviewerTraceError",
    "ReviewerTraceUnavailable",
    "TRACE_FILENAME",
    "TRACE_SCHEMA_RELATIVE",
    "TRACE_SCHEMA_VERSION",
    "build_trace_artifact",
    "build_trace_package",
    "derive_selection",
    "validate_trace_artifact",
]
