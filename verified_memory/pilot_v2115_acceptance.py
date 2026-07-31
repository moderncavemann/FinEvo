"""Zero-provider scientific-dispatch acceptance for FinEvo V2.11.5.

This gate is intentionally separate from stage execution.  It is run once on
the clean annotated science tag, after the three operational import stages and
before any hosted-provider credential is loaded.  The immutable receipt binds
the complete prospective scientific denominator, every runner configuration,
the five Experiment-D checkpoint groups, and the full projected budget.

Later scientific stages verify the receipt and its two ledger event prefixes
before provider/catalog construction.  Legitimate scientific execution may
append ledger events, but it cannot rewrite the accepted prefix.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Mapping, Sequence
from unittest import mock

import llm_providers as canonical_llm_providers

from .pilot_budget import PilotBudgetLedger, RunProjection
from .pilot_checkpoint import config_from_dict
from .pilot_contract import PilotContract, canonical_sha256, load_pilot_contract
from .pilot_v2115_gate import (
    V2115_ALLOWED_MODELS,
    V2115_POST_GATE_RELATIVE_PATH,
    v2115_observed_p95_projection_path,
    v2115_observed_p95_receipt_path,
    verified_v2115_gate_authority_binding,
    verified_v2115_observed_p95_authority_binding,
)
from .pilot_v2115_parent_import import (
    V2115_CONTRACT_ID,
    V2115_CONTRACT_PATH,
    V2115_DEFAULT_RECEIPT_PATH,
    V2115_RAW_ROOT,
    verify_v2115_parent_import_receipt,
)
from . import pilot_orchestrator as orch
from . import pilot_provider_catalog as canonical_provider_catalog
from .observed_p95_authority import verified_observed_p95_authority_binding
from .runner import (
    has_sealed_observed_p95_authority,
    observed_p95_authority_repo_context,
    serialized_has_sealed_observed_p95_authority,
    validate_preflight_p95_reservations,
)


V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.5-scientific-dispatch-acceptance-v1"
)
V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_FILENAME = "scientific_dispatch_acceptance.json"
V2115_SCIENTIFIC_STAGE_IDS = (
    "experiment-c",
    "experiment-a",
    "experiment-d",
    "experiment-b",
    "cross-model",
)
V2115_OPERATIONAL_STAGE_IDS = (
    "parent-import",
    "capability-gate",
    "long-context-preflight",
)
V2115_EXPECTED_STAGE_CELL_COUNTS = {
    "parent-import": 1,
    "capability-gate": 2,
    "long-context-preflight": 2,
    "experiment-c": 25,
    "experiment-a": 20,
    "experiment-d": 55,
    "experiment-b": 25,
    "cross-model": 6,
}
V2115_EXPECTED_LEDGER_CELLS = 136
V2115_EXPECTED_OPERATIONAL_CELLS = 5
V2115_EXPECTED_SCIENTIFIC_CELLS = 131
V2115_EXPECTED_PROVIDER_CELLS = 126
V2115_EXPECTED_OFFLINE_CELLS = 5
V2115_EXPECTED_D_GROUPS = 5
V2115_EXPECTED_D_CELLS_PER_GROUP = 11
V2115_EXPECTED_PROJECTION_UNITS = 81
V2115_EXPECTED_ACTION_CALLS = 4_848
V2115_EXPECTED_SEMANTIC_CALLS = 968
V2115_EXPECTED_PROVIDER_CALLS = 5_816
V2115_EXPECTED_FRESH_STORAGE_BYTES = 1_830_000_000
V2115_EXPECTED_ACCEPTED_RUN_EVENTS = 7
V2115_EXPECTED_ACCEPTED_BUDGET_EVENTS = 12
V2115_ACCEPTANCE_LEDGER_EVENT_TYPE = "acceptance_receipt_bound"

_CLAIM_BOUNDARY = (
    "Pre-dispatch integrity and budget acceptance only. This receipt "
    "contains no treatment outcomes and supports no effectiveness claim."
)
_ACCEPTANCE_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "go",
        "contract_id",
        "contract_sha256",
        "release",
        "raw_namespace",
        "pre_science_namespace",
        "denominator",
        "operational_gates",
        "authorities",
        "runner_configs",
        "experiment_d",
        "budget_projection",
        "ledger_prefixes",
        "bound_source_file_sha256",
        "provider_boundary",
        "scientific_evidence",
        "claim_boundary",
        "integrity",
    }
)
_LEDGER_PREFIX_FIELDS = frozenset(
    {"file_sha256", "ledger_sha256", "event_count", "event_chain_head"}
)
_PRE_SCIENCE_NAMESPACE = {
    "scientific_stage_paths_present": 0,
    "legacy_scientific_run_paths_present": 0,
    "decoded_scientific_completion_reuse": False,
}

_PROVIDER_KEY_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)
_CANONICALIZATION = "json-sort-keys-utf8-v1"


class PilotV2115AcceptanceError(RuntimeError):
    """Raised before V2.11.5 can acquire scientific dispatch authority."""


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


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PilotV2115AcceptanceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_regular(path: Path, *, name: str) -> bytes:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            component = current.lstat()
        except FileNotFoundError as exc:
            raise PilotV2115AcceptanceError(
                f"{name} path component is missing: {current}"
            ) from exc
        if stat.S_ISLNK(component.st_mode):
            raise PilotV2115AcceptanceError(
                f"{name} path contains a symlink component: {current}"
            )
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise PilotV2115AcceptanceError(f"{name} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PilotV2115AcceptanceError(
            f"{name} must be a non-symlink regular file: {path}"
        )
    try:
        return absolute.read_bytes()
    except OSError as exc:
        raise PilotV2115AcceptanceError(f"{name} cannot be read: {path}") from exc


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    raw = _read_regular(path, name=name)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PilotV2115AcceptanceError(
                    f"{name} contains non-finite JSON token {token!r}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotV2115AcceptanceError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PilotV2115AcceptanceError(f"{name} root must be an object")
    return value


def _file_sha256(path: Path, *, name: str) -> str:
    return hashlib.sha256(_read_regular(path, name=name)).hexdigest()


def _receipt_content_sha256(value: Mapping[str, Any]) -> str:
    copied = _json_copy(dict(value))
    integrity = copied.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("content_sha256", None)
    return canonical_sha256(copied)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(dict(value))
    result["integrity"] = {"canonicalization": _CANONICALIZATION}
    result["integrity"]["content_sha256"] = _receipt_content_sha256(result)
    return result


def _verify_seal(value: Mapping[str, Any]) -> None:
    integrity = value.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != _CANONICALIZATION
        or integrity.get("content_sha256") != _receipt_content_sha256(value)
    ):
        raise PilotV2115AcceptanceError(
            "scientific-dispatch acceptance receipt self-hash mismatch"
        )


def _relative(repo_root: Path, path: Path, *, name: str) -> str:
    try:
        relative = path.absolute().relative_to(repo_root)
    except ValueError as exc:
        raise PilotV2115AcceptanceError(
            f"{name} escaped the release repository"
        ) from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise PilotV2115AcceptanceError(f"{name} path is not normalized")
    return PurePosixPath(*relative.parts).as_posix()


def _exact_roots(
    repo_root: str | Path,
    raw_root: str | Path,
) -> tuple[Path, Path]:
    repository = Path(repo_root).resolve(strict=True)
    raw = Path(raw_root)
    if not raw.is_absolute():
        raw = repository.joinpath(*PurePosixPath(str(raw)).parts)
    raw = raw.absolute()
    expected = repository.joinpath(*V2115_RAW_ROOT.parts)
    if raw != expected:
        raise PilotV2115AcceptanceError(
            "V2.11.5 acceptance requires its exact ignored raw namespace"
        )
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise PilotV2115AcceptanceError(
                f"V2.11.5 raw namespace component is missing: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PilotV2115AcceptanceError(
                f"V2.11.5 raw namespace contains a symlink: {current}"
            )
    if not stat.S_ISDIR(raw.lstat().st_mode):
        raise PilotV2115AcceptanceError("V2.11.5 raw namespace must be a directory")
    return repository, raw


def _require_contract(contract: PilotContract, repo_root: Path) -> None:
    contract_path = repo_root.joinpath(*V2115_CONTRACT_PATH.parts)
    persisted = load_pilot_contract(contract_path)
    if (
        contract.contract_id != V2115_CONTRACT_ID
        or contract.schema_version != "finevo-pilot-contract-v2"
        or contract.status != "frozen"
        or persisted.to_dict() != contract.to_dict()
        or tuple(contract.stage_ids)
        != (*V2115_OPERATIONAL_STAGE_IDS, *V2115_SCIENTIFIC_STAGE_IDS)
        or {
            stage_id: len(contract.expand(stage=stage_id))
            for stage_id in contract.stage_ids
        }
        != V2115_EXPECTED_STAGE_CELL_COUNTS
    ):
        raise PilotV2115AcceptanceError(
            "acceptance requires the exact frozen V2.11.5 contract denominator"
        )
    amendment = contract.v2115_consumer_authority_normalization_amendment
    forward = contract.v2115_forward_boundary
    if not isinstance(amendment, Mapping) or not isinstance(forward, Mapping):
        raise PilotV2115AcceptanceError("V2.11.5 acceptance amendment is missing")
    matrix = forward.get("matrix")
    fresh = amendment.get("fresh_science_dispatch")
    if (
        not isinstance(matrix, Mapping)
        or not isinstance(fresh, Mapping)
        or matrix.get("ledger_cells") != V2115_EXPECTED_LEDGER_CELLS
        or matrix.get("operational_cells") != V2115_EXPECTED_OPERATIONAL_CELLS
        or matrix.get("scientific_cells") != V2115_EXPECTED_SCIENTIFIC_CELLS
        or matrix.get("provider_backed_scientific_cells")
        != V2115_EXPECTED_PROVIDER_CELLS
        or matrix.get("offline_scientific_cells") != V2115_EXPECTED_OFFLINE_CELLS
        or matrix.get("fresh_action_calls") != V2115_EXPECTED_ACTION_CALLS
        or matrix.get("fresh_semantic_calls") != V2115_EXPECTED_SEMANTIC_CALLS
        or matrix.get("fresh_scientific_provider_calls")
        != V2115_EXPECTED_PROVIDER_CALLS
        or fresh.get("registered_scientific_cells") != V2115_EXPECTED_SCIENTIFIC_CELLS
        or fresh.get("registered_provider_calls") != V2115_EXPECTED_PROVIDER_CALLS
        or fresh.get("matrix_shrink") != "forbidden"
        or fresh.get("failed_seed_replacement") != "forbidden"
    ):
        raise PilotV2115AcceptanceError(
            "V2.11.5 frozen matrix declaration differs from acceptance constants"
        )


def _require_provider_keys_absent() -> list[str]:
    present = [name for name in _PROVIDER_KEY_ENV_NAMES if os.environ.get(name)]
    if present:
        raise PilotV2115AcceptanceError(
            "acceptance must run before provider credentials are loaded; present="
            + ",".join(present)
        )
    return list(_PROVIDER_KEY_ENV_NAMES)


def _expected_provider_boundary() -> dict[str, Any]:
    return {
        "credential_environment_names_checked": list(_PROVIDER_KEY_ENV_NAMES),
        "credential_values_present": False,
        "provider_construction_calls": 0,
        "provider_catalog_calls": 0,
        "provider_completion_calls": 0,
        "zero_provider_acceptance": True,
    }


def _provider_boundary_stack() -> ExitStack:
    """Turn any accidental provider/catalog construction into a hard failure."""

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise PilotV2115AcceptanceError(
            "zero-provider acceptance attempted provider/catalog construction"
        )

    stack = ExitStack()
    for name in (
        "_provider_for_profile",
        "create_llm_provider",
        "validate_live_provider_catalog",
        "MultiModelLLM",
    ):
        if hasattr(orch, name):
            stack.enter_context(mock.patch.object(orch, name, side_effect=forbidden))
    for module, names in (
        (canonical_llm_providers, ("create_llm_provider", "MultiModelLLM")),
        (canonical_provider_catalog, ("validate_live_provider_catalog",)),
    ):
        for name in names:
            if hasattr(module, name):
                stack.enter_context(
                    mock.patch.object(module, name, side_effect=forbidden)
                )
    return stack


def _expected_pre_science_file_allowlist(contract: PilotContract) -> set[str]:
    allowed = {
        ".real-stage-execution.lock",
        "budget_ledger.json",
        "release_attestation.json",
        "run_ledger.json",
        "scientific_dispatch_acceptance.json",
        "scientific_launch_input.json",
        "parent-import/parent_import_receipt.json",
        "parent-import/stage_receipt.json",
        "capability-gate/stage_receipt.json",
        "long-context-preflight/post_gate_authority.json",
        "long-context-preflight/stage_receipt.json",
    }
    for spec in contract.expand():
        if spec.stage_id not in V2115_OPERATIONAL_STAGE_IDS:
            continue
        allowed.add(f"{spec.stage_id}/summaries/{spec.run_id}.json")
        if spec.stage_id == "capability-gate":
            allowed.update(
                {
                    f"capability-gate/runs/{spec.run_id}/capability.json",
                    f"capability-gate/runs/{spec.run_id}/gate_receipt.json",
                }
            )
        elif spec.stage_id == "long-context-preflight":
            allowed.update(
                {
                    (
                        "long-context-preflight/imported_observed_p95/"
                        f"{spec.model_id}/observed_p95_authority_receipt.json"
                    ),
                    (
                        "long-context-preflight/imported_observed_p95/"
                        f"{spec.model_id}/projection_p95.json"
                    ),
                    (
                        "long-context-preflight/runs/"
                        f"{spec.run_id}/preflight_authority.json"
                    ),
                    (
                        "long-context-preflight/runs/"
                        f"{spec.run_id}/gate_receipt.json"
                    ),
                }
            )
    return allowed


def _audit_pre_science_namespace(
    raw_root: Path, contract: PilotContract
) -> dict[str, Any]:
    """Require an empty scientific namespace before the acceptance marker.

    Operational imports may carry aggregate capability and preflight
    authorities, but they may not copy a prior-release scientific run, provider
    catalog, decoded completion, or current-release scientific artifact into
    the V2.11.5 raw tree.  All legitimate science output is created only after
    this zero-call acceptance has been bound into both ledgers.
    """

    stage_roots = set(V2115_SCIENTIFIC_STAGE_IDS)
    legacy_tokens = (
        "finevo-pilot-v2.11.2--experiment-",
        "finevo-pilot-v2.11.2--cross-model--",
        "finevo-pilot-v2.11.3--experiment-",
        "finevo-pilot-v2.11.3--cross-model--",
        "finevo-pilot-v2.11.4--experiment-",
        "finevo-pilot-v2.11.4--cross-model--",
    )
    allowed_files = _expected_pre_science_file_allowlist(contract)
    scientific_paths: list[str] = []
    legacy_paths: list[str] = []
    unexpected_files: list[str] = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_symlink():
            raise PilotV2115AcceptanceError(
                "pre-science raw namespace contains a symlink: "
                + path.relative_to(raw_root).as_posix()
            )
        relative = path.relative_to(raw_root).as_posix()
        parts = PurePosixPath(relative).parts
        if parts and parts[0] in stage_roots:
            scientific_paths.append(relative)
        if any(token in relative for token in legacy_tokens):
            legacy_paths.append(relative)
        if path.is_file() and relative not in allowed_files:
            unexpected_files.append(relative)
    if scientific_paths or legacy_paths or unexpected_files:
        raise PilotV2115AcceptanceError(
            "pre-science raw namespace contains scientific artifacts: "
            f"stage_paths={scientific_paths[:5]}, legacy_paths={legacy_paths[:5]}, "
            f"unexpected_files={unexpected_files[:5]}"
        )
    return dict(_PRE_SCIENCE_NAMESPACE)


def _ledger_prefix(snapshot: Mapping[str, Any], path: Path) -> dict[str, Any]:
    events = snapshot.get("events")
    if not isinstance(events, list) or not events:
        raise PilotV2115AcceptanceError("acceptance requires ledger event chains")
    head = events[-1].get("event_sha256")
    ledger_sha256 = snapshot.get("ledger_sha256")
    if not isinstance(head, str) or not isinstance(ledger_sha256, str):
        raise PilotV2115AcceptanceError("ledger event-chain binding is malformed")
    return {
        "file_sha256": _file_sha256(path, name=path.name),
        "ledger_sha256": ledger_sha256,
        "event_count": len(events),
        "event_chain_head": head,
    }


def _accepted_budget_baseline(
    contract: PilotContract,
    budget_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the acceptance-time total from immutable operational rows."""

    parent = budget_snapshot.get("parent_debit")
    runs = budget_snapshot.get("runs")
    caps = budget_snapshot.get("caps")
    if (
        not isinstance(parent, Mapping)
        or not isinstance(runs, Mapping)
        or not isinstance(caps, Mapping)
        or not isinstance(caps.get("stage_usd_caps"), Mapping)
    ):
        raise PilotV2115AcceptanceError(
            "budget ledger lacks its parent/operational baseline"
        )
    result = {
        "cost_usd": float(parent["cost_usd"]),
        "completions": int(parent["hosted_completions"]),
        "storage_bytes": int(parent["storage_bytes"]),
        "stage_cost_usd": {str(stage_id): 0.0 for stage_id in caps["stage_usd_caps"]},
    }
    result["stage_cost_usd"][str(parent["stage_bucket"])] = float(parent["cost_usd"])
    operational_ids = {
        spec.run_id
        for stage_id in V2115_OPERATIONAL_STAGE_IDS
        for spec in contract.expand(stage=stage_id)
    }
    if not operational_ids.issubset(runs):
        raise PilotV2115AcceptanceError(
            "budget ledger lost an accepted operational row"
        )
    for run_id in sorted(operational_ids):
        row = runs[run_id]
        actual = row.get("actual")
        if row.get("status") != "complete" or not isinstance(actual, Mapping):
            raise PilotV2115AcceptanceError(
                f"accepted operational budget row drifted: {run_id}"
            )
        stage_bucket = str(row["stage_bucket"])
        result["cost_usd"] += float(actual["cost_usd"])
        result["completions"] += int(actual["completions"])
        result["storage_bytes"] += int(actual["storage_bytes"])
        result["stage_cost_usd"][stage_bucket] += float(actual["cost_usd"])
    return result


def _verify_ledger_prefix(
    accepted: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    name: str,
) -> None:
    events = current.get("events")
    count = accepted.get("event_count")
    if (
        set(accepted) != _LEDGER_PREFIX_FIELDS
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(events, list)
        or count > len(events)
        or any(
            not isinstance(accepted.get(field), str)
            or len(str(accepted[field])) != 64
            or any(character not in "0123456789abcdef" for character in accepted[field])
            for field in ("file_sha256", "ledger_sha256", "event_chain_head")
        )
        or events[count - 1].get("event_sha256") != accepted.get("event_chain_head")
    ):
        raise PilotV2115AcceptanceError(f"{name} accepted event prefix drifted")


def _verify_unmarked_ledger_identity(
    accepted: Mapping[str, Any],
    current: Mapping[str, Any],
    ledger: Any,
    *,
    name: str,
) -> None:
    """Verify file/self hashes while the accepted prefix is still the head.

    Once an acceptance marker or legitimate science event has been appended,
    the whole-file and top-level ledger hashes necessarily change and the
    immutable event prefix becomes the durable proof. Before that first marker
    exists, however, all three identities can and must match exactly.
    """

    events = current.get("events")
    count = accepted.get("event_count")
    if (
        not isinstance(events, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
    ):
        raise PilotV2115AcceptanceError(f"{name} identity inputs are malformed")
    if len(events) != count:
        return
    path = getattr(ledger, "path", None)
    if not isinstance(path, Path):
        path = Path(path) if isinstance(path, (str, os.PathLike)) else None
    if (
        path is None
        or current.get("ledger_sha256") != accepted.get("ledger_sha256")
        or _file_sha256(path, name=f"{name} pre-marker file")
        != accepted.get("file_sha256")
    ):
        raise PilotV2115AcceptanceError(
            f"{name} pre-marker file or self-hash differs from acceptance"
        )


def _verify_acceptance_event_binding(
    receipt: Mapping[str, Any],
    run_snapshot: Mapping[str, Any],
    budget_snapshot: Mapping[str, Any],
    *,
    contract: PilotContract,
    receipt_path: str,
) -> None:
    """Require one marker immediately after each accepted ledger prefix."""

    prefixes = receipt.get("ledger_prefixes")
    integrity = receipt.get("integrity")
    if (
        not isinstance(prefixes, Mapping)
        or not isinstance(prefixes.get("run_ledger"), Mapping)
        or not isinstance(prefixes.get("budget_ledger"), Mapping)
        or not isinstance(integrity, Mapping)
    ):
        raise PilotV2115AcceptanceError(
            "acceptance receipt marker inputs are malformed"
        )
    run_prefix = prefixes["run_ledger"]
    budget_prefix = prefixes["budget_ledger"]
    expected_payload = {
        "receipt_schema_version": V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION,
        "receipt_path": receipt_path,
        "receipt_content_sha256": integrity.get("content_sha256"),
        "accepted_run_event_count": run_prefix.get("event_count"),
        "accepted_run_event_chain_head": run_prefix.get("event_chain_head"),
        "accepted_budget_event_count": budget_prefix.get("event_count"),
        "accepted_budget_event_chain_head": budget_prefix.get("event_chain_head"),
    }
    run_events = run_snapshot.get("events")
    run_index = run_prefix.get("event_count")
    prior_run_payload = (
        run_events[run_index - 1].get("payload")
        if isinstance(run_events, list)
        and isinstance(run_index, int)
        and not isinstance(run_index, bool)
        and 0 < run_index <= len(run_events)
        and isinstance(run_events[run_index - 1], Mapping)
        else None
    )
    expected_run_state_hash = (
        prior_run_payload.get("runs_sha256")
        if isinstance(prior_run_payload, Mapping)
        else None
    )
    budget_runs = budget_snapshot.get("runs")
    operational_ids = {
        spec.run_id
        for stage_id in V2115_OPERATIONAL_STAGE_IDS
        for spec in contract.expand(stage=stage_id)
    }
    expected_budget_state_hash = (
        canonical_sha256(
            {run_id: budget_runs[run_id] for run_id in sorted(operational_ids)}
        )
        if isinstance(budget_runs, Mapping) and operational_ids.issubset(budget_runs)
        else None
    )
    for name, snapshot, prefix, state_hash_field, expected_state_hash in (
        (
            "run",
            run_snapshot,
            run_prefix,
            "runs_sha256",
            expected_run_state_hash,
        ),
        (
            "budget",
            budget_snapshot,
            budget_prefix,
            "budget_runs_sha256",
            expected_budget_state_hash,
        ),
    ):
        events = snapshot.get("events")
        index = prefix.get("event_count")
        if (
            not isinstance(events, list)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index >= len(events)
        ):
            raise PilotV2115AcceptanceError(
                f"acceptance receipt is not bound after its {name}-ledger prefix"
            )
        marker = events[index]
        payload = marker.get("payload") if isinstance(marker, Mapping) else None
        marker_payload = (
            {key: payload.get(key) for key in expected_payload}
            if isinstance(payload, Mapping)
            else None
        )
        marker_events = [
            event
            for event in events
            if isinstance(event, Mapping)
            and event.get("event_type") == V2115_ACCEPTANCE_LEDGER_EVENT_TYPE
        ]
        if (
            len(marker_events) != 1
            or marker_events[0] is not marker
            or marker.get("event_index") != index
            or marker.get("event_type") != V2115_ACCEPTANCE_LEDGER_EVENT_TYPE
            or marker.get("previous_event_sha256") != prefix.get("event_chain_head")
            or not isinstance(payload, Mapping)
            or set(payload) != {*expected_payload, state_hash_field}
            or marker_payload != expected_payload
            or not isinstance(payload.get(state_hash_field), str)
            or len(payload[state_hash_field]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in payload[state_hash_field]
            )
            or payload.get(state_hash_field) != expected_state_hash
        ):
            raise PilotV2115AcceptanceError(
                f"acceptance receipt {name}-ledger marker differs from the "
                "sealed receipt"
            )


def _verify_current_budget_rows(
    contract: PilotContract,
    budget_snapshot: Mapping[str, Any],
    projection_receipt: Mapping[str, Any],
) -> None:
    """Allow only exact operational or preregistered scientific reservations."""

    runs = budget_snapshot.get("runs")
    scientific_hashes = projection_receipt.get("projection_sha256_by_run")
    if not isinstance(runs, Mapping) or not isinstance(scientific_hashes, Mapping):
        raise PilotV2115AcceptanceError(
            "budget ledger or accepted projection inventory is malformed"
        )
    operational: dict[str, dict[str, Any]] = {}
    for stage_id in V2115_OPERATIONAL_STAGE_IDS:
        for spec in contract.expand(stage=stage_id):
            projection = (
                orch._v2115_parent_import_projection(spec)
                if spec.execution_mode == "parent_authority_import"
                else orch._v2115_operational_import_projection(spec)
            )
            operational[spec.run_id] = projection.to_dict()
    allowed = set(operational) | set(scientific_hashes)
    unknown = set(runs) - allowed
    if unknown:
        raise PilotV2115AcceptanceError(
            "budget ledger contains non-preregistered reservation rows: "
            + ",".join(sorted(unknown))
        )
    if not set(operational).issubset(runs):
        raise PilotV2115AcceptanceError("budget ledger lost an operational row")
    for run_id, row in runs.items():
        reservation = row.get("reservation")
        if not isinstance(reservation, Mapping):
            raise PilotV2115AcceptanceError(
                f"budget reservation is malformed: {run_id}"
            )
        if row.get("stage_bucket") != reservation.get("stage_bucket"):
            raise PilotV2115AcceptanceError(
                f"budget row stage bucket drifted: {run_id}"
            )
        if run_id in operational:
            if reservation != operational[run_id]:
                raise PilotV2115AcceptanceError(
                    f"operational budget reservation drifted: {run_id}"
                )
        elif canonical_sha256(reservation) != scientific_hashes[run_id]:
            raise PilotV2115AcceptanceError(
                f"scientific budget reservation drifted: {run_id}"
            )


def _open_ledgers(
    contract: PilotContract,
    raw_root: Path,
) -> tuple[orch.PilotRunLedger, PilotBudgetLedger, Path, Path]:
    run_path = raw_root / "run_ledger.json"
    budget_path = raw_root / "budget_ledger.json"
    _read_regular(run_path, name="V2.11.5 run ledger")
    _read_regular(budget_path, name="V2.11.5 budget ledger")
    try:
        run_ledger = orch.PilotRunLedger(
            run_path,
            contract_hash=contract.canonical_hash,
            tamper_evident=True,
        )
        budget_ledger = PilotBudgetLedger(
            budget_path,
            contract_hash=contract.canonical_hash,
            caps=orch._budget_caps(contract),
            tamper_evident=True,
            parent_debit=orch._parent_budget_debit(contract),
        )
    except Exception as exc:
        raise PilotV2115AcceptanceError(
            f"V2.11.5 ledger validation failed: {exc}"
        ) from exc
    return run_ledger, budget_ledger, run_path, budget_path


def _expected_denominator(contract: PilotContract) -> dict[str, Any]:
    operational_specs = tuple(
        spec
        for stage_id in V2115_OPERATIONAL_STAGE_IDS
        for spec in contract.expand(stage=stage_id)
    )
    scientific_specs = tuple(
        spec
        for stage_id in V2115_SCIENTIFIC_STAGE_IDS
        for spec in contract.expand(stage=stage_id)
    )
    offline_specs = tuple(
        spec
        for spec in scientific_specs
        if spec.execution_mode == "offline_candidate_admission"
    )
    return {
        "ledger_cells": len(operational_specs) + len(scientific_specs),
        "operational_cells": len(operational_specs),
        "scientific_cells": len(scientific_specs),
        "provider_backed_scientific_cells": len(scientific_specs) - len(offline_specs),
        "offline_scientific_cells": len(offline_specs),
        "stage_cell_counts": dict(V2115_EXPECTED_STAGE_CELL_COUNTS),
        "operational_status": "complete",
        "scientific_status": "scheduled",
        "scientific_run_ids_sha256": canonical_sha256(
            sorted(spec.run_id for spec in scientific_specs)
        ),
    }


def _audit_denominator(
    contract: PilotContract,
    run_snapshot: Mapping[str, Any],
    budget_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    specs = tuple(contract.expand())
    runs = run_snapshot.get("runs")
    if (
        len(specs) != V2115_EXPECTED_LEDGER_CELLS
        or not isinstance(runs, Mapping)
        or set(runs) != {spec.run_id for spec in specs}
        or any(runs[spec.run_id].get("spec") != spec.to_dict() for spec in specs)
    ):
        raise PilotV2115AcceptanceError("V2.11.5 run-ledger denominator drifted")

    operational_specs = tuple(
        spec
        for stage_id in V2115_OPERATIONAL_STAGE_IDS
        for spec in contract.expand(stage=stage_id)
    )
    scientific_specs = tuple(
        spec
        for stage_id in V2115_SCIENTIFIC_STAGE_IDS
        for spec in contract.expand(stage=stage_id)
    )
    offline_specs = tuple(
        spec
        for spec in scientific_specs
        if spec.execution_mode == "offline_candidate_admission"
    )
    provider_specs = tuple(
        spec
        for spec in scientific_specs
        if spec.execution_mode != "offline_candidate_admission"
    )
    if (
        len(operational_specs) != V2115_EXPECTED_OPERATIONAL_CELLS
        or len(scientific_specs) != V2115_EXPECTED_SCIENTIFIC_CELLS
        or len(provider_specs) != V2115_EXPECTED_PROVIDER_CELLS
        or len(offline_specs) != V2115_EXPECTED_OFFLINE_CELLS
        or any(
            runs[spec.run_id].get("status") != "complete" for spec in operational_specs
        )
        or any(
            runs[spec.run_id].get("status") != "scheduled" for spec in scientific_specs
        )
        or {(spec.stage_id, spec.arm_id, spec.model_id) for spec in offline_specs}
        != {("experiment-c", "verified-error-candidate", "gpt52_main")}
    ):
        raise PilotV2115AcceptanceError(
            "V2.11.5 acceptance requires five complete operational cells and "
            "all 131 scientific cells still scheduled"
        )

    budget_runs = budget_snapshot.get("runs")
    operational_ids = {spec.run_id for spec in operational_specs}
    if not isinstance(budget_runs, Mapping) or set(budget_runs) != operational_ids:
        raise PilotV2115AcceptanceError(
            "budget ledger must contain exactly the five zero-call operational rows"
        )
    operational_by_id = {spec.run_id: spec for spec in operational_specs}
    for run_id, row in budget_runs.items():
        reservation = row.get("reservation")
        actual = row.get("actual")
        spec = operational_by_id[run_id]
        expected_projection = (
            orch._v2115_parent_import_projection(spec)
            if spec.execution_mode == "parent_authority_import"
            else orch._v2115_operational_import_projection(spec)
        ).to_dict()
        if (
            row.get("status") != "complete"
            or not isinstance(reservation, Mapping)
            or not isinstance(actual, Mapping)
            or reservation != expected_projection
            or row.get("stage_bucket") != spec.budget_bucket
            or float(reservation.get("cost_usd", math.nan)) != 0.0
            or int(reservation.get("completions", -1)) != 0
            or float(actual.get("cost_usd", math.nan)) != 0.0
            or int(actual.get("completions", -1)) != 0
        ):
            raise PilotV2115AcceptanceError(
                f"operational budget row is not complete and zero-call: {run_id}"
            )
    return _expected_denominator(contract)


def _audit_operational_receipts(
    contract: PilotContract,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: orch.GitProvenance,
    run_ledger: orch.PilotRunLedger,
) -> tuple[dict[str, Any], dict[str, str]]:
    result: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for stage_id in V2115_OPERATIONAL_STAGE_IDS:
        path = orch._stage_receipt_path(raw_root, stage_id)
        receipt = _strict_json(path, name=f"{stage_id} stage receipt")
        try:
            verified = orch._verify_v2_stage_receipt(
                contract,
                stage_id,
                receipt,
                raw_root=raw_root,
                ledger=run_ledger,
                paid=paid,
                authority_repo_root=repo_root,
            )
        except Exception as exc:
            raise PilotV2115AcceptanceError(
                f"{stage_id} stage receipt failed replay: {exc}"
            ) from exc
        if (
            verified.get("status") != "complete"
            or verified.get("terminal") is not True
            or verified.get("go") is not True
            or verified.get("registered_run_count")
            != V2115_EXPECTED_STAGE_CELL_COUNTS[stage_id]
        ):
            raise PilotV2115AcceptanceError(
                f"{stage_id} is not an exact complete/go operational gate"
            )
        relative = _relative(repo_root, path, name=f"{stage_id} stage receipt")
        digest = _file_sha256(path, name=f"{stage_id} stage receipt")
        sources[relative] = digest
        result[stage_id] = {
            "path": relative,
            "file_sha256": digest,
            "content_sha256": verified["integrity"]["content_sha256"],
            "status": verified["status"],
            "go": verified["go"],
            "registered_run_count": verified["registered_run_count"],
        }
    return result, sources


def _audit_authorities(
    contract: PilotContract,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: orch.GitProvenance,
) -> tuple[dict[str, Any], dict[str, str]]:
    sources: dict[str, str] = {}
    parent_path = repo_root.joinpath(*V2115_DEFAULT_RECEIPT_PATH.parts)
    try:
        parent = verify_v2115_parent_import_receipt(
            parent_path,
            repo_root=repo_root,
            contract=contract,
            raw_root=raw_root,
            expected_git_commit=paid.head_commit,
            evidence_repo_root=repo_root,
        )
    except Exception as exc:
        raise PilotV2115AcceptanceError(
            f"V2.11.5 parent receipt failed verification: {exc}"
        ) from exc
    parent_relative = _relative(repo_root, parent_path, name="parent receipt")
    sources[parent_relative] = _file_sha256(parent_path, name="parent receipt")

    per_model: dict[str, Any] = {}
    for model_id in V2115_ALLOWED_MODELS:
        receipt_path = v2115_observed_p95_receipt_path(raw_root, model_id)
        projection_path = v2115_observed_p95_projection_path(raw_root, model_id)
        try:
            binding = verified_v2115_observed_p95_authority_binding(
                receipt_path,
                repo_root=repo_root,
                raw_root=raw_root,
                expected_git_commit=paid.head_commit,
                contract=contract,
            )
        except Exception as exc:
            raise PilotV2115AcceptanceError(
                f"{model_id} observed-p95 authority failed verification: {exc}"
            ) from exc
        receipt_relative = _relative(
            repo_root, receipt_path, name=f"{model_id} p95 receipt"
        )
        projection_relative = _relative(
            repo_root, projection_path, name=f"{model_id} p95 projection"
        )
        sources[receipt_relative] = _file_sha256(
            receipt_path, name=f"{model_id} p95 receipt"
        )
        sources[projection_relative] = _file_sha256(
            projection_path, name=f"{model_id} p95 projection"
        )
        per_model[model_id] = {
            "binding_sha256": canonical_sha256(binding),
            "receipt_path": receipt_relative,
            "receipt_file_sha256": binding["receipt_file_sha256"],
            "receipt_content_sha256": binding["receipt_content_sha256"],
            "projection_path": projection_relative,
            "projection_file_sha256": sources[projection_relative],
        }

    global_relative = V2115_POST_GATE_RELATIVE_PATH.as_posix()
    try:
        dedicated = verified_v2115_gate_authority_binding(
            global_relative,
            repo_root=repo_root,
            expected_git_commit=paid.head_commit,
            expected_contract_sha256=contract.canonical_hash,
            contract=contract,
        )
        generic = verified_observed_p95_authority_binding(
            global_relative,
            repo_root=repo_root,
            expected_git_commit=paid.head_commit,
        )
    except Exception as exc:
        raise PilotV2115AcceptanceError(
            f"V2.11.5 global observed-p95 consumer failed verification: {exc}"
        ) from exc
    if generic != dedicated:
        raise PilotV2115AcceptanceError(
            "generic and dedicated V2.11.5 authority bindings differ"
        )
    global_path = repo_root.joinpath(*V2115_POST_GATE_RELATIVE_PATH.parts)
    sources[global_relative] = _file_sha256(global_path, name="global p95 gate")
    return (
        {
            "parent_import": {
                "path": parent_relative,
                "file_sha256": sources[parent_relative],
                "content_sha256": parent["integrity"]["content_sha256"],
                "scientific_evidence": False,
            },
            "per_model_observed_p95": per_model,
            "global_observed_p95": {
                "binding_sha256": canonical_sha256(dedicated),
                "receipt_path": global_relative,
                "receipt_file_sha256": dedicated["receipt_file_sha256"],
                "receipt_content_sha256": dedicated["receipt_content_sha256"],
                "generic_adapter_exact_match": True,
            },
        },
        sources,
    )


def _science_specs(contract: PilotContract) -> tuple[Any, ...]:
    return tuple(
        spec
        for stage_id in V2115_SCIENTIFIC_STAGE_IDS
        for spec in contract.expand(stage=stage_id)
    )


def _audit_configs_and_d_groups(
    contract: PilotContract,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: orch.GitProvenance,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provider_specs = tuple(
        spec
        for spec in _science_specs(contract)
        if spec.execution_mode != "offline_candidate_admission"
    )
    config_hashes: dict[str, str] = {}
    configs: dict[str, Any] = {}
    with observed_p95_authority_repo_context(repo_root):
        for spec in provider_specs:
            try:
                reservations = orch._runner_p95_reservations(
                    contract,
                    spec.model_id,
                    raw_root=raw_root,
                    paid=paid,
                    authority_repo_root=repo_root,
                )
                config = orch.config_for_spec(
                    contract,
                    spec,
                    raw_root=raw_root,
                    paid_provenance=paid,
                    authority_repo_root=repo_root,
                    verify_bound_inputs=True,
                    preflight_p95_reservations=reservations,
                )
                payload = config.to_dict()
                restored = config_from_dict(payload)
                if (
                    restored.to_dict() != payload
                    or not has_sealed_observed_p95_authority(config)
                    or not has_sealed_observed_p95_authority(restored)
                    or not serialized_has_sealed_observed_p95_authority(
                        payload, authority_repo_root=repo_root
                    )
                ):
                    raise PilotV2115AcceptanceError(
                        f"{spec.run_id} runner config lost its sealed p95 authority"
                    )
                validated = validate_preflight_p95_reservations(
                    config,
                    provider_model_name=orch._runtime_model_for_profile(
                        contract.provider_profiles[spec.model_id]
                    ),
                )
                if not validated:
                    raise PilotV2115AcceptanceError(
                        f"{spec.run_id} has no validated dispatch reservations"
                    )
            except PilotV2115AcceptanceError:
                raise
            except Exception as exc:
                raise PilotV2115AcceptanceError(
                    f"{spec.run_id} runner config failed validation: {exc}"
                ) from exc
            configs[spec.run_id] = config
            config_hashes[spec.run_id] = canonical_sha256(payload)

    if len(config_hashes) != V2115_EXPECTED_PROVIDER_CELLS:
        raise PilotV2115AcceptanceError(
            "runner-config acceptance did not cover all 126 provider cells"
        )

    d_specs = tuple(contract.expand(stage="experiment-d"))
    groups: list[dict[str, Any]] = []
    for model_id, seed in sorted(
        {(spec.model_id, spec.environment_seed) for spec in d_specs}
    ):
        group_specs = tuple(
            spec
            for spec in d_specs
            if spec.model_id == model_id and spec.environment_seed == seed
        )
        representative = next(
            (spec for spec in group_specs if spec.arm_id == "matched-a"), None
        )
        if representative is None:
            raise PilotV2115AcceptanceError(
                "Experiment D group lacks its matched-a representative"
            )
        try:
            plan = orch.build_v2115_experiment_d_group_plan(
                contract,
                group_specs,
                base_config=configs[representative.run_id],
            )
        except Exception as exc:
            raise PilotV2115AcceptanceError(
                f"Experiment D group {model_id}/{seed} failed validation: {exc}"
            ) from exc
        receipt = plan.to_receipt()
        if receipt.get("cell_count") != V2115_EXPECTED_D_CELLS_PER_GROUP:
            raise PilotV2115AcceptanceError(
                "Experiment D group does not contain exactly eleven cells"
            )
        groups.append(receipt)
    if len(groups) != V2115_EXPECTED_D_GROUPS:
        raise PilotV2115AcceptanceError(
            "Experiment D acceptance requires exactly five checkpoint groups"
        )
    return (
        {
            "provider_config_count": len(config_hashes),
            "config_sha256_by_run": dict(sorted(config_hashes.items())),
            "config_set_sha256": canonical_sha256(config_hashes),
            "roundtrip_exact": True,
            "sealed_observed_p95_authority": True,
        },
        {
            "group_count": len(groups),
            "cells_per_group": V2115_EXPECTED_D_CELLS_PER_GROUP,
            "groups": groups,
            "groups_sha256": canonical_sha256(groups),
        },
    )


def _audit_projections(
    contract: PilotContract,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: orch.GitProvenance,
    run_ledger: orch.PilotRunLedger,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    projections: list[tuple[str, RunProjection]] = []
    for stage_id in V2115_SCIENTIFIC_STAGE_IDS:
        specs = tuple(contract.expand(stage=stage_id))
        if stage_id == "experiment-d":
            for model_id, seed in sorted(
                {(spec.model_id, spec.environment_seed) for spec in specs}
            ):
                group_specs = tuple(
                    spec
                    for spec in specs
                    if spec.model_id == model_id and spec.environment_seed == seed
                )
                representative = next(
                    spec for spec in group_specs if spec.arm_id == "matched-a"
                )
                projections.append(
                    (
                        model_id,
                        orch._d_group_projection(
                            contract,
                            representative,
                            raw_root=raw_root,
                            paid=paid,
                            authority_repo_root=repo_root,
                        ),
                    )
                )
            continue
        for spec in specs:
            if spec.execution_mode == "offline_candidate_admission":
                projection = RunProjection(
                    run_id=spec.run_id,
                    stage_bucket=spec.budget_bucket,
                    cost_usd=0.0,
                    completions=0,
                    storage_bytes=2_000_000,
                    basis={"method": "offline-zero-provider-call"},
                )
            else:
                projection = orch.projection_from_preflight(
                    contract,
                    spec,
                    raw_root=raw_root,
                    paid=paid,
                    authority_repo_root=repo_root,
                )
            projections.append((spec.model_id, projection))

    values = [projection for _, projection in projections]
    if len(values) != V2115_EXPECTED_PROJECTION_UNITS or len(
        {projection.run_id for projection in values}
    ) != len(values):
        raise PilotV2115AcceptanceError(
            "scientific projection denominator must contain 81 unique units"
        )
    calls = {"action": 0, "semantic": 0}
    calls_by_model = {
        model_id: {"action": 0, "semantic": 0} for model_id in V2115_ALLOWED_MODELS
    }
    for model_id, projection in projections:
        by_kind = projection.basis.get("calls_by_kind", {})
        if not isinstance(by_kind, Mapping):
            raise PilotV2115AcceptanceError("projection call-kind basis is malformed")
        for call_kind in calls:
            count = int(by_kind.get(call_kind, 0))
            calls[call_kind] += count
            calls_by_model[model_id][call_kind] += count
    forward = contract.v2115_forward_boundary
    if not isinstance(forward, Mapping) or not isinstance(
        forward.get("matrix"), Mapping
    ):
        raise PilotV2115AcceptanceError(
            "V2.11.5 forward matrix is unavailable during projection audit"
        )
    expected_by_model = forward["matrix"]["fresh_calls_by_model"]
    if (
        calls
        != {
            "action": V2115_EXPECTED_ACTION_CALLS,
            "semantic": V2115_EXPECTED_SEMANTIC_CALLS,
        }
        or sum(calls.values()) != V2115_EXPECTED_PROVIDER_CALLS
        or calls_by_model != expected_by_model
        or sum(projection.completions for projection in values)
        != V2115_EXPECTED_PROVIDER_CALLS
        or sum(projection.storage_bytes for projection in values)
        != V2115_EXPECTED_FRESH_STORAGE_BYTES
    ):
        raise PilotV2115AcceptanceError(
            "scientific projection calls/storage differ from the frozen matrix"
        )
    try:
        orch._assert_projection_matrix_fits(budget_ledger, values)
    except Exception as exc:
        raise PilotV2115AcceptanceError(
            f"complete V2.11.5 projected matrix exceeds a hard cap: {exc}"
        ) from exc

    current = _accepted_budget_baseline(contract, budget_ledger.snapshot())
    projected_stage = dict(current["stage_cost_usd"])
    for projection in values:
        projected_stage[projection.stage_bucket] = (
            float(projected_stage[projection.stage_bucket]) + projection.cost_usd
        )
    projected = {
        "cost_usd": float(current["cost_usd"])
        + sum(projection.cost_usd for projection in values),
        "completions": int(current["completions"])
        + sum(projection.completions for projection in values),
        "storage_bytes": int(current["storage_bytes"])
        + sum(projection.storage_bytes for projection in values),
        "stage_cost_usd": projected_stage,
    }
    caps = budget_ledger.caps.to_dict()
    if (
        projected["cost_usd"] > float(caps["dispatchable_usd"]) + 1e-12
        or projected["completions"] > int(caps["max_completions"])
        or projected["storage_bytes"] > int(caps["max_storage_bytes"])
    ):
        raise PilotV2115AcceptanceError("projected matrix exceeds global hard caps")
    return {
        "projection_unit_count": len(values),
        "projection_sha256_by_run": {
            projection.run_id: canonical_sha256(projection.to_dict())
            for projection in values
        },
        "projection_set_sha256": canonical_sha256(
            {projection.run_id: projection.to_dict() for projection in values}
        ),
        "fresh_calls_by_kind": calls,
        "fresh_calls_by_model": calls_by_model,
        "fresh_provider_calls": sum(calls.values()),
        "fresh_projected_cost_usd": sum(projection.cost_usd for projection in values),
        "fresh_projected_completions": sum(
            projection.completions for projection in values
        ),
        "fresh_projected_storage_bytes": sum(
            projection.storage_bytes for projection in values
        ),
        "current_committed_plus_reserved": current,
        "cumulative_after_full_projection": projected,
        "hard_caps": caps,
        "full_matrix_fits": True,
    }


def _source_binding(repo_root: Path, path: Path, *, name: str) -> dict[str, Any]:
    value = _strict_json(path, name=name)
    return {
        "path": _relative(repo_root, path, name=name),
        "file_sha256": _file_sha256(path, name=name),
        "content_sha256": canonical_sha256(value),
    }


def _static_acceptance_material(
    contract: PilotContract,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: orch.GitProvenance,
    run_ledger: orch.PilotRunLedger,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    """Recompute every deterministic acceptance field from bound sources."""

    launch_path = raw_root / "scientific_launch_input.json"
    release_path = raw_root / "release_attestation.json"
    release_value = _strict_json(release_path, name="release attestation")
    if paid.release_attestation is None or release_value != dict(
        paid.release_attestation
    ):
        raise PilotV2115AcceptanceError(
            "persisted release attestation differs from verified CI provenance"
        )
    operational, stage_sources = _audit_operational_receipts(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
        run_ledger=run_ledger,
    )
    authorities, authority_sources = _audit_authorities(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    configs, d_groups = _audit_configs_and_d_groups(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    projections = _audit_projections(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
        run_ledger=run_ledger,
        budget_ledger=budget_ledger,
    )
    launch = _source_binding(repo_root, launch_path, name="scientific launch input")
    release = _source_binding(repo_root, release_path, name="release attestation")
    sources = {**stage_sources, **authority_sources}
    sources[launch["path"]] = launch["file_sha256"]
    sources[release["path"]] = release["file_sha256"]
    return {
        "release": {
            "git_tag": paid.git_tag,
            "git_commit": paid.head_commit,
            "tag_commit": paid.tag_commit,
            "tag_object_type": paid.tag_object_type,
            "worktree_clean": paid.worktree_clean,
            "contract_binding_sha256": canonical_sha256(paid.contract_binding),
            "scientific_launch_input": launch,
            "release_attestation": release,
        },
        "operational_gates": operational,
        "authorities": authorities,
        "runner_configs": configs,
        "experiment_d": d_groups,
        "budget_projection": projections,
        "bound_source_file_sha256": dict(sorted(sources.items())),
    }


def audit_v2115_scientific_dispatch(
    contract: PilotContract,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    scientific_launch_input_path: str | Path,
) -> dict[str, Any]:
    """Build, but do not persist, the complete zero-provider acceptance."""

    repository, raw = _exact_roots(repo_root, raw_root)
    _require_contract(contract, repository)
    _require_provider_keys_absent()
    launch_path = Path(scientific_launch_input_path).absolute()
    expected_launch = raw / "scientific_launch_input.json"
    if launch_path != expected_launch:
        raise PilotV2115AcceptanceError(
            "acceptance requires the exact raw scientific_launch_input.json"
        )

    run_ledger, budget_ledger, run_path, budget_path = _open_ledgers(contract, raw)
    run_before = _file_sha256(run_path, name="V2.11.5 run ledger")
    budget_before = _file_sha256(budget_path, name="V2.11.5 budget ledger")
    run_snapshot = run_ledger.snapshot()
    budget_snapshot = budget_ledger.snapshot()
    if (
        len(run_snapshot.get("events", ())) != V2115_EXPECTED_ACCEPTED_RUN_EVENTS
        or len(budget_snapshot.get("events", ()))
        != V2115_EXPECTED_ACCEPTED_BUDGET_EVENTS
    ):
        raise PilotV2115AcceptanceError(
            "acceptance ledger prefixes must end immediately after five "
            "operational cells"
        )

    with _provider_boundary_stack():
        try:
            paid = orch.verify_paid_provenance(
                contract,
                repo_root=repository,
                scientific_launch_input_path=launch_path,
            )
        except Exception as exc:
            raise PilotV2115AcceptanceError(
                f"scientific release provenance failed: {exc}"
            ) from exc
        denominator = _audit_denominator(contract, run_snapshot, budget_snapshot)
        material = _static_acceptance_material(
            contract,
            repo_root=repository,
            raw_root=raw,
            paid=paid,
            run_ledger=run_ledger,
            budget_ledger=budget_ledger,
        )

    if (
        _file_sha256(run_path, name="V2.11.5 run ledger") != run_before
        or _file_sha256(budget_path, name="V2.11.5 budget ledger") != budget_before
    ):
        raise PilotV2115AcceptanceError("acceptance audit mutated a scientific ledger")
    receipt = {
        "schema_version": V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION,
        "status": "go",
        "go": True,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "release": material["release"],
        "raw_namespace": _relative(repository, raw, name="V2.11.5 raw root"),
        "pre_science_namespace": _audit_pre_science_namespace(raw, contract),
        "denominator": denominator,
        "operational_gates": material["operational_gates"],
        "authorities": material["authorities"],
        "runner_configs": material["runner_configs"],
        "experiment_d": material["experiment_d"],
        "budget_projection": material["budget_projection"],
        "ledger_prefixes": {
            "run_ledger": _ledger_prefix(run_snapshot, run_path),
            "budget_ledger": _ledger_prefix(budget_snapshot, budget_path),
        },
        "bound_source_file_sha256": material["bound_source_file_sha256"],
        "provider_boundary": _expected_provider_boundary(),
        "scientific_evidence": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    return _seal(receipt)


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write all bytes, exposing one narrow injection point for crash tests."""

    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("acceptance receipt write made no progress")
        remaining = remaining[written:]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _persist_exact_receipt(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise PilotV2115AcceptanceError("acceptance receipt must not be a symlink")
    if path.exists():
        existing = _strict_json(path, name="scientific-dispatch acceptance receipt")
        if existing != dict(value):
            raise PilotV2115AcceptanceError(
                "immutable scientific-dispatch acceptance receipt drifted"
            )
        return
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise PilotV2115AcceptanceError(
                "acceptance receipt appeared concurrently; audit before reuse"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(path.parent)


def _require_recoverable_marker_state(
    run_snapshot: Mapping[str, Any],
    budget_snapshot: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Allow only the four crash-recovery states before science starts."""

    result: list[bool] = []
    for name, snapshot, prefix_count in (
        ("run", run_snapshot, V2115_EXPECTED_ACCEPTED_RUN_EVENTS),
        ("budget", budget_snapshot, V2115_EXPECTED_ACCEPTED_BUDGET_EVENTS),
    ):
        events = snapshot.get("events")
        if not isinstance(events, list) or len(events) not in {
            prefix_count,
            prefix_count + 1,
        }:
            raise PilotV2115AcceptanceError(
                f"{name} ledger is outside the recoverable acceptance marker states"
            )
        present = len(events) == prefix_count + 1
        if (
            present
            and events[prefix_count].get("event_type")
            != V2115_ACCEPTANCE_LEDGER_EVENT_TYPE
        ):
            raise PilotV2115AcceptanceError(
                f"{name} ledger has a non-acceptance event after the sealed prefix"
            )
        result.append(present)
    return result[0], result[1]


def accept_v2115_scientific_dispatch(
    *,
    contract_path: str | Path,
    repo_root: str | Path,
    raw_root: str | Path,
    scientific_launch_input_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit and atomically persist the unique V2.11.5 acceptance receipt."""

    repository, raw = _exact_roots(repo_root, raw_root)
    contract_candidate = Path(contract_path)
    if not contract_candidate.is_absolute():
        contract_candidate = repository / contract_candidate
    if contract_candidate.absolute() != repository.joinpath(*V2115_CONTRACT_PATH.parts):
        raise PilotV2115AcceptanceError(
            "acceptance requires experiments/pilot_v2_11_5.yaml"
        )
    contract = load_pilot_contract(contract_candidate)
    output = (
        raw / V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_FILENAME
        if receipt_path is None
        else Path(receipt_path).absolute()
    )
    expected = raw / V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_FILENAME
    if output != expected:
        raise PilotV2115AcceptanceError(
            "scientific-dispatch acceptance must use its unique raw receipt path"
        )
    with orch._exclusive_real_stage_lock(
        raw, stage_id="scientific-dispatch-acceptance"
    ):
        if not output.exists():
            audited = audit_v2115_scientific_dispatch(
                contract,
                repo_root=repository,
                raw_root=raw,
                scientific_launch_input_path=scientific_launch_input_path,
            )
            _persist_exact_receipt(output, audited)

        _require_contract(contract, repository)
        _require_provider_keys_absent()
        _audit_pre_science_namespace(raw, contract)
        launch_path = Path(scientific_launch_input_path).absolute()
        if launch_path != raw / "scientific_launch_input.json":
            raise PilotV2115AcceptanceError(
                "acceptance requires the exact raw scientific_launch_input.json"
            )
        run_ledger, budget_ledger, _run_path, _budget_path = _open_ledgers(
            contract, raw
        )
        run_snapshot = run_ledger.snapshot()
        budget_snapshot = budget_ledger.snapshot()
        _audit_denominator(contract, run_snapshot, budget_snapshot)
        _require_recoverable_marker_state(run_snapshot, budget_snapshot)
        with _provider_boundary_stack():
            try:
                paid = orch.verify_paid_provenance(
                    contract,
                    repo_root=repository,
                    scientific_launch_input_path=launch_path,
                )
            except Exception as exc:
                raise PilotV2115AcceptanceError(
                    f"scientific release provenance failed: {exc}"
                ) from exc
        receipt = _verify_v2115_scientific_dispatch_acceptance_core(
            output,
            contract=contract,
            repo_root=repository,
            raw_root=raw,
            paid=paid,
            run_ledger=run_ledger,
            budget_ledger=budget_ledger,
            require_markers=False,
        )
        prefixes = receipt["ledger_prefixes"]
        run_prefix = prefixes["run_ledger"]
        budget_prefix = prefixes["budget_ledger"]
        relative_receipt_path = _relative(
            repository, output, name="scientific-dispatch acceptance receipt"
        )
        try:
            run_ledger.bind_acceptance_receipt(
                receipt_schema_version=(
                    V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION
                ),
                receipt_path=relative_receipt_path,
                receipt_content_sha256=receipt["integrity"]["content_sha256"],
                accepted_run_event_count=run_prefix["event_count"],
                accepted_run_event_chain_head=run_prefix["event_chain_head"],
                accepted_budget_event_count=budget_prefix["event_count"],
                accepted_budget_event_chain_head=budget_prefix["event_chain_head"],
            )
        except Exception as exc:
            raise PilotV2115AcceptanceError(
                f"failed to bind acceptance receipt into the run ledger: {exc}"
            ) from exc
        try:
            budget_ledger.bind_acceptance_receipt(
                receipt_schema_version=(
                    V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION
                ),
                receipt_path=relative_receipt_path,
                receipt_content_sha256=receipt["integrity"]["content_sha256"],
                accepted_run_event_count=run_prefix["event_count"],
                accepted_run_event_chain_head=run_prefix["event_chain_head"],
                accepted_budget_event_count=budget_prefix["event_count"],
                accepted_budget_event_chain_head=budget_prefix["event_chain_head"],
            )
        except Exception as exc:
            raise PilotV2115AcceptanceError(
                f"failed to bind acceptance receipt into the budget ledger: {exc}"
            ) from exc

        reloaded_run, reloaded_budget, _run_path, _budget_path = _open_ledgers(
            contract, raw
        )
        return verify_v2115_scientific_dispatch_acceptance(
            output,
            contract=contract,
            repo_root=repository,
            raw_root=raw,
            paid=paid,
            run_ledger=reloaded_run,
            budget_ledger=reloaded_budget,
        )


def _verify_v2115_scientific_dispatch_acceptance_core(
    receipt_path: str | Path,
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: orch.GitProvenance,
    run_ledger: orch.PilotRunLedger,
    budget_ledger: PilotBudgetLedger,
    require_markers: bool,
) -> dict[str, Any]:
    """Verify an accepted prefix before any scientific provider construction."""

    repository, raw = _exact_roots(repo_root, raw_root)
    _require_contract(contract, repository)
    path = Path(receipt_path).absolute()
    expected = raw / V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_FILENAME
    if path != expected:
        raise PilotV2115AcceptanceError(
            "scientific stage requires the unique acceptance receipt path"
        )
    receipt = _strict_json(path, name="scientific-dispatch acceptance receipt")
    _verify_seal(receipt)
    provider = receipt.get("provider_boundary")
    release = receipt.get("release")
    denominator = receipt.get("denominator")
    configs = receipt.get("runner_configs")
    d_groups = receipt.get("experiment_d")
    projection = receipt.get("budget_projection")
    prefixes = receipt.get("ledger_prefixes")
    if (
        set(receipt) != _ACCEPTANCE_TOP_LEVEL_FIELDS
        or receipt.get("schema_version")
        != V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION
        or receipt.get("status") != "go"
        or receipt.get("go") is not True
        or receipt.get("contract_id") != contract.contract_id
        or receipt.get("contract_sha256") != contract.canonical_hash
        or receipt.get("raw_namespace") != V2115_RAW_ROOT.as_posix()
        or receipt.get("pre_science_namespace") != _PRE_SCIENCE_NAMESPACE
        or receipt.get("scientific_evidence") is not False
        or receipt.get("claim_boundary") != _CLAIM_BOUNDARY
        or provider != _expected_provider_boundary()
        or not isinstance(release, Mapping)
        or release.get("git_tag") != paid.git_tag
        or release.get("git_commit") != paid.head_commit
        or release.get("tag_object_type") != "tag"
        or release.get("worktree_clean") is not True
        or denominator != _expected_denominator(contract)
        or not isinstance(configs, Mapping)
        or configs.get("provider_config_count") != V2115_EXPECTED_PROVIDER_CELLS
        or not isinstance(d_groups, Mapping)
        or d_groups.get("group_count") != V2115_EXPECTED_D_GROUPS
        or d_groups.get("cells_per_group") != V2115_EXPECTED_D_CELLS_PER_GROUP
        or not isinstance(projection, Mapping)
        or projection.get("projection_unit_count") != V2115_EXPECTED_PROJECTION_UNITS
        or projection.get("fresh_provider_calls") != V2115_EXPECTED_PROVIDER_CALLS
        or projection.get("fresh_calls_by_kind")
        != {
            "action": V2115_EXPECTED_ACTION_CALLS,
            "semantic": V2115_EXPECTED_SEMANTIC_CALLS,
        }
        or projection.get("full_matrix_fits") is not True
        or not isinstance(prefixes, Mapping)
        or set(prefixes) != {"run_ledger", "budget_ledger"}
    ):
        raise PilotV2115AcceptanceError(
            "scientific-dispatch acceptance identity or denominator drifted"
        )

    run_prefix = prefixes.get("run_ledger")
    budget_prefix = prefixes.get("budget_ledger")
    if (
        not isinstance(run_prefix, Mapping)
        or set(run_prefix) != _LEDGER_PREFIX_FIELDS
        or run_prefix.get("event_count") != V2115_EXPECTED_ACCEPTED_RUN_EVENTS
        or not isinstance(budget_prefix, Mapping)
        or set(budget_prefix) != _LEDGER_PREFIX_FIELDS
        or budget_prefix.get("event_count") != V2115_EXPECTED_ACCEPTED_BUDGET_EVENTS
    ):
        raise PilotV2115AcceptanceError(
            "acceptance receipt does not bind the exact post-operational prefixes"
        )

    run_snapshot = run_ledger.snapshot()
    budget_snapshot = budget_ledger.snapshot()
    if not require_markers:
        _verify_unmarked_ledger_identity(
            run_prefix,
            run_snapshot,
            run_ledger,
            name="run ledger",
        )
        _verify_unmarked_ledger_identity(
            budget_prefix,
            budget_snapshot,
            budget_ledger,
            name="budget ledger",
        )
    else:
        _verify_acceptance_event_binding(
            receipt,
            run_snapshot,
            budget_snapshot,
            contract=contract,
            receipt_path=_relative(
                repository, path, name="scientific-dispatch acceptance receipt"
            ),
        )

    with _provider_boundary_stack():
        recomputed = _static_acceptance_material(
            contract,
            repo_root=repository,
            raw_root=raw,
            paid=paid,
            run_ledger=run_ledger,
            budget_ledger=budget_ledger,
        )
    for field in (
        "release",
        "operational_gates",
        "authorities",
        "runner_configs",
        "experiment_d",
        "budget_projection",
        "bound_source_file_sha256",
    ):
        if receipt.get(field) != recomputed[field]:
            raise PilotV2115AcceptanceError(
                f"scientific-dispatch acceptance field {field!r} differs from "
                "source recomputation"
            )

    sources = receipt.get("bound_source_file_sha256")
    if not isinstance(sources, Mapping) or not sources:
        raise PilotV2115AcceptanceError("acceptance source bindings are missing")
    for relative, expected_hash in sources.items():
        candidate = PurePosixPath(str(relative))
        if candidate.is_absolute() or any(
            part in {"", ".", ".."} for part in candidate.parts
        ):
            raise PilotV2115AcceptanceError("acceptance source path is not normalized")
        observed = _file_sha256(
            repository.joinpath(*candidate.parts), name=f"accepted source {relative}"
        )
        if observed != expected_hash:
            raise PilotV2115AcceptanceError(f"accepted source file drifted: {relative}")

    _verify_ledger_prefix(
        prefixes.get("run_ledger", {}), run_snapshot, name="run ledger"
    )
    _verify_ledger_prefix(
        prefixes.get("budget_ledger", {}), budget_snapshot, name="budget ledger"
    )
    current_runs = run_snapshot.get("runs")
    specs = tuple(contract.expand())
    if (
        not isinstance(current_runs, Mapping)
        or set(current_runs) != {spec.run_id for spec in specs}
        or any(
            current_runs[spec.run_id].get("spec") != spec.to_dict() for spec in specs
        )
    ):
        raise PilotV2115AcceptanceError(
            "current run ledger no longer preserves the accepted ITT denominator"
        )
    if budget_snapshot.get("caps") != projection.get("hard_caps"):
        raise PilotV2115AcceptanceError("current budget caps differ from acceptance")
    _verify_current_budget_rows(contract, budget_snapshot, projection)
    return _json_copy(receipt)


def verify_v2115_scientific_dispatch_acceptance(
    receipt_path: str | Path,
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: orch.GitProvenance,
    run_ledger: orch.PilotRunLedger,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    """Verify the accepted receipt and require both append-only markers."""

    return _verify_v2115_scientific_dispatch_acceptance_core(
        receipt_path,
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
        run_ledger=run_ledger,
        budget_ledger=budget_ledger,
        require_markers=True,
    )


__all__ = [
    "PilotV2115AcceptanceError",
    "V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_FILENAME",
    "V2115_SCIENTIFIC_DISPATCH_ACCEPTANCE_SCHEMA_VERSION",
    "accept_v2115_scientific_dispatch",
    "audit_v2115_scientific_dispatch",
    "verify_v2115_scientific_dispatch_acceptance",
]
