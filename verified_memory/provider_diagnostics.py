"""Bounded, non-scientific provider interface probes.

This module deliberately does not share the scientific ``run_pilot.py`` entry.
It emits one redacted receipt for one provider call, requires a clean annotated
debug tag, and refuses to write into the immutable pilot-v1 namespace.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping

from llm_providers import (
    PINNED_PROVIDER_SDK_VERSIONS,
    SAFE_PROVIDER_ERROR_TYPES,
    ProviderErrorDetails,
    StructuredCompletion,
    create_llm_provider,
)

from .budget import BudgetExceeded, BudgetLimits, RunBudget, UsageRecord
from .pilot_contract import (
    ProviderRequestProfile,
    canonical_sha256,
    load_pilot_contract,
)


PROVIDER_INTERFACE_PROBE_SCHEMA_VERSION = "finevo-provider-interface-probe-v1"
PROVIDER_DIAGNOSTIC_BUDGET_SCHEMA_VERSION = (
    "finevo-provider-diagnostic-budget-v1"
)
DIAGNOSTIC_CUMULATIVE_CAP_USD = 0.30
PRIOR_MANUAL_DIAGNOSTIC_RESERVE_USD = 0.01
DEFAULT_INTERFACE_PROBE_MAX_TOKENS = 128
DIAGNOSTIC_OUTPUT_RELATIVE_ROOT = (
    Path("experiment_results")
    / "pilot-v2-debug"
    / "raw"
    / "interface"
)
_PINNED_SDK_VERSIONS = {
    "openai": (
        "openai-python",
        PINNED_PROVIDER_SDK_VERSIONS["openai"],
    ),
    "openrouter": (
        "openai-python",
        PINNED_PROVIDER_SDK_VERSIONS["openai"],
    ),
    "ollama": (
        "requests",
        PINNED_PROVIDER_SDK_VERSIONS["requests"],
    ),
}
_PROMPT = "Return exactly one JSON object: {\"ok\": true}. No other text."
_ENDPOINT_IDENTITIES = {
    "openai": "openai-official-v1",
    "openrouter": "openrouter-official-v1",
    "ollama": "ollama-loopback-v1",
}


class ProviderDiagnosticError(RuntimeError):
    """Raised before dispatch when a diagnostic probe is not provenance-safe."""


def _safe_error_type(exc: BaseException) -> str:
    candidate = type(exc).__name__
    return candidate if candidate in SAFE_PROVIDER_ERROR_TYPES else "ProviderError"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProviderDiagnosticError(
            result.stderr.strip() or result.stdout.strip() or "git command failed"
        )
    return result.stdout.strip()


def verify_debug_provenance(
    repo_root: str | Path,
    *,
    required_tag: str,
) -> dict[str, Any]:
    """Require a clean HEAD at one annotated diagnostic tag."""

    root = Path(repo_root).resolve()
    if not required_tag or required_tag.strip() != required_tag:
        raise ProviderDiagnosticError("required_tag must be a non-empty exact tag")
    tag_ref = f"refs/tags/{required_tag}"
    tag_type = _git(root, "cat-file", "-t", tag_ref)
    if tag_type != "tag":
        raise ProviderDiagnosticError(
            f"{required_tag!r} must be an annotated tag, observed {tag_type!r}"
        )
    head = _git(root, "rev-parse", "HEAD")
    peeled = _git(root, "rev-parse", f"{tag_ref}^{{commit}}")
    if head != peeled:
        raise ProviderDiagnosticError(
            "provider probe requires HEAD to equal the peeled diagnostic tag"
        )
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ProviderDiagnosticError("provider probe requires a clean worktree")
    return {
        "git_tag": required_tag,
        "head_commit": head,
        "tag_commit": peeled,
        "tag_object_id": _git(root, "rev-parse", tag_ref),
        "tag_object_type": tag_type,
        "worktree_clean": True,
    }


def _assert_diagnostic_output_path(repo_root: Path, output_path: Path) -> None:
    resolved = output_path.resolve()
    allowed_root = (repo_root / DIAGNOSTIC_OUTPUT_RELATIVE_ROOT).resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ProviderDiagnosticError(
            "diagnostic receipt must be inside "
            "experiment_results/pilot-v2-debug/raw/interface"
        ) from exc
    if resolved.suffix != ".json":
        raise ProviderDiagnosticError("diagnostic receipt must use a .json suffix")
    internal_paths = {
        _ledger_path(repo_root).resolve(),
        _ledger_lock_path(repo_root).resolve(),
    }
    if resolved in internal_paths:
        raise ProviderDiagnosticError(
            "diagnostic receipt path collides with an internal ledger path"
        )
    if resolved.exists():
        raise ProviderDiagnosticError(
            "diagnostic receipt already exists and cannot be overwritten"
        )
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(resolved)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProviderDiagnosticError(
            "diagnostic receipt path must be ignored by git"
        )


def _ledger_path(repo_root: Path) -> Path:
    return (
        repo_root
        / DIAGNOSTIC_OUTPUT_RELATIVE_ROOT
        / "diagnostic_budget_ledger.json"
    )


def _ledger_lock_path(repo_root: Path) -> Path:
    return (
        repo_root
        / DIAGNOSTIC_OUTPUT_RELATIVE_ROOT
        / ".diagnostic-budget.lock"
    )


@contextmanager
def _diagnostic_ledger_lock(repo_root: Path) -> Iterable[None]:
    lock_path = _ledger_lock_path(repo_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _new_diagnostic_ledger() -> dict[str, Any]:
    return {
        "schema_version": PROVIDER_DIAGNOSTIC_BUDGET_SCHEMA_VERSION,
        "scientific_evidence": False,
        "diagnostic_only": True,
        "hard_cap_usd": DIAGNOSTIC_CUMULATIVE_CAP_USD,
        "prior_manual_diagnostic_reserve_usd": (
            PRIOR_MANUAL_DIAGNOSTIC_RESERVE_USD
        ),
        "prior_manual_probe_note": (
            "Conservative reserve for pre-runner GPT-5.2, GPT-5.6, and "
            "OpenRouter request-shape diagnostics."
        ),
        "entries": [],
        "updated_at": _utc_now(),
    }


def _read_diagnostic_ledger(repo_root: Path) -> dict[str, Any]:
    path = _ledger_path(repo_root)
    if not path.exists():
        return _new_diagnostic_ledger()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderDiagnosticError(
            "diagnostic budget ledger is unreadable"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version")
        != PROVIDER_DIAGNOSTIC_BUDGET_SCHEMA_VERSION
        or not isinstance(value.get("entries"), list)
        or value.get("scientific_evidence") is not False
        or value.get("diagnostic_only") is not True
        or value.get("hard_cap_usd") != DIAGNOSTIC_CUMULATIVE_CAP_USD
        or value.get("prior_manual_diagnostic_reserve_usd")
        != PRIOR_MANUAL_DIAGNOSTIC_RESERVE_USD
    ):
        raise ProviderDiagnosticError(
            "diagnostic budget ledger does not match the frozen schema"
        )
    expected_hash = value.get("ledger_sha256")
    copied = dict(value)
    copied.pop("ledger_sha256", None)
    if expected_hash != canonical_sha256(copied):
        raise ProviderDiagnosticError(
            "diagnostic budget ledger hash mismatch"
        )
    return value


def _write_diagnostic_ledger(repo_root: Path, value: Mapping[str, Any]) -> None:
    payload = json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    payload["updated_at"] = _utc_now()
    payload.pop("ledger_sha256", None)
    payload["ledger_sha256"] = canonical_sha256(payload)
    _atomic_json(_ledger_path(repo_root), payload)


def _ledger_accounted_cost(value: Mapping[str, Any]) -> float:
    total = float(value["prior_manual_diagnostic_reserve_usd"])
    for entry in value["entries"]:
        if not isinstance(entry, Mapping):
            raise ProviderDiagnosticError(
                "diagnostic budget ledger entry is invalid"
            )
        if entry.get("status") in {"reserved", "dispatching"}:
            total += float(entry["reserved_cost_usd"])
        else:
            total += float(entry["accounted_cost_usd"])
    return total


def _reserve_cumulative_diagnostic_budget(
    repo_root: Path,
    *,
    required_tag: str,
    model_id: str,
    profile_id: str,
    output_path: Path,
    reserved_cost_usd: float,
    force_json_object: bool,
) -> dict[str, Any]:
    with _diagnostic_ledger_lock(repo_root):
        ledger = _read_diagnostic_ledger(repo_root)
        duplicate = [
            entry
            for entry in ledger["entries"]
            if entry.get("git_tag") == required_tag
            and entry.get("model_id") == model_id
            and entry.get("force_json_object") is force_json_object
        ]
        if duplicate:
            raise ProviderDiagnosticError(
                "this tag/model/override diagnostic cell is already registered"
            )
        if any(
            entry.get("output_path") == str(output_path)
            for entry in ledger["entries"]
        ):
            raise ProviderDiagnosticError(
                "this diagnostic output path is already registered"
            )
        before = _ledger_accounted_cost(ledger)
        if (
            before + float(reserved_cost_usd)
            > DIAGNOSTIC_CUMULATIVE_CAP_USD + 1e-12
        ):
            raise ProviderDiagnosticError(
                "cumulative provider diagnostic budget is exhausted"
            )
        created_at = _utc_now()
        reservation_id = canonical_sha256(
            {
                "git_tag": required_tag,
                "model_id": model_id,
                "profile_id": profile_id,
                "output_path": str(output_path),
                "created_at": created_at,
            }
        )[:24]
        entry = {
            "reservation_id": reservation_id,
            "git_tag": required_tag,
            "model_id": model_id,
            "profile_id": profile_id,
            "force_json_object": force_json_object,
            "output_path": str(output_path),
            "status": "reserved",
            "reserved_cost_usd": float(reserved_cost_usd),
            "accounted_cost_usd": None,
            "created_at": created_at,
            "dispatch_started_at": None,
            "completed_at": None,
            "receipt_sha256": None,
        }
        ledger["entries"].append(entry)
        _write_diagnostic_ledger(repo_root, ledger)
        return {
            "reservation_id": reservation_id,
            "hard_cap_usd": DIAGNOSTIC_CUMULATIVE_CAP_USD,
            "accounted_before_usd": before,
            "reserved_cost_usd": float(reserved_cost_usd),
            "prior_manual_diagnostic_reserve_usd": (
                PRIOR_MANUAL_DIAGNOSTIC_RESERVE_USD
            ),
        }


def _mark_cumulative_diagnostic_dispatching(
    repo_root: Path,
    *,
    reservation_id: str,
) -> None:
    """Durably distinguish a possibly dispatched call from a reservation."""

    with _diagnostic_ledger_lock(repo_root):
        ledger = _read_diagnostic_ledger(repo_root)
        matches = [
            entry
            for entry in ledger["entries"]
            if entry.get("reservation_id") == reservation_id
        ]
        if len(matches) != 1 or matches[0].get("status") != "reserved":
            raise ProviderDiagnosticError(
                "diagnostic budget reservation cannot enter dispatching state"
            )
        entry = matches[0]
        entry["status"] = "dispatching"
        entry["dispatch_started_at"] = _utc_now()
        _write_diagnostic_ledger(repo_root, ledger)


def _finalize_cumulative_diagnostic_budget(
    repo_root: Path,
    *,
    reservation_id: str,
    status: str,
    accounted_cost_usd: float,
    receipt_sha256: str,
) -> dict[str, Any]:
    with _diagnostic_ledger_lock(repo_root):
        ledger = _read_diagnostic_ledger(repo_root)
        matches = [
            entry
            for entry in ledger["entries"]
            if entry.get("reservation_id") == reservation_id
        ]
        if (
            len(matches) != 1
            or matches[0].get("status") not in {"reserved", "dispatching"}
        ):
            raise ProviderDiagnosticError(
                "diagnostic budget reservation is missing or already finalized"
            )
        entry = matches[0]
        entry["status"] = status
        entry["accounted_cost_usd"] = float(accounted_cost_usd)
        entry["completed_at"] = _utc_now()
        entry["receipt_sha256"] = receipt_sha256
        _write_diagnostic_ledger(repo_root, ledger)
        return {
            "reservation_id": reservation_id,
            "status": status,
            "accounted_cost_usd": float(accounted_cost_usd),
            "cumulative_accounted_usd": _ledger_accounted_cost(ledger),
            "hard_cap_usd": DIAGNOSTIC_CUMULATIVE_CAP_USD,
        }


def _cumulative_reservation_cost(
    profile: ProviderRequestProfile,
    *,
    max_cost_usd: float,
) -> float:
    """Reserve the caller cap for hosted probes and no dollars for local ones."""

    return 0.0 if profile.transport == "ollama" else float(max_cost_usd)


def _effective_profile(
    source: ProviderRequestProfile,
    *,
    force_json_object: bool,
) -> tuple[ProviderRequestProfile, dict[str, Any]]:
    if not force_json_object:
        return source, {}
    payload = source.to_dict()
    payload["profile_id"] = f"{source.profile_id}-diagnostic-json"
    payload["json_mode"] = "json_object"
    return ProviderRequestProfile.from_dict(payload), {
        "json_mode": {
            "source": source.json_mode,
            "effective": "json_object",
            "reason": "diagnostic-only interface compatibility probe",
        }
    }


def _provider_for_profile(profile: ProviderRequestProfile):
    provider_type = {
        "openai": "openai",
        "openrouter": "thirdparty",
        "ollama": "ollama",
    }.get(profile.transport)
    if provider_type is None:
        raise ProviderDiagnosticError(
            f"unsupported diagnostic transport: {profile.transport}"
        )
    return create_llm_provider(
        provider_type,
        model=profile.requested_model,
        max_retries=1,
        request_profile=profile,
    )


def _estimated_usage(
    profile: ProviderRequestProfile,
    *,
    max_tokens: int,
) -> UsageRecord:
    rates = profile.price_snapshot.costs_per_1k()
    prompt_tokens = max(1, math.ceil(len(_PROMPT) / 4))
    return UsageRecord(
        prompt_tokens=prompt_tokens,
        completion_tokens=max_tokens,
        cost_usd=(
            prompt_tokens * rates["prompt"]
            + max_tokens * rates["completion"]
        )
        / 1000.0,
    )


def _strict_probe_json(text: str) -> bool:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(value, Mapping)
        and set(value) == {"ok"}
        and value["ok"] is True
    )


def _expected_request_parameters(
    profile: ProviderRequestProfile,
) -> tuple[str, ...]:
    if profile.transport == "openai":
        names = {"messages", "model", "top_p"}
        names.add(
            "max_completion_tokens"
            if profile.requested_model.startswith(("gpt-5", "o1", "o3"))
            else "max_tokens"
        )
        if not profile.requested_model.startswith(("gpt-5", "o1", "o3")):
            names.add("temperature")
        if profile.seed_capability != "unsupported":
            names.add("seed")
        if profile.json_mode == "json_object":
            names.add("response_format")
        if profile.reasoning.mode == "fixed":
            names.add("reasoning_effort")
        return tuple(sorted(names))
    if profile.transport == "openrouter":
        names = {
            "extra_body",
            "max_tokens",
            "messages",
            "model",
            "response_format",
            "temperature",
            "top_p",
        }
        if profile.seed_capability != "unsupported":
            names.add("seed")
        return tuple(sorted(names))
    if profile.transport == "ollama":
        names = {"messages", "model", "options", "stream"}
        if profile.json_mode == "json_object":
            names.add("format")
        return tuple(sorted(names))
    raise ProviderDiagnosticError(
        f"unsupported diagnostic transport: {profile.transport}"
    )


def _interface_checks(
    completion: StructuredCompletion,
    profile: ProviderRequestProfile,
    *,
    strict_json_valid: bool,
) -> dict[str, bool]:
    expected_sdk_name, expected_sdk_version = _PINNED_SDK_VERSIONS[
        profile.transport
    ]
    checks = {
        "provider_success": completion.ok and completion.text != "Error",
        "strict_json_valid": strict_json_valid,
        "served_model_exact": completion.response_model == profile.served_model,
        "request_profile_exact": completion.request_profile_id == profile.profile_id,
        "single_attempt": completion.attempts == 1,
        "finish_stop": completion.finish_reason == "stop",
        "response_completed": completion.response_completed is True,
        "sdk_name_exact": completion.provider_sdk_name == expected_sdk_name,
        "sdk_version_exact": (
            completion.provider_sdk_version == expected_sdk_version
        ),
        "error_details_empty": completion.provider_error_details is None,
        "request_parameters_exact": (
            completion.request_parameters
            == _expected_request_parameters(profile)
        ),
    }
    if profile.transport == "openrouter":
        checks.update(
            {
                "route_attestation_pass": (
                    completion.route_attestation_code == "OR_RA_PASS"
                ),
                "provider_pin_exact": (
                    completion.response_provider in set(profile.provider_pin)
                ),
                "route_snapshot_exact": (
                    completion.response_route
                    == dict(profile.artifact_identity).get("served_snapshot")
                ),
                "temperature_dispatch_exact": (
                    completion.temperature_dispatch == "explicit"
                ),
            }
        )
    elif profile.transport == "openai":
        checks.update(
            {
                "provider_pin_exact": (
                    completion.response_provider == "OpenAI-direct"
                ),
                "route_snapshot_exact": completion.response_route == "direct",
                "temperature_dispatch_exact": (
                    completion.temperature_dispatch
                    == (
                        "omitted_unsupported"
                        if profile.requested_model.startswith(
                            ("gpt-5", "o1", "o3")
                        )
                        else "explicit"
                    )
                ),
            }
        )
    else:
        checks.update(
            {
                "provider_pin_exact": (
                    completion.response_provider == "local-ollama"
                ),
                "route_snapshot_exact": completion.response_route == "local",
                "json_format_dispatched": "format"
                in completion.request_parameters,
                "temperature_dispatch_exact": (
                    completion.temperature_dispatch == "explicit"
                ),
            }
        )
    return checks


def _observed_sdk_identity(
    profile: ProviderRequestProfile,
) -> tuple[str, str | None]:
    sdk_name, _ = _PINNED_SDK_VERSIONS[profile.transport]
    package = "requests" if profile.transport == "ollama" else "openai"
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return sdk_name, version


def _prepare_receipt_destination(path: Path) -> None:
    """Check destination writability before any provider dispatch."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ProviderDiagnosticError(
            "diagnostic receipt already exists and cannot be overwritten"
        )
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".provider-diagnostic-preflight-",
            dir=path.parent,
            delete=True,
        ) as handle:
            handle.write(b"ready\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ProviderDiagnosticError(
            "diagnostic receipt destination is not writable"
        ) from exc


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    refuse_overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise ProviderDiagnosticError(
            "diagnostic receipt already exists and cannot be overwritten"
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        if refuse_overwrite and path.exists():
            raise ProviderDiagnosticError(
                "diagnostic receipt already exists and cannot be overwritten"
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def verify_provider_interface_receipt(value: Mapping[str, Any]) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version")
        != PROVIDER_INTERFACE_PROBE_SCHEMA_VERSION
    ):
        raise ProviderDiagnosticError(
            "provider interface receipt has an unsupported schema"
        )
    expected = value.get("receipt_sha256")
    copied = dict(value)
    copied.pop("receipt_sha256", None)
    if expected != canonical_sha256(copied):
        raise ProviderDiagnosticError("provider interface receipt hash mismatch")
    dispatch = value.get("dispatch")
    if not isinstance(dispatch, Mapping):
        raise ProviderDiagnosticError(
            "provider interface receipt lacks dispatch metadata"
        )
    completion = value.get("completion")
    if dispatch.get("provider_call_attempted") is True:
        if (
            dispatch.get("provider_constructed") is not True
            or not isinstance(completion, Mapping)
            or "text" in completion
            or value.get("pre_dispatch_failure") is not None
        ):
            raise ProviderDiagnosticError(
                "provider interface receipt contains unsafe completion content"
            )
    elif (
        dispatch.get("provider_call_attempted") is not False
        or completion is not None
        or not isinstance(value.get("pre_dispatch_failure"), Mapping)
    ):
        raise ProviderDiagnosticError(
            "provider interface receipt has invalid pre-dispatch metadata"
        )
    effective_profile = value.get("effective_profile")
    request = value.get("request")
    if not isinstance(effective_profile, Mapping) or not isinstance(
        request, Mapping
    ):
        raise ProviderDiagnosticError(
            "provider interface receipt lacks request/profile metadata"
        )
    transport = effective_profile.get("transport")
    if request.get("endpoint_identity") != _ENDPOINT_IDENTITIES.get(transport):
        raise ProviderDiagnosticError(
            "provider interface receipt endpoint identity is invalid"
        )
    if (
        value.get("scientific_evidence") is not False
        or value.get("diagnostic_only") is not True
        or value.get("denominator_inclusion") is not False
    ):
        raise ProviderDiagnosticError(
            "provider interface receipt violates its scientific boundary"
        )


def _build_probe_receipt(
    *,
    resolved_contract_path: Path,
    contract_hash: str,
    source_profile: ProviderRequestProfile,
    profile: ProviderRequestProfile,
    overrides: Mapping[str, Any],
    provenance: Mapping[str, Any],
    cumulative_reservation: Mapping[str, Any],
    max_tokens: int,
    max_cost_usd: float,
    seed: int | None,
    estimate: UsageRecord,
    status: str,
    checks: Mapping[str, bool],
    budget_overage_reasons: Iterable[str],
    completion: StructuredCompletion | None,
    budget: RunBudget,
    provider_constructed: bool,
    provider_call_attempted: bool,
    pre_dispatch_failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": PROVIDER_INTERFACE_PROBE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": status,
        "scientific_evidence": False,
        "diagnostic_only": True,
        "denominator_inclusion": False,
        "claim_boundary": (
            "This receipt validates one provider interface shape only and is "
            "not evidence for model capability or FinEvo effectiveness."
        ),
        "source_contract": str(resolved_contract_path),
        "source_contract_sha256": contract_hash,
        "source_profile_id": source_profile.profile_id,
        "source_profile_sha256": canonical_sha256(source_profile.to_dict()),
        "effective_profile": profile.to_dict(),
        "diagnostic_overrides": dict(overrides),
        "provenance": dict(provenance),
        "cumulative_budget_reservation": dict(cumulative_reservation),
        "request": {
            "prompt_sha256": canonical_sha256({"prompt": _PROMPT}),
            "endpoint_identity": _ENDPOINT_IDENTITIES[profile.transport],
            "max_tokens": max_tokens,
            "per_call_cost_cap_usd": float(max_cost_usd),
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "estimated_usage": estimate.to_dict(),
        },
        "dispatch": {
            "provider_constructed": provider_constructed,
            "provider_call_attempted": provider_call_attempted,
        },
        "checks": dict(checks),
        "budget_overage_reasons": list(budget_overage_reasons),
        "pre_dispatch_failure": (
            dict(pre_dispatch_failure)
            if pre_dispatch_failure is not None
            else None
        ),
        "completion": (
            completion.safe_audit_dict()
            if completion is not None
            else None
        ),
        "budget": budget.snapshot().to_dict(),
    }


def _seal_and_write_receipt(
    destination: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    payload["receipt_sha256"] = canonical_sha256(payload)
    verify_provider_interface_receipt(payload)
    _atomic_json(destination, payload, refuse_overwrite=True)
    return payload


def run_provider_interface_probe(
    *,
    contract_path: str | Path,
    model_id: str,
    output_path: str | Path,
    repo_root: str | Path,
    required_tag: str,
    max_tokens: int = DEFAULT_INTERFACE_PROBE_MAX_TOKENS,
    max_cost_usd: float = 0.05,
    force_json_object: bool = False,
    provider_factory: Callable[[ProviderRequestProfile], Any] | None = None,
    provenance_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one budgeted provider call and persist only a safe receipt."""

    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise ProviderDiagnosticError("max_tokens must be a positive integer")
    if (
        isinstance(max_cost_usd, bool)
        or not isinstance(max_cost_usd, (int, float))
        or not 0 < float(max_cost_usd) <= 0.10
    ):
        raise ProviderDiagnosticError(
            "diagnostic max_cost_usd must be positive and at most $0.10"
        )
    root = Path(repo_root).resolve()
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    _assert_diagnostic_output_path(root, destination)
    verify = provenance_verifier or verify_debug_provenance
    provenance = dict(verify(root, required_tag=required_tag))

    resolved_contract_path = Path(contract_path)
    if not resolved_contract_path.is_absolute():
        resolved_contract_path = root / resolved_contract_path
    resolved_contract_path = resolved_contract_path.resolve()
    contract = load_pilot_contract(resolved_contract_path)
    try:
        source_profile = contract.provider_profiles[model_id]
    except KeyError as exc:
        raise ProviderDiagnosticError(f"unknown model_id: {model_id}") from exc
    profile, overrides = _effective_profile(
        source_profile,
        force_json_object=force_json_object,
    )
    estimate = _estimated_usage(profile, max_tokens=max_tokens)
    budget = RunBudget(
        BudgetLimits(
            max_calls=1,
            max_completion_tokens=max_tokens,
            max_cost_usd=float(max_cost_usd),
        ),
        budget_id=f"provider-interface-{model_id}",
    )
    _prepare_receipt_destination(destination)
    cumulative = _reserve_cumulative_diagnostic_budget(
        root,
        required_tag=required_tag,
        model_id=model_id,
        profile_id=profile.profile_id,
        output_path=destination,
        reserved_cost_usd=_cumulative_reservation_cost(
            profile,
            max_cost_usd=float(max_cost_usd),
        ),
        force_json_object=force_json_object,
    )
    seed = 2010922376 if profile.seed_capability != "unsupported" else None

    provider = None
    provider_constructed = False
    reservation = None
    pre_dispatch_error: Exception | None = None
    pre_dispatch_stage: str | None = None
    if profile.transport == "ollama" and profile.json_mode != "json_object":
        pre_dispatch_error = ProviderDiagnosticError(
            "Ollama interface probes require explicit JSON mode"
        )
        pre_dispatch_stage = "diagnostic.profile.validate"
    elif estimate.cost_usd > float(max_cost_usd):
        pre_dispatch_error = ProviderDiagnosticError(
            "frozen price estimate exceeds the diagnostic cost cap"
        )
        pre_dispatch_stage = "diagnostic.budget.estimate"
    else:
        try:
            provider = (provider_factory or _provider_for_profile)(profile)
            provider_constructed = True
        except Exception as exc:
            pre_dispatch_error = exc
            pre_dispatch_stage = "diagnostic.provider.construct"

    if pre_dispatch_error is None:
        try:
            reservation = budget.reserve_call(
                estimated_usage=estimate,
                label=f"diagnostic-interface:{model_id}",
                model=f"{profile.transport}/{profile.requested_model}",
                tags={
                    "scientific_evidence": False,
                    "diagnostic_only": True,
                    "model_id": model_id,
                },
            )
        except Exception as exc:
            pre_dispatch_error = exc
            pre_dispatch_stage = "diagnostic.call_budget.reserve"
    if pre_dispatch_error is None:
        try:
            _mark_cumulative_diagnostic_dispatching(
                root,
                reservation_id=cumulative["reservation_id"],
            )
        except Exception as exc:
            pre_dispatch_error = exc
            pre_dispatch_stage = "diagnostic.cumulative_budget.dispatch"

    if pre_dispatch_error is not None:
        sdk_name, sdk_version = _observed_sdk_identity(profile)
        failure = ProviderErrorDetails(
            error_type=_safe_error_type(pre_dispatch_error),
            stage=pre_dispatch_stage or "diagnostic.pre_dispatch",
            sdk_name=sdk_name,
            sdk_version=sdk_version,
        ).to_dict()
        payload = _build_probe_receipt(
            resolved_contract_path=resolved_contract_path,
            contract_hash=contract.canonical_hash,
            source_profile=source_profile,
            profile=profile,
            overrides=overrides,
            provenance=provenance,
            cumulative_reservation=cumulative,
            max_tokens=max_tokens,
            max_cost_usd=float(max_cost_usd),
            seed=seed,
            estimate=estimate,
            status="no-go",
            checks={
                "provider_constructed": provider_constructed,
                "provider_call_attempted": False,
            },
            budget_overage_reasons=(),
            completion=None,
            budget=budget,
            provider_constructed=provider_constructed,
            provider_call_attempted=False,
            pre_dispatch_failure=failure,
        )
        _seal_and_write_receipt(destination, payload)
        _finalize_cumulative_diagnostic_budget(
            root,
            reservation_id=cumulative["reservation_id"],
            status="pre-dispatch-no-go",
            accounted_cost_usd=0.0,
            receipt_sha256=payload["receipt_sha256"],
        )
        return payload

    if provider is None or reservation is None:  # pragma: no cover - guarded above
        raise AssertionError("provider dispatch reached an impossible state")
    try:
        completion = provider.get_structured_completion(
            [{"role": "user", "content": _PROMPT}],
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
            max_retries=1,
            seed=seed,
        )
        if not isinstance(completion, StructuredCompletion):
            raise TypeError(
                "provider structured API must return StructuredCompletion"
            )
    except Exception as exc:
        sdk_name, sdk_version = _observed_sdk_identity(profile)
        completion = StructuredCompletion(
            text="Error",
            usage=estimate,
            model=profile.requested_model,
            provider=profile.transport,
            attempts=1,
            latency_seconds=0.0,
            error_type=_safe_error_type(exc),
            request_seed=seed,
            provider_error_details=ProviderErrorDetails(
                error_type=_safe_error_type(exc),
                stage="diagnostic.provider.dispatch",
                sdk_name=sdk_name,
                sdk_version=sdk_version,
            ),
            provider_sdk_name=sdk_name,
            provider_sdk_version=sdk_version,
            output_disposition="unavailable_due_to_provider_error",
        )
    if completion.error_type is not None:
        conservative = UsageRecord(
            prompt_tokens=max(
                completion.usage.prompt_tokens,
                estimate.prompt_tokens,
            ),
            completion_tokens=max(
                completion.usage.completion_tokens,
                estimate.completion_tokens,
            ),
            cost_usd=max(
                float(completion.usage.cost_usd),
                float(estimate.cost_usd),
            ),
        )
        if conservative != completion.usage:
            completion = replace(completion, usage=conservative)
    budget_overage_reasons: list[str] = []
    try:
        budget.complete_call(reservation, completion.usage)
    except BudgetExceeded as exc:
        budget_overage_reasons = [reason.value for reason in exc.reasons]
    strict_json_valid = _strict_probe_json(completion.text)
    checks = _interface_checks(
        completion,
        profile,
        strict_json_valid=strict_json_valid,
    )
    status = (
        "pass"
        if all(checks.values()) and not budget_overage_reasons
        else "no-go"
    )
    payload = _build_probe_receipt(
        resolved_contract_path=resolved_contract_path,
        contract_hash=contract.canonical_hash,
        source_profile=source_profile,
        profile=profile,
        overrides=overrides,
        provenance=provenance,
        cumulative_reservation=cumulative,
        max_tokens=max_tokens,
        max_cost_usd=float(max_cost_usd),
        seed=seed,
        estimate=estimate,
        status=status,
        checks=checks,
        budget_overage_reasons=budget_overage_reasons,
        completion=completion,
        budget=budget,
        provider_constructed=True,
        provider_call_attempted=True,
        pre_dispatch_failure=None,
    )
    _seal_and_write_receipt(destination, payload)
    _finalize_cumulative_diagnostic_budget(
        root,
        reservation_id=cumulative["reservation_id"],
        status=status,
        accounted_cost_usd=float(
            budget.snapshot().accounted_usage.cost_usd
        ),
        receipt_sha256=payload["receipt_sha256"],
    )
    return payload


__all__ = [
    "PROVIDER_INTERFACE_PROBE_SCHEMA_VERSION",
    "PROVIDER_DIAGNOSTIC_BUDGET_SCHEMA_VERSION",
    "DEFAULT_INTERFACE_PROBE_MAX_TOKENS",
    "ProviderDiagnosticError",
    "run_provider_interface_probe",
    "verify_provider_interface_receipt",
    "verify_debug_provenance",
]
