"""Provider-free V2.11.11 fresh-cohort authority and acceptance controls.

V2.11.11 never resumes a V2.11.10 cell.  It binds the immutable V2.11.5
scientific authority and V2.11.10 terminal no-go, imports only their cumulative
budget debit, and registers 86 new scientific run identities on five new seeds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

from .pilot_budget import ParentBudgetDebit
from .pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_11,
    PILOT_CONTRACT_TAG_V2_11_11,
    PilotContract,
    PilotRunSpec,
    canonical_sha256,
    load_pilot_contract,
)
from .pilot_v2115_parent_import import (
    PilotV2115ParentImportError,
    V2115_ALLOWED_MODELS,
    capability_wrappers_from_v2115_receipt,
    calibration_wrapper_from_v2115_receipt,
    preflight_wrappers_from_v2115_receipt,
    validate_v2115_parent_import_receipt,
)


V21111_CONTRACT_ID = PILOT_CONTRACT_ID_V2_11_11
V21111_SCIENCE_TAG = PILOT_CONTRACT_TAG_V2_11_11
V21111_RAW_NAMESPACE = "experiment_results/pilot-v2.11.11/raw"
V21111_DIAGNOSTICS_NAMESPACE = "experiment_results/pilot-v2.11.11/diagnostics"
V21111_PARENT_IMPORT_SCHEMA = "finevo-pilot-v2.11.11-parent-import-v1"
V21111_ACCEPTANCE_SCHEMA = "finevo-pilot-v2.11.11-scientific-dispatch-acceptance-v1"
V21111_D_COORDINATOR_SCHEMA = "finevo-pilot-v2.11.11-d-coordinator-v1"
V21111_PARENT_RECEIPT_FILENAME = "parent_import_receipt.json"
V21111_AUTHORITY_MODELS = ("gpt52_main", "gpt56_diagnostic")
V21111_V2115_PARENT_RECEIPT_PATH = (
    "experiment_results/pilot-v2.11.5/raw/parent-import/" "parent_import_receipt.json"
)
V21111_PARENT_CLAIM_BOUNDARY = (
    "Immutable lineage, capability/interface, calibration, preflight, and budget "
    "authority only; no historical effect cell or decoded completion is imported."
)
V21111_PARENT_COST_QUANTUM_USD = Decimal("0.00000001")
V21111_PARENT_COST_MAX_ROUNDING_RESIDUAL_USD = Decimal("0.000000005")
V21111_ACCEPTANCE_FILENAME = "scientific_dispatch_acceptance.json"
V21111_FULL_FAKE_DIRECTORY = "provider-free-full-fake-acceptance"
V21111_FULL_FAKE_RECEIPT_FILENAME = "acceptance_receipt.json"
V21111_FULL_FAKE_SCHEMA = "finevo-pilot-v2.11.11-full-fake-acceptance-v1"
V21111_FAULT_ACCEPTANCE_CHECKS = frozenset(
    {
        "branch_failure_is_single_cell",
        "external_provider_calls_zero",
        "interruption_replay_no_redispatch",
        "interruption_stops_only_reserved_branch",
        "prefix_failure_does_not_block_b_or_cross",
        "prefix_failure_marks_exact_11_cells",
        "prefix_interruption_conservatively_stops_all_11",
        "terminal_commit_window_repairs_without_redispatch",
    }
)

FRESH_MAIN_SEEDS = (
    877_361,
    1_410_637_959,
    416_755_402,
    357_136_200,
    1_541_219_789,
)
OLD_MAIN_SEEDS = (
    1_099_057_501,
    1_421_875_452,
    1_769_977_770,
    959_809_858,
    617_806_385,
)
SCIENTIFIC_STAGE_IDS = ("experiment-b", "experiment-d", "cross-model")
OPERATIONAL_STAGE_IDS = ("parent-import",)
D_BRANCH_IDS = (
    "matched-a",
    "matched-b",
    "no-memory",
    "shuffled-episodic",
    "wrong-context",
    "error-verified",
    "error-unverified",
    "narrative-none",
    "narrative-aligned",
    "narrative-paraphrase",
    "narrative-opposite",
)
_PROVIDER_KEY_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


class PilotV21111FreshCohortError(RuntimeError):
    """Raised before V2.11.11 is allowed to construct a provider."""


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_copy(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise PilotV21111FreshCohortError(
        f"value of type {type(value).__name__} is not canonical JSON"
    )


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotV21111FreshCohortError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PilotV21111FreshCohortError(
                    f"{name} contains non-finite value {value!r}"
                )
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotV21111FreshCohortError(f"{name} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise PilotV21111FreshCohortError(f"{name} must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_ledger_self_hash(value: Mapping[str, Any], *, name: str) -> str:
    claimed = value.get("ledger_sha256")
    if not _is_sha256(claimed):
        raise PilotV21111FreshCohortError(f"{name} self-hash is malformed")
    unsigned = _json_copy(value)
    unsigned.pop("ledger_sha256", None)
    if canonical_sha256(unsigned) != claimed:
        raise PilotV21111FreshCohortError(f"{name} self-hash drifted")
    return str(claimed)


def _quantized_parent_cost(rows: Mapping[str, Any]) -> Decimal:
    """Sum JSON float rows then bind exactly at the frozen 1e-8 USD quantum."""

    try:
        costs: list[Decimal] = []
        for row in rows.values():
            if not isinstance(row, Mapping) or not isinstance(
                row.get("actual"), Mapping
            ):
                raise TypeError("budget row actual is malformed")
            raw_cost = row["actual"].get("cost_usd")
            if isinstance(raw_cost, bool) or not isinstance(raw_cost, int | float):
                raise TypeError("budget row cost is not numeric")
            cost = Decimal(str(raw_cost))
            if not cost.is_finite() or cost < 0:
                raise ValueError("budget row cost is not finite and nonnegative")
            costs.append(cost)
        raw_total = sum(costs, Decimal("0"))
        quantized = raw_total.quantize(
            V21111_PARENT_COST_QUANTUM_USD,
            rounding=ROUND_HALF_EVEN,
        )
    except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
        raise PilotV21111FreshCohortError(
            "V2.11.10 current actual cost rows are malformed"
        ) from exc
    if (
        raw_total < 0
        or abs(raw_total - quantized) > V21111_PARENT_COST_MAX_ROUNDING_RESIDUAL_USD
    ):
        raise PilotV21111FreshCohortError(
            "V2.11.10 current actual cost exceeds the frozen quantization bound"
        )
    return quantized


def _verified_parent_current_actual(
    rows: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if set(expected) != {"cost_usd", "hosted_completions", "storage_bytes"}:
        raise PilotV21111FreshCohortError(
            "V2.11.10 frozen current-actual shape drifted"
        )
    try:
        expected_cost = Decimal(str(expected["cost_usd"]))
        expected_completions = expected["hosted_completions"]
        expected_storage = expected["storage_bytes"]
        if (
            expected_cost
            != expected_cost.quantize(
                V21111_PARENT_COST_QUANTUM_USD,
                rounding=ROUND_HALF_EVEN,
            )
            or isinstance(expected_completions, bool)
            or not isinstance(expected_completions, int)
            or isinstance(expected_storage, bool)
            or not isinstance(expected_storage, int)
        ):
            raise ValueError("frozen current actual types drifted")
        observed_completions = 0
        observed_storage = 0
        for row in rows.values():
            if not isinstance(row, Mapping) or not isinstance(
                row.get("actual"), Mapping
            ):
                raise TypeError("budget row actual is malformed")
            completions = row["actual"].get("completions")
            storage = row["actual"].get("storage_bytes")
            if (
                isinstance(completions, bool)
                or not isinstance(completions, int)
                or completions < 0
                or isinstance(storage, bool)
                or not isinstance(storage, int)
                or storage < 0
            ):
                raise TypeError("budget row integer actuals are malformed")
            observed_completions += completions
            observed_storage += storage
    except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
        raise PilotV21111FreshCohortError(
            "V2.11.10 current actual rows are malformed"
        ) from exc
    actual = {
        "cost_usd": _quantized_parent_cost(rows),
        "hosted_completions": observed_completions,
        "storage_bytes": observed_storage,
    }
    if (
        actual["cost_usd"] != expected_cost
        or actual["hosted_completions"] != expected_completions
        or actual["storage_bytes"] != expected_storage
    ):
        raise PilotV21111FreshCohortError("V2.11.10 current actual debit drifted")
    return actual


def _real_directory(value: str | Path, *, name: str) -> Path:
    lexical = Path(value).absolute()
    try:
        mode = lexical.lstat().st_mode
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise PilotV21111FreshCohortError(f"{name} is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise PilotV21111FreshCohortError(
            f"{name} must be a real non-symlink directory"
        )
    return resolved


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PilotV21111FreshCohortError(
            f"git identity check failed for {root.name}"
        ) from exc
    return result.stdout.strip()


def _verify_release_git(
    root: Path,
    *,
    tag: str,
    tag_object: str,
    commit: str,
    name: str,
) -> dict[str, Any]:
    if _git(root, "cat-file", "-t", f"refs/tags/{tag}") != "tag":
        raise PilotV21111FreshCohortError(f"{name} tag is not annotated")
    if _git(root, "rev-parse", f"refs/tags/{tag}") != tag_object:
        raise PilotV21111FreshCohortError(f"{name} tag object drifted")
    if _git(root, "rev-parse", f"refs/tags/{tag}^{{commit}}") != commit:
        raise PilotV21111FreshCohortError(f"{name} peeled commit drifted")
    if _git(root, "rev-parse", "HEAD") != commit:
        raise PilotV21111FreshCohortError(f"{name} checkout is not at the science tag")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise PilotV21111FreshCohortError(f"{name} checkout is dirty")
    return {
        "tag": tag,
        "tag_object": tag_object,
        "commit": commit,
        "clean": True,
    }


def _raw_inventory(raw: Path, *, excluded: Sequence[str]) -> dict[str, Any]:
    if raw.is_symlink() or not raw.is_dir():
        raise PilotV21111FreshCohortError("bound raw namespace is unavailable")
    excluded_set = set(excluded)
    rows: list[dict[str, Any]] = []
    for path in sorted(raw.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise PilotV21111FreshCohortError("bound raw namespace contains a symlink")
        if path.is_file():
            relative = path.relative_to(raw).as_posix()
            if relative in excluded_set:
                continue
            rows.append(
                {
                    "path": relative,
                    "byte_size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
        elif not path.is_dir():
            raise PilotV21111FreshCohortError(
                "bound raw namespace contains a non-regular entry"
            )
    return {
        "file_count": len(rows),
        "storage_bytes": sum(int(row["byte_size"]) for row in rows),
        "inventory_sha256": canonical_sha256(rows),
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(payload)
    result["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(result),
    }
    return result


def _verify_seal(payload: Mapping[str, Any], *, name: str) -> None:
    integrity = payload.get("integrity")
    if not isinstance(integrity, Mapping):
        raise PilotV21111FreshCohortError(f"{name} lacks integrity")
    content = _json_copy(payload)
    content.pop("integrity", None)
    if integrity.get("canonicalization") != "json-sort-keys-utf8-v1" or integrity.get(
        "content_sha256"
    ) != canonical_sha256(content):
        raise PilotV21111FreshCohortError(f"{name} integrity drifted")


def _atomic_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PilotV21111FreshCohortError(f"refusing to overwrite {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(
                (
                    json.dumps(
                        payload,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise PilotV21111FreshCohortError(
            f"concurrent writer already created {path.name}"
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def require_exact_raw_namespace(
    *,
    contract_path: str | Path,
    raw_root: str | Path,
) -> Path:
    """Bind V2.11.11 writes to the new, lexical, non-symlink raw root."""

    contract_file = Path(contract_path).absolute()
    repository = contract_file.parent.parent
    if contract_file != repository / "experiments" / "pilot_v2_11_11.yaml":
        raise PilotV21111FreshCohortError(
            "V2.11.11 requires experiments/pilot_v2_11_11.yaml"
        )
    if not contract_file.is_file() or contract_file.is_symlink():
        raise PilotV21111FreshCohortError(
            "V2.11.11 contract must be a real non-symlink file"
        )
    try:
        if repository.resolve(strict=True) != repository:
            raise PilotV21111FreshCohortError(
                "V2.11.11 repository path contains a symlink"
            )
    except OSError as exc:
        raise PilotV21111FreshCohortError("V2.11.11 repository is unavailable") from exc
    expected = repository / V21111_RAW_NAMESPACE
    raw = Path(raw_root).absolute()
    if raw != expected:
        raise PilotV21111FreshCohortError(
            "V2.11.11 requires its exact fresh raw namespace"
        )
    # Walk every existing component below the verified repository.  In
    # particular, reject a fresh-looking pilot-v2.11.11 directory that aliases
    # any historical raw tree through a parent symlink.
    cursor = repository
    for component in Path(V21111_RAW_NAMESPACE).parts:
        cursor = cursor / component
        # ``Path.exists`` is false for a dangling symlink, so lstat-style
        # detection must precede the existence check.
        if cursor.is_symlink():
            raise PilotV21111FreshCohortError(
                "V2.11.11 raw namespace has a symlink parent"
            )
        if cursor.exists() and cursor.resolve(strict=True) != cursor:
            raise PilotV21111FreshCohortError(
                "V2.11.11 raw namespace overlaps a resolved alternate tree"
            )
    return raw


def require_exact_diagnostics_namespace(
    *,
    contract_path: str | Path,
    diagnostics_root: str | Path,
) -> Path:
    """Bind fake-provider artifacts outside the fresh scientific raw tree."""

    contract_file = Path(contract_path).absolute()
    repository = contract_file.parent.parent
    if contract_file != repository / "experiments" / "pilot_v2_11_11.yaml":
        raise PilotV21111FreshCohortError(
            "V2.11.11 diagnostics require experiments/pilot_v2_11_11.yaml"
        )
    if not contract_file.is_file() or contract_file.is_symlink():
        raise PilotV21111FreshCohortError(
            "V2.11.11 contract must be a real non-symlink file"
        )
    expected = repository / V21111_DIAGNOSTICS_NAMESPACE
    diagnostics = Path(diagnostics_root).absolute()
    if diagnostics != expected:
        raise PilotV21111FreshCohortError(
            "V2.11.11 fake acceptance requires its exact diagnostics namespace"
        )
    raw = repository / V21111_RAW_NAMESPACE
    if diagnostics == raw or raw in diagnostics.parents or diagnostics in raw.parents:
        raise PilotV21111FreshCohortError(
            "V2.11.11 fake acceptance must not overlap the scientific raw namespace"
        )
    try:
        if repository.resolve(strict=True) != repository:
            raise PilotV21111FreshCohortError(
                "V2.11.11 repository path contains a symlink"
            )
    except OSError as exc:
        raise PilotV21111FreshCohortError("V2.11.11 repository is unavailable") from exc
    cursor = repository
    for component in Path(V21111_DIAGNOSTICS_NAMESPACE).parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise PilotV21111FreshCohortError(
                "V2.11.11 diagnostics namespace has a symlink component"
            )
        if cursor.exists() and cursor.resolve(strict=True) != cursor:
            raise PilotV21111FreshCohortError(
                "V2.11.11 diagnostics namespace has a symlink component"
            )
    return diagnostics


def require_fresh_raw_before_parent_import(
    *,
    contract_path: str | Path,
    raw_root: str | Path,
) -> Path:
    """Require an absent or exactly empty raw tree before operational import."""

    raw = require_exact_raw_namespace(
        contract_path=contract_path,
        raw_root=raw_root,
    )
    if raw.exists():
        if not raw.is_dir():
            raise PilotV21111FreshCohortError(
                "V2.11.11 raw namespace is not a directory"
            )
        if any(raw.iterdir()):
            raise PilotV21111FreshCohortError(
                "V2.11.11 raw namespace must be empty before parent import"
            )
    return raw


def require_provider_keys_absent() -> None:
    """Fail closed before a provider-free lineage or acceptance operation."""

    if any(os.environ.get(name) for name in _PROVIDER_KEY_ENV_NAMES):
        raise PilotV21111FreshCohortError(
            "provider credentials must not be loaded during provider-free validation"
        )


def stage_partition_from_contract(
    contract: PilotContract,
) -> dict[str, tuple[str, ...]]:
    """Read the stage partition only from contract-declared evidence classes."""

    if any(stage.evidence_class is None for stage in contract.stages):
        raise PilotV21111FreshCohortError(
            "every V2.11.11 stage must declare evidence_class"
        )
    result = {
        "operational": tuple(
            stage.stage_id
            for stage in contract.stages
            if stage.evidence_class == "operational"
        ),
        "scientific": tuple(
            stage.stage_id
            for stage in contract.stages
            if stage.evidence_class == "scientific"
        ),
    }
    if result != {
        "operational": OPERATIONAL_STAGE_IDS,
        "scientific": SCIENTIFIC_STAGE_IDS,
    }:
        raise PilotV21111FreshCohortError(
            "V2.11.11 evidence-class stage partition drifted"
        )
    return result


def parent_budget_debit_for_v21111(contract: PilotContract) -> ParentBudgetDebit:
    if contract.contract_id != V21111_CONTRACT_ID:
        raise PilotV21111FreshCohortError("parent debit requested for another contract")
    boundary = contract.v21111_fresh_cohort_boundary
    if not isinstance(boundary, Mapping):
        raise PilotV21111FreshCohortError("fresh-cohort boundary is absent")
    try:
        debit = ParentBudgetDebit.from_dict(boundary["parent_budget_debit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotV21111FreshCohortError("parent budget debit drifted") from exc
    if debit.to_dict() != boundary["parent_budget_debit"]:
        raise PilotV21111FreshCohortError("parent budget debit round trip drifted")
    return debit


def _v2115_child_release_binding(contract: PilotContract) -> dict[str, str]:
    boundary = contract.v21111_fresh_cohort_boundary
    if not isinstance(boundary, Mapping):
        raise PilotV21111FreshCohortError("fresh-cohort boundary is absent")
    authority = boundary.get("v2115_scientific_authority")
    if not isinstance(authority, Mapping):
        raise PilotV21111FreshCohortError("V2.11.5 authority binding is absent")
    expected = {
        "contract_id": str(authority.get("contract_id")),
        "contract_sha256": str(authority.get("contract_sha256")),
        "git_tag": str(authority.get("science_tag")),
        "resolved_git_commit": str(authority.get("science_commit")),
    }
    if (
        expected["contract_id"] != "finevo-pilot-v2.11.5"
        or not _is_sha256(expected["contract_sha256"])
        or expected["git_tag"] != "pilot-v2.11.5-science"
        or len(expected["resolved_git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in expected["resolved_git_commit"]
        )
    ):
        raise PilotV21111FreshCohortError("V2.11.5 authority release identity drifted")
    return expected


def _authority_model_ids(contract: PilotContract) -> tuple[str, ...]:
    registered = tuple(
        sorted(
            model_id
            for model_id, role in contract.model_roles.items()
            if role.role != "calibration_only" and role.dispatch_eligible
        )
    )
    if tuple(V2115_ALLOWED_MODELS) != V21111_AUTHORITY_MODELS or registered != tuple(
        sorted(V21111_AUTHORITY_MODELS)
    ):
        raise PilotV21111FreshCohortError(
            "V2.11.11 imported capability model denominator drifted"
        )
    return V21111_AUTHORITY_MODELS


def _validate_zero_provider_capability_wrapper(
    contract: PilotContract,
    model_id: str,
    wrapper: Mapping[str, Any],
) -> None:
    capability = wrapper.get("capability")
    if not isinstance(capability, Mapping):
        raise PilotV21111FreshCohortError(
            f"imported {model_id} capability payload is malformed"
        )
    profile = contract.provider_profiles.get(model_id)
    if profile is None:
        raise PilotV21111FreshCohortError(
            f"imported {model_id} capability lacks a provider profile"
        )
    category_totals = capability.get("category_totals")
    expected_categories = {
        "utility-ranking": (12, 10),
        "rule-application": (12, 10),
        "rule-proposal": (6, 5),
    }
    categories_valid = isinstance(category_totals, Mapping) and set(
        category_totals
    ) == set(expected_categories)
    if categories_valid:
        for category, (denominator, required) in expected_categories.items():
            row = category_totals[category]
            if (
                not isinstance(row, Mapping)
                or row.get("denominator") != denominator
                or row.get("registered_total") != denominator
                or not isinstance(row.get("registered_correct"), int)
                or isinstance(row.get("registered_correct"), bool)
                or int(row["registered_correct"]) < required
                or row.get("required") != required
                or row.get("interface_failure_count") != 0
            ):
                categories_valid = False
                break
    checks = capability.get("checks")
    assessment = capability.get("capability_assessment")
    interface_gate = capability.get("interface_gate")
    expected_runtime_model = f"{profile.transport}/{profile.requested_model}"
    if (
        wrapper.get("provider_construction_current_attempt") is not False
        or wrapper.get("provider_calls_current_attempt") != 0
        or wrapper.get("hosted_provider_calls_current_attempt") != 0
        or wrapper.get("current_attempt_usage")
        != {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
        or wrapper.get("imported_effect_cells") != 0
        or wrapper.get("imported_preflight_samples") != 0
        or wrapper.get("scientific_evidence") is not False
        or wrapper.get("evidence_scope") != "preregistered_task_capability_gate"
        or capability.get("model_id") != model_id
        or capability.get("requested_model") != profile.requested_model
        or capability.get("served_model") != profile.served_model
        or capability.get("runtime_model") != expected_runtime_model
        or capability.get("action_sample_count") != 24
        or capability.get("semantic_sample_count") != 6
        or capability.get("historical_source_calls") != 30
        or capability.get("parse_failure_count") != 0
        or capability.get("provider_failure_count") != 0
        or capability.get("truncation_count") != 0
        or capability.get("capability_pass") is not True
        or capability.get("interface_pass") is not True
        or checks
        != {
            "utility-ranking": True,
            "rule-application": True,
            "rule-proposal": True,
        }
        or not isinstance(assessment, Mapping)
        or assessment.get("pass") is not True
        or assessment.get("status") != "pass"
        or assessment.get("checks") != checks
        or not isinstance(interface_gate, Mapping)
        or interface_gate.get("pass") is not True
        or interface_gate.get("failure_count") != 0
        or not _is_sha256(capability.get("taskset_sha256"))
        or not _is_sha256(capability.get("stage_receipt_content_sha256"))
        or not categories_valid
    ):
        raise PilotV21111FreshCohortError(
            f"imported {model_id} capability authority drifted"
        )


def _validate_imported_authority_wrappers(
    contract: PilotContract,
    *,
    calibration: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    preflights: Mapping[str, Any],
) -> dict[str, Any]:
    model_ids = _authority_model_ids(contract)
    if set(capabilities) != set(model_ids) or set(preflights) != set(model_ids):
        raise PilotV21111FreshCohortError(
            "V2.11.11 imported wrapper denominator drifted"
        )
    expected_release = _v2115_child_release_binding(contract)
    common_lineage = {
        key: calibration.get(key)
        for key in ("child_release", "parent_release", "source_manifest")
    }
    source_manifest = common_lineage["source_manifest"]
    if (
        common_lineage["child_release"] != expected_release
        or not isinstance(common_lineage["parent_release"], Mapping)
        or not isinstance(source_manifest, Mapping)
        or source_manifest.get("path")
        != "experiments/pilot_v2_11_5_source_manifest.json"
        or not _is_sha256(source_manifest.get("file_sha256"))
        or not _is_sha256(source_manifest.get("content_sha256"))
        or calibration.get("provider_construction_during_import") is not False
        or calibration.get("provider_calls_during_import") != 0
        or calibration.get("hosted_provider_calls_during_import") != 0
        or calibration.get("hosted_cost_usd_during_import") != 0.0
        or calibration.get("imported_effect_cells") != 0
        or calibration.get("imported_scientific_run_summaries") != 0
        or calibration.get("imported_scientific_outcome_artifacts") != []
        or calibration.get("decoded_completion_reuse") is not False
        or calibration.get("scientific_evidence") is not False
    ):
        raise PilotV21111FreshCohortError(
            "V2.11.11 imported calibration authority drifted"
        )
    for model_id in model_ids:
        capability = capabilities[model_id]
        preflight = preflights[model_id]
        if not isinstance(capability, Mapping) or not isinstance(preflight, Mapping):
            raise PilotV21111FreshCohortError(
                f"imported {model_id} authority wrapper is malformed"
            )
        if any(
            wrapper.get(key) != value
            for wrapper in (capability, preflight)
            for key, value in common_lineage.items()
        ):
            raise PilotV21111FreshCohortError(
                f"imported {model_id} authority lineage drifted"
            )
        _validate_zero_provider_capability_wrapper(
            contract,
            model_id,
            capability,
        )
        expected_runtime_model = (
            f"{contract.provider_profiles[model_id].transport}/"
            f"{contract.provider_profiles[model_id].requested_model}"
        )
        reservations = preflight.get("reservations")
        if (
            preflight.get("model_id") != model_id
            or preflight.get("runtime_model") != expected_runtime_model
            or preflight.get("sample_counts") != {"action": 24, "semantic": 8}
            or preflight.get("provider_construction_current_attempt") is not False
            or preflight.get("provider_calls_current_attempt") != 0
            or preflight.get("hosted_provider_calls_current_attempt") != 0
            or preflight.get("historical_provider_calls") != 32
            or preflight.get("historical_calls_already_in_parent_debit") is not True
            or preflight.get("imported_effect_cells") != 0
            or preflight.get("scientific_evidence") is not False
            or not isinstance(reservations, Mapping)
            or set(reservations) != {"action", "semantic"}
            or canonical_sha256(reservations)
            != preflight.get("source_reservation_sha256")
        ):
            raise PilotV21111FreshCohortError(
                f"imported {model_id} preflight authority drifted"
            )
    return {
        "model_ids": list(model_ids),
        "capability_wrapper_count": len(capabilities),
        "preflight_wrapper_count": len(preflights),
        "calibration_wrapper_content_sha256": calibration["integrity"][
            "content_sha256"
        ],
        "capability_wrapper_content_sha256": {
            model_id: capabilities[model_id]["integrity"]["content_sha256"]
            for model_id in model_ids
        },
        "preflight_wrapper_content_sha256": {
            model_id: preflights[model_id]["integrity"]["content_sha256"]
            for model_id in model_ids
        },
    }


def _verify_publication_hashes(repo: Path, boundary: Mapping[str, Any]) -> None:
    checks = (
        (
            "evidence/current_v2/pilot-v2.11.5/checksums.json",
            boundary["v2115_scientific_authority"]["publication_checksums_file_sha256"],
        ),
        (
            "evidence/current_v2/pilot-v2.11.5/package_manifest.json",
            boundary["v2115_scientific_authority"][
                "publication_package_manifest_file_sha256"
            ],
        ),
        (
            "evidence/current_v2/pilot-v2.11.10/checksums.json",
            boundary["v21110_terminal_release"]["publication_checksums_file_sha256"],
        ),
        (
            "evidence/current_v2/pilot-v2.11.10/package_manifest.json",
            boundary["v21110_terminal_release"][
                "publication_package_manifest_file_sha256"
            ],
        ),
    )
    for relative, expected in checks:
        path = repo / relative
        if path.is_symlink() or not path.is_file() or _file_sha256(path) != expected:
            raise PilotV21111FreshCohortError(
                f"publication authority hash drifted for {relative}"
            )


def verify_parent_sources(
    contract: PilotContract,
    *,
    repo_root: str | Path,
    v21110_repo_root: str | Path,
    v2115_repo_root: str | Path,
) -> dict[str, Any]:
    """Verify both immutable releases and the exact cumulative parent debit."""

    if contract.contract_id != V21111_CONTRACT_ID:
        raise PilotV21111FreshCohortError("source verification uses wrong contract")
    repo = _real_directory(repo_root, name="V2.11.11 repository")
    terminal = _real_directory(v21110_repo_root, name="V2.11.10 terminal release")
    authority = _real_directory(v2115_repo_root, name="V2.11.5 authority release")
    if len({repo, terminal, authority}) != 3:
        raise PilotV21111FreshCohortError("release roots must be distinct")
    boundary = contract.v21111_fresh_cohort_boundary
    assert isinstance(boundary, Mapping)
    v21110 = boundary["v21110_terminal_release"]
    v2115 = boundary["v2115_scientific_authority"]
    terminal_git = _verify_release_git(
        terminal,
        tag=v21110["science_tag"],
        tag_object=v21110["science_tag_object"],
        commit=v21110["science_commit"],
        name="V2.11.10 terminal release",
    )
    authority_git = _verify_release_git(
        authority,
        tag=v2115["science_tag"],
        tag_object=v2115["science_tag_object"],
        commit=v2115["science_commit"],
        name="V2.11.5 authority release",
    )
    for root, relative, expected, name in (
        (
            terminal,
            "experiments/pilot_v2_11_10.yaml",
            v21110["contract_file_sha256"],
            "V2.11.10 contract",
        ),
        (
            authority,
            "experiments/pilot_v2_11_5.yaml",
            v2115["contract_file_sha256"],
            "V2.11.5 contract",
        ),
    ):
        path = root / relative
        if path.is_symlink() or not path.is_file() or _file_sha256(path) != expected:
            raise PilotV21111FreshCohortError(f"{name} file hash drifted")
    terminal_contract = load_pilot_contract(
        terminal / "experiments/pilot_v2_11_10.yaml"
    )
    authority_contract = load_pilot_contract(
        authority / "experiments/pilot_v2_11_5.yaml"
    )
    if terminal_contract.canonical_hash != v21110["contract_sha256"]:
        raise PilotV21111FreshCohortError("V2.11.10 contract identity drifted")
    if authority_contract.canonical_hash != v2115["contract_sha256"]:
        raise PilotV21111FreshCohortError("V2.11.5 contract identity drifted")

    terminal_raw = terminal / "experiment_results/pilot-v2.11.10/raw"
    authority_raw = authority / "experiment_results/pilot-v2.11.5/raw"
    if _raw_inventory(
        terminal_raw,
        excluded=v21110["raw_inventory"]["excluded_operational_paths"],
    ) != {
        key: v21110["raw_inventory"][key]
        for key in ("file_count", "storage_bytes", "inventory_sha256")
    }:
        raise PilotV21111FreshCohortError("V2.11.10 raw inventory drifted")
    if _raw_inventory(authority_raw, excluded=(".real-stage-execution.lock",)) != dict(
        v2115["raw_inventory"]
    ):
        raise PilotV21111FreshCohortError("V2.11.5 raw inventory drifted")

    run_ledger_path = terminal_raw / "run_ledger.json"
    budget_ledger_path = terminal_raw / "budget_ledger.json"
    if _file_sha256(run_ledger_path) != v21110["run_ledger"]["file_sha256"]:
        raise PilotV21111FreshCohortError("V2.11.10 run ledger file drifted")
    if _file_sha256(budget_ledger_path) != v21110["budget_ledger"]["file_sha256"]:
        raise PilotV21111FreshCohortError("V2.11.10 budget ledger file drifted")
    run_ledger = _strict_json(run_ledger_path, name="V2.11.10 run ledger")
    budget_ledger = _strict_json(budget_ledger_path, name="V2.11.10 budget ledger")
    run_ledger_self_hash = _verify_ledger_self_hash(
        run_ledger,
        name="V2.11.10 run ledger",
    )
    budget_ledger_self_hash = _verify_ledger_self_hash(
        budget_ledger,
        name="V2.11.10 budget ledger",
    )
    if (
        run_ledger_self_hash != v21110["run_ledger"]["ledger_sha256"]
        or len(run_ledger.get("runs", {})) != 87
        or dict(Counter(row["status"] for row in run_ledger["runs"].values()))
        != dict(v21110["run_ledger"]["status_counts"])
    ):
        raise PilotV21111FreshCohortError("V2.11.10 terminal run ledger drifted")
    if budget_ledger_self_hash != v21110["budget_ledger"]["ledger_sha256"]:
        raise PilotV21111FreshCohortError("V2.11.10 budget ledger identity drifted")
    budget_runs = budget_ledger.get("runs")
    if not isinstance(budget_runs, Mapping):
        raise PilotV21111FreshCohortError("V2.11.10 budget rows are malformed")
    expected_actual = v21110["budget_ledger"]["current_actual"]
    if not isinstance(expected_actual, Mapping):
        raise PilotV21111FreshCohortError("V2.11.10 frozen current actual is malformed")
    fresh_actual = _verified_parent_current_actual(budget_runs, expected_actual)
    parent = parent_budget_debit_for_v21111(contract)
    old_parent = budget_ledger["parent_debit"]
    if (
        Decimal(str(old_parent["cost_usd"])) + fresh_actual["cost_usd"]
        != Decimal(str(parent.cost_usd))
        or old_parent["hosted_completions"] + fresh_actual["hosted_completions"]
        != parent.hosted_completions
        or old_parent["storage_bytes"] + fresh_actual["storage_bytes"]
        != parent.storage_bytes
    ):
        raise PilotV21111FreshCohortError("cumulative V2.11.11 parent debit drifted")
    _verify_publication_hashes(repo, boundary)
    authority_parent_path = (
        authority_raw / "parent-import" / "parent_import_receipt.json"
    )
    authority_parent = _strict_json(
        authority_parent_path,
        name="V2.11.5 parent import receipt",
    )
    try:
        authority_parent = validate_v2115_parent_import_receipt(
            authority_parent,
            contract=authority_contract,
            child_git_commit=str(v2115["science_commit"]),
            repo_root=authority,
        )
        calibration_wrapper = calibration_wrapper_from_v2115_receipt(authority_parent)
        capability_wrappers = capability_wrappers_from_v2115_receipt(authority_parent)
        preflight_wrappers = preflight_wrappers_from_v2115_receipt(authority_parent)
    except PilotV2115ParentImportError as exc:
        raise PilotV21111FreshCohortError(
            f"V2.11.5 imported authority wrapper failed validation: {exc}"
        ) from exc
    wrapper_denominator = _validate_imported_authority_wrappers(
        contract,
        calibration=calibration_wrapper,
        capabilities=capability_wrappers,
        preflights=preflight_wrappers,
    )
    return _seal(
        {
            "schema_version": V21111_PARENT_IMPORT_SCHEMA,
            "status": "go",
            "go": True,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "v2115_authority": authority_git,
            "v2115_parent_receipt": {
                "path": V21111_V2115_PARENT_RECEIPT_PATH,
                "file_sha256": _file_sha256(authority_parent_path),
                "content_sha256": authority_parent["integrity"]["content_sha256"],
            },
            "calibration_wrapper": calibration_wrapper,
            "capability_wrappers": capability_wrappers,
            "preflight_authority_wrappers": preflight_wrappers,
            "authority_wrapper_denominator": wrapper_denominator,
            "v21110_terminal": terminal_git,
            "parent_budget_debit": parent.to_dict(),
            "evidence_partition": stage_partition_from_contract(contract),
            "provider_boundary": {
                "provider_construction": False,
                "provider_calls": 0,
                "hosted_cost_usd": 0.0,
            },
            "scientific_evidence": False,
            "claim_boundary": V21111_PARENT_CLAIM_BOUNDARY,
        }
    )


def write_parent_import_receipt(
    contract: PilotContract,
    *,
    repo_root: str | Path,
    v21110_repo_root: str | Path,
    v2115_repo_root: str | Path,
    raw_root: str | Path,
) -> dict[str, Any]:
    require_provider_keys_absent()
    raw = require_exact_raw_namespace(
        contract_path=Path(repo_root) / "experiments" / "pilot_v2_11_11.yaml",
        raw_root=raw_root,
    )
    receipt = verify_parent_sources(
        contract,
        repo_root=repo_root,
        v21110_repo_root=v21110_repo_root,
        v2115_repo_root=v2115_repo_root,
    )
    path = raw / "parent-import" / V21111_PARENT_RECEIPT_FILENAME
    _atomic_new_json(path, receipt)
    return {**receipt, "receipt_path": str(path)}


def call_plan_for_v21111(contract: PilotContract) -> dict[str, Any]:
    """Build the exact provider-free simulated call denominator."""

    if contract.contract_id != V21111_CONTRACT_ID:
        raise PilotV21111FreshCohortError("call plan uses wrong contract")
    by_run: dict[str, int] = {}
    d_prefix_rows: list[dict[str, Any]] = []
    for spec in contract.expand():
        if spec.stage_id == "parent-import":
            by_run[spec.run_id] = 0
        elif spec.stage_id == "experiment-b":
            by_run[spec.run_id] = (
                48
                if spec.arm_id
                in {
                    "no-memory",
                    "episodic-only",
                }
                else 64
            )
        elif spec.stage_id == "cross-model":
            by_run[spec.run_id] = 48 if spec.arm_id == "no-memory" else 64
        elif spec.stage_id == "experiment-d":
            by_run[spec.run_id] = 24
        else:  # pragma: no cover - exact contract parser prevents this
            raise PilotV21111FreshCohortError("unexpected fresh-cohort stage")
    for seed in FRESH_MAIN_SEEDS:
        d_prefix_rows.append(
            {
                "coordinator_id": (
                    f"{contract.contract_id}--experiment-d--gpt52_main--"
                    f"checkpoint-prefix--s{seed}"
                ),
                "environment_seed": seed,
                "action_calls": 24,
                "semantic_calls": 8,
                "simulated_calls": 32,
                "denominator_cell": False,
            }
        )
    stage_totals = {
        stage_id: sum(
            calls for run_id, calls in by_run.items() if f"--{stage_id}--" in run_id
        )
        + (
            sum(row["simulated_calls"] for row in d_prefix_rows)
            if stage_id == "experiment-d"
            else 0
        )
        for stage_id in SCIENTIFIC_STAGE_IDS
    }
    if stage_totals != {
        "experiment-b": 1440,
        "experiment-d": 1480,
        "cross-model": 336,
    }:
        raise PilotV21111FreshCohortError(
            f"simulated call stage totals drifted: {stage_totals}"
        )
    if len(by_run) != 87 or sum(stage_totals.values()) != 3256:
        raise PilotV21111FreshCohortError("simulated call denominator drifted")
    return _seal(
        {
            "schema_version": "finevo-pilot-v2.11.11-simulated-call-plan-v1",
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "registered_cells": len(by_run),
            "scientific_cells": 86,
            "call_counts_by_run": dict(sorted(by_run.items())),
            "d_prefix_coordinators": d_prefix_rows,
            "calls_by_stage": stage_totals,
            "simulated_provider_calls": 3256,
            "provider_construction": False,
            "provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "scientific_evidence": False,
        }
    )


def build_provider_free_acceptance(contract: PilotContract) -> dict[str, Any]:
    """Exercise the full matrix shape without constructing a provider."""

    partition = stage_partition_from_contract(contract)
    plan = call_plan_for_v21111(contract)
    boundary = contract.v21111_fresh_cohort_boundary
    assert isinstance(boundary, Mapping)
    execution = boundary["execution_policy"]
    actor = contract.task_output_contracts["actor-action"]
    semantic = contract.task_output_contracts["semantic-proposal"]
    if (
        actor.max_completion_tokens != 8192
        or semantic.max_completion_tokens != 4096
        or execution["hosted_max_in_flight"] != 1
        or execution["request_timeout_seconds"] != 300
        or execution["provider_attempts_per_request"] != 1
        or any(
            profile.max_attempts != 1
            for profile_id, profile in contract.provider_profiles.items()
            if profile_id != "qref_scripted"
        )
    ):
        raise PilotV21111FreshCohortError("hosted dispatch envelope drifted")
    budget = boundary["budget_envelope"]
    if (
        budget["projected_cumulative_cost_usd"] > budget["hard_cap_usd"]
        or budget["projected_cumulative_hosted_completions"]
        > budget["hard_completion_cap"]
        or plan["simulated_provider_calls"] != 3256
    ):
        raise PilotV21111FreshCohortError("fresh matrix does not fit hard caps")
    return _seal(
        {
            "schema_version": V21111_ACCEPTANCE_SCHEMA,
            "status": "go",
            "go": True,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "registered_cells": 87,
            "scientific_cells": 86,
            "stage_partition": partition,
            "simulated_call_plan": plan,
            "budget_envelope": _json_copy(budget),
            "dispatch_envelope": _json_copy(execution),
            "dispatch_refresh_plan": _json_copy(boundary["dispatch_refresh"]),
            "provider_boundary": {
                "provider_construction": False,
                "provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "simulated_provider_calls_are_not_calls": True,
            },
            "scientific_evidence": False,
            "claim_boundary": (
                "Provider-free infrastructure acceptance only; 3256 is a logical "
                "simulated call count and no model output exists."
            ),
        }
    )


def verify_parent_import_receipt(
    path: str | Path,
    *,
    contract: PilotContract,
) -> dict[str, Any]:
    receipt_path = Path(path).absolute()
    if (
        receipt_path.name != V21111_PARENT_RECEIPT_FILENAME
        or receipt_path.parent.name != "parent-import"
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
    ):
        raise PilotV21111FreshCohortError(
            "V2.11.11 parent import receipt path is not exact"
        )
    try:
        if receipt_path.resolve(strict=True) != receipt_path:
            raise PilotV21111FreshCohortError(
                "V2.11.11 parent import receipt path contains a symlink"
            )
    except OSError as exc:
        raise PilotV21111FreshCohortError(
            "V2.11.11 parent import receipt is unavailable"
        ) from exc
    receipt = _strict_json(receipt_path, name="V2.11.11 parent import receipt")
    _verify_seal(receipt, name="V2.11.11 parent import receipt")
    expected_top_level = {
        "schema_version",
        "status",
        "go",
        "contract_id",
        "contract_sha256",
        "v2115_authority",
        "v2115_parent_receipt",
        "calibration_wrapper",
        "capability_wrappers",
        "preflight_authority_wrappers",
        "authority_wrapper_denominator",
        "v21110_terminal",
        "parent_budget_debit",
        "evidence_partition",
        "provider_boundary",
        "scientific_evidence",
        "claim_boundary",
        "integrity",
    }
    if set(receipt) != expected_top_level:
        raise PilotV21111FreshCohortError(
            "V2.11.11 parent import receipt top-level shape drifted"
        )
    calibration = receipt.get("calibration_wrapper")
    capabilities = receipt.get("capability_wrappers")
    preflight = receipt.get("preflight_authority_wrappers")
    try:
        if (
            not isinstance(calibration, Mapping)
            or not isinstance(capabilities, Mapping)
            or not isinstance(preflight, Mapping)
        ):
            raise PilotV2115ParentImportError("imported authority wrappers are absent")
        calibration = calibration_wrapper_from_v2115_receipt(
            {"calibration_wrapper": calibration}
        )
        capabilities = capability_wrappers_from_v2115_receipt(
            {"capability_wrappers": capabilities}
        )
        preflight = preflight_wrappers_from_v2115_receipt(
            {"preflight_authority_wrappers": preflight}
        )
    except PilotV2115ParentImportError as exc:
        raise PilotV21111FreshCohortError(
            f"parent imported authority wrapper drifted: {exc}"
        ) from exc
    expected_denominator = _validate_imported_authority_wrappers(
        contract,
        calibration=calibration,
        capabilities=capabilities,
        preflights=preflight,
    )
    boundary = contract.v21111_fresh_cohort_boundary
    if not isinstance(boundary, Mapping):
        raise PilotV21111FreshCohortError("fresh-cohort boundary is absent")
    v2115 = boundary.get("v2115_scientific_authority")
    v21110 = boundary.get("v21110_terminal_release")
    source_receipt = receipt.get("v2115_parent_receipt")
    expected_v2115_release = {
        "tag": v2115.get("science_tag") if isinstance(v2115, Mapping) else None,
        "tag_object": (
            v2115.get("science_tag_object") if isinstance(v2115, Mapping) else None
        ),
        "commit": (v2115.get("science_commit") if isinstance(v2115, Mapping) else None),
        "clean": True,
    }
    expected_v21110_release = {
        "tag": v21110.get("science_tag") if isinstance(v21110, Mapping) else None,
        "tag_object": (
            v21110.get("science_tag_object") if isinstance(v21110, Mapping) else None
        ),
        "commit": (
            v21110.get("science_commit") if isinstance(v21110, Mapping) else None
        ),
        "clean": True,
    }
    if (
        receipt.get("schema_version") != V21111_PARENT_IMPORT_SCHEMA
        or receipt.get("status") != "go"
        or receipt.get("go") is not True
        or receipt.get("contract_id") != contract.contract_id
        or receipt.get("contract_sha256") != contract.canonical_hash
        or receipt.get("v2115_authority") != expected_v2115_release
        or receipt.get("v21110_terminal") != expected_v21110_release
        or not isinstance(source_receipt, Mapping)
        or set(source_receipt) != {"path", "file_sha256", "content_sha256"}
        or source_receipt.get("path") != V21111_V2115_PARENT_RECEIPT_PATH
        or not _is_sha256(source_receipt.get("file_sha256"))
        or not _is_sha256(source_receipt.get("content_sha256"))
        or receipt.get("authority_wrapper_denominator") != expected_denominator
        or receipt.get("parent_budget_debit")
        != parent_budget_debit_for_v21111(contract).to_dict()
        or receipt.get("evidence_partition")
        != _json_copy(stage_partition_from_contract(contract))
        or receipt.get("provider_boundary")
        != {
            "provider_construction": False,
            "provider_calls": 0,
            "hosted_cost_usd": 0.0,
        }
        or receipt.get("scientific_evidence") is not False
        or receipt.get("claim_boundary") != V21111_PARENT_CLAIM_BOUNDARY
    ):
        raise PilotV21111FreshCohortError("parent import receipt drifted")
    return receipt


def verified_capability_wrapper_for_v21111(
    contract: PilotContract,
    model_id: str,
    *,
    raw_root: str | Path,
) -> dict[str, Any]:
    """Return one child-bound, zero-call V2.11.5 capability authority."""

    if model_id not in _authority_model_ids(contract):
        raise PilotV21111FreshCohortError(
            f"{model_id} has no V2.11.11 imported capability authority"
        )
    receipt = verify_parent_import_receipt(
        Path(raw_root) / "parent-import" / V21111_PARENT_RECEIPT_FILENAME,
        contract=contract,
    )
    wrapper = receipt["capability_wrappers"][model_id]
    _validate_zero_provider_capability_wrapper(contract, model_id, wrapper)
    return _json_copy(wrapper)


def verified_calibration_for_v21111(
    contract: PilotContract,
    *,
    raw_root: str | Path,
) -> dict[str, Any]:
    """Return the outcome-blind V2.11.5 calibration through the child receipt."""

    if contract.contract_id != V21111_CONTRACT_ID:
        raise PilotV21111FreshCohortError("calibration adapter uses wrong contract")
    receipt = verify_parent_import_receipt(
        Path(raw_root) / "parent-import" / V21111_PARENT_RECEIPT_FILENAME,
        contract=contract,
    )
    wrapper = receipt["calibration_wrapper"]
    calibration = wrapper.get("calibration")
    if not isinstance(calibration, Mapping):
        raise PilotV21111FreshCohortError("imported calibration is malformed")
    selected = calibration.get("selected_utility_profile")
    threshold = calibration.get("stage0_absolute_flow_utility_threshold")
    if (
        not isinstance(selected, Mapping)
        or not isinstance(threshold, Mapping)
        or threshold.get("treatment_outcomes_inspected") is not False
        or calibration.get("q_ref") != selected.get("consumption_scale")
    ):
        raise PilotV21111FreshCohortError("imported calibration scope drifted")
    return {
        "selected_profile_id": selected.get("profile_id"),
        "selected_utility": _json_copy(selected),
        "absolute_flow_utility_threshold": _json_copy(threshold),
        "q_ref": calibration.get("q_ref"),
        "receipt": {
            "path": str(
                Path(raw_root) / "parent-import" / V21111_PARENT_RECEIPT_FILENAME
            ),
            "integrity": _json_copy(receipt["integrity"]),
        },
        "source_wrapper_content_sha256": wrapper["integrity"]["content_sha256"],
    }


def verified_preflight_wrapper_for_v21111(
    contract: PilotContract,
    model_id: str,
    *,
    raw_root: str | Path,
) -> dict[str, Any]:
    """Return one imported model-by-call-kind reservation authority."""

    if model_id not in {"gpt52_main", "gpt56_diagnostic"}:
        raise PilotV21111FreshCohortError(
            f"{model_id} has no V2.11.11 imported preflight authority"
        )
    receipt = verify_parent_import_receipt(
        Path(raw_root) / "parent-import" / V21111_PARENT_RECEIPT_FILENAME,
        contract=contract,
    )
    wrapper = receipt["preflight_authority_wrappers"][model_id]
    reservations = wrapper.get("reservations")
    if (
        wrapper.get("model_id") != model_id
        or not isinstance(reservations, Mapping)
        or set(reservations) != {"action", "semantic"}
        or any(
            not isinstance(reservations[kind], Mapping)
            or not isinstance(reservations[kind].get("reservation"), Mapping)
            for kind in ("action", "semantic")
        )
    ):
        raise PilotV21111FreshCohortError("imported p95 wrapper drifted")
    return _json_copy(wrapper)


def verified_projection_for_v21111(
    contract: PilotContract,
    model_id: str,
    *,
    raw_root: str | Path,
) -> tuple[dict[str, Any], Path]:
    wrapper = verified_preflight_wrapper_for_v21111(
        contract,
        model_id,
        raw_root=raw_root,
    )
    profile = contract.provider_profiles[model_id]
    receipt_path = Path(raw_root) / "parent-import" / V21111_PARENT_RECEIPT_FILENAME
    return (
        {
            "schema_version": "finevo-pilot-v2.11.11-imported-p95-view-v1",
            "model_id": model_id,
            "served_model": profile.served_model,
            "projection": {
                f"{profile.served_model}::{kind}": _json_copy(
                    wrapper["reservations"][kind]["reservation"]
                )
                for kind in ("action", "semantic")
            },
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "parent_import_content_sha256": verify_parent_import_receipt(
                    receipt_path,
                    contract=contract,
                )["integrity"]["content_sha256"],
                "source_kind": "v2.11.5-sealed-parent-import",
                "source_wrapper_content_sha256": wrapper["integrity"]["content_sha256"],
            },
        },
        receipt_path,
    )


def verify_full_fake_acceptance(
    diagnostics_root: str | Path,
    *,
    contract: PilotContract,
) -> dict[str, Any]:
    """Verify the full runner-path fake acceptance and its exact ledgers."""

    root = Path(diagnostics_root) / V21111_FULL_FAKE_DIRECTORY
    if root.is_symlink() or not root.is_dir():
        raise PilotV21111FreshCohortError(
            "full fake-provider acceptance directory is unavailable"
        )
    receipt_path = root / V21111_FULL_FAKE_RECEIPT_FILENAME
    receipt = _strict_json(receipt_path, name="V2.11.11 full fake acceptance")
    unsigned = dict(receipt)
    claimed_sha256 = unsigned.pop("receipt_sha256", None)
    checks = receipt.get("checks")
    expected_checks = {
        "all_86_cells_complete",
        "exact_3256_simulated_science_calls",
        "exact_stage_call_counts",
        "provider_adapter_visited_every_call",
        "external_provider_calls_zero",
        "actor_cap_8192_exercised",
        "semantic_cap_4096_exercised",
        "valid_1024_byte_action_parsed",
        "hosted_max_in_flight_one",
        "second_replay_no_redispatch",
        "static_projection_agrees",
    }
    if (
        receipt.get("schema_version") != V21111_FULL_FAKE_SCHEMA
        or receipt.get("contract_id") != contract.contract_id
        or receipt.get("contract_sha256") != contract.canonical_hash
        or receipt.get("status") != "pass"
        or receipt.get("registered_science_cells") != 86
        or receipt.get("status_counts") != {"complete": 86}
        or receipt.get("simulated_science_calls") != 3_256
        or receipt.get("simulated_calls_by_stage")
        != {"experiment-b": 1_440, "experiment-d": 1_480, "cross-model": 336}
        or receipt.get("fake_provider_adapter_calls") != 3_256
        or receipt.get("external_provider_calls") != 0
        or receipt.get("hosted_cost_usd") != 0.0
        or receipt.get("task_call_counts")
        != {"actor-action": 2_928, "semantic-proposal": 328}
        or receipt.get("diagnostic_only") is not True
        or receipt.get("scientific_evidence") is not False
        or not isinstance(checks, Mapping)
        or set(checks) != expected_checks
        or any(value is not True for value in checks.values())
        or claimed_sha256 != canonical_sha256(unsigned)
    ):
        raise PilotV21111FreshCohortError(
            "full fake-provider acceptance receipt drifted"
        )
    bindings = receipt.get("ledger_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {"run", "budget"}:
        raise PilotV21111FreshCohortError(
            "full fake-provider ledger bindings are absent"
        )
    ledgers: dict[str, dict[str, Any]] = {}
    for kind, filename in (
        ("run", "run_ledger.json"),
        ("budget", "budget_ledger.json"),
    ):
        binding = bindings.get(kind)
        if not isinstance(binding, Mapping) or binding.get("relative_path") != filename:
            raise PilotV21111FreshCohortError(
                f"full fake-provider {kind} ledger path drifted"
            )
        path = root / filename
        if path.is_symlink() or _file_sha256(path) != binding.get("file_sha256"):
            raise PilotV21111FreshCohortError(
                f"full fake-provider {kind} ledger file drifted"
            )
        ledger = _strict_json(path, name=f"V2.11.11 fake {kind} ledger")
        if ledger.get("ledger_sha256") != binding.get("ledger_sha256"):
            raise PilotV21111FreshCohortError(
                f"full fake-provider {kind} ledger identity drifted"
            )
        ledgers[kind] = ledger
    if dict(Counter(row["status"] for row in ledgers["run"]["runs"].values())) != {
        "complete": 86
    }:
        raise PilotV21111FreshCohortError(
            "full fake-provider run denominator is not complete"
        )
    resolved_root = root.resolve()
    for run_id, row in ledgers["run"]["runs"].items():
        artifact = row.get("artifact")
        binding = row.get("artifact_binding")
        if not isinstance(artifact, str) or not isinstance(binding, Mapping):
            raise PilotV21111FreshCohortError(
                f"full fake-provider artifact binding is absent: {run_id}"
            )
        path = Path(artifact)
        absolute = path.absolute()
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or path.resolve() != absolute
            or not path.resolve().is_relative_to(resolved_root)
            or set(binding) != {"path", "file_sha256", "byte_size"}
            or binding.get("path") != artifact
            or binding.get("file_sha256") != _file_sha256(path)
            or binding.get("byte_size") != path.stat().st_size
        ):
            raise PilotV21111FreshCohortError(
                f"full fake-provider artifact binding drifted: {run_id}"
            )
    completions = sum(
        int(row["actual"]["completions"])
        for row in ledgers["budget"]["runs"].values()
        if isinstance(row.get("actual"), Mapping)
    )
    if completions != 3_256:
        raise PilotV21111FreshCohortError(
            "full fake-provider budget denominator drifted"
        )
    return receipt


def _acceptance_binding_common(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    prefixes = receipt.get("ledger_prefixes")
    if not isinstance(prefixes, Mapping) or set(prefixes) != {"run", "budget"}:
        raise PilotV21111FreshCohortError(
            "scientific acceptance ledger prefixes are malformed"
        )
    run = prefixes["run"]
    budget = prefixes["budget"]
    if not isinstance(run, Mapping) or not isinstance(budget, Mapping):
        raise PilotV21111FreshCohortError(
            "scientific acceptance ledger prefixes are malformed"
        )
    return {
        "receipt_schema_version": V21111_ACCEPTANCE_SCHEMA,
        "receipt_path": (
            Path(V21111_RAW_NAMESPACE) / V21111_ACCEPTANCE_FILENAME
        ).as_posix(),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "accepted_run_event_count": run.get("event_count"),
        "accepted_run_event_chain_head": run.get("event_chain_head"),
        "accepted_budget_event_count": budget.get("event_count"),
        "accepted_budget_event_chain_head": budget.get("event_chain_head"),
    }


def _validate_recoverable_acceptance_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: PilotContract,
    paid: Any,
    full_fake_binding: Mapping[str, Any],
    fault_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate receipt bytes before completing a crashed dual-ledger bind."""

    _verify_seal(receipt, name="V2.11.11 scientific acceptance")
    base = build_provider_free_acceptance(contract)
    expected_keys = (set(base) - {"integrity"}) | {
        "full_fake_acceptance_binding",
        "fault_acceptance_binding",
        "ledger_prefixes",
        "release_provenance",
        "integrity",
    }
    release = receipt.get("release_provenance")
    prefixes = receipt.get("ledger_prefixes")
    if (
        set(receipt) != expected_keys
        or any(
            receipt.get(key) != value
            for key, value in base.items()
            if key != "integrity"
        )
        or receipt.get("full_fake_acceptance_binding") != dict(full_fake_binding)
        or receipt.get("fault_acceptance_binding") != dict(fault_binding)
        or not isinstance(release, Mapping)
        or release
        != {
            "git_tag": paid.git_tag,
            "git_commit": paid.head_commit,
            "worktree_clean": True,
        }
        or not isinstance(prefixes, Mapping)
        or set(prefixes) != {"run", "budget"}
    ):
        raise PilotV21111FreshCohortError(
            "existing scientific acceptance receipt drifted"
        )
    for kind in ("run", "budget"):
        prefix = prefixes[kind]
        if (
            not isinstance(prefix, Mapping)
            or set(prefix) != {"event_count", "event_chain_head", "ledger_sha256"}
            or isinstance(prefix.get("event_count"), bool)
            or not isinstance(prefix.get("event_count"), int)
            or int(prefix["event_count"]) < 1
            or not isinstance(prefix.get("event_chain_head"), str)
            or len(prefix["event_chain_head"]) != 64
            or not isinstance(prefix.get("ledger_sha256"), str)
            or len(prefix["ledger_sha256"]) != 64
        ):
            raise PilotV21111FreshCohortError(
                "existing scientific acceptance prefix drifted"
            )
    return _acceptance_binding_common(receipt)


def _acceptance_marker_present(
    snapshot: Mapping[str, Any],
    *,
    kind: str,
    common: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> bool:
    prefix = receipt["ledger_prefixes"][kind]
    count = int(prefix["event_count"])
    events = snapshot.get("events")
    if (
        not isinstance(events, list)
        or len(events) < count
        or events[count - 1].get("event_sha256") != prefix["event_chain_head"]
    ):
        raise PilotV21111FreshCohortError(
            f"scientific acceptance {kind} prefix drifted during recovery"
        )
    if len(events) == count:
        if snapshot.get("ledger_sha256") != prefix["ledger_sha256"]:
            raise PilotV21111FreshCohortError(
                f"scientific acceptance {kind} pre-bind ledger drifted"
            )
        return False
    marker = events[count]
    payload = marker.get("payload") if isinstance(marker, Mapping) else None
    if (
        not isinstance(marker, Mapping)
        or marker.get("event_type") != "acceptance_receipt_bound"
        or not isinstance(payload, Mapping)
        or any(payload.get(key) != value for key, value in common.items())
        or sum(
            event.get("event_type") == "acceptance_receipt_bound"
            for event in events
            if isinstance(event, Mapping)
        )
        != 1
    ):
        raise PilotV21111FreshCohortError(
            f"scientific acceptance {kind} marker drifted during recovery"
        )
    return True


def accept_scientific_dispatch(
    *,
    contract_path: str | Path,
    repo_root: str | Path,
    raw_root: str | Path,
    scientific_launch_input_path: str | Path,
    diagnostics_root: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Seal zero-provider acceptance after a frozen parent import exists."""

    require_provider_keys_absent()
    contract = load_pilot_contract(contract_path)
    if contract.contract_id != V21111_CONTRACT_ID or contract.status != "frozen":
        raise PilotV21111FreshCohortError(
            "scientific dispatch acceptance requires frozen V2.11.11"
        )
    raw = require_exact_raw_namespace(
        contract_path=contract_path,
        raw_root=raw_root,
    )
    verify_parent_import_receipt(
        raw / "parent-import" / V21111_PARENT_RECEIPT_FILENAME,
        contract=contract,
    )
    diagnostics = require_exact_diagnostics_namespace(
        contract_path=contract_path,
        diagnostics_root=(
            diagnostics_root
            if diagnostics_root is not None
            else Path(contract_path).absolute().parent.parent
            / V21111_DIAGNOSTICS_NAMESPACE
        ),
    )
    full_fake = verify_full_fake_acceptance(diagnostics, contract=contract)
    fault_path = (
        diagnostics
        / "provider-free-fault-isolation-acceptance"
        / ("acceptance_receipt.json")
    )
    fault = _strict_json(fault_path, name="V2.11.11 fake fault acceptance")
    unsigned_fault = dict(fault)
    claimed_fault_sha256 = unsigned_fault.pop("receipt_sha256", None)
    if (
        fault.get("schema_version") != "finevo-pilot-v2.11.11-fake-fault-acceptance-v1"
        or fault.get("contract_sha256") != contract.canonical_hash
        or fault.get("status") != "pass"
        or fault.get("external_provider_calls") != 0
        or not isinstance(fault.get("checks"), Mapping)
        or set(fault["checks"]) != V21111_FAULT_ACCEPTANCE_CHECKS
        or any(value is not True for value in fault["checks"].values())
        or claimed_fault_sha256 != canonical_sha256(unsigned_fault)
    ):
        raise PilotV21111FreshCohortError(
            "fake fault-isolation acceptance receipt drifted"
        )
    # Import lazily to keep the provider-free lineage module acyclic at import
    # time. This validates the annotated release, clean worktree, exact CI
    # selection, and scientific launch binding without constructing a provider.
    from . import pilot_orchestrator as orch

    try:
        fault = orch.verify_v21111_fake_fault_acceptance(
            diagnostics,
            contract=contract,
        )
    except orch.PilotOrchestrationError as exc:
        raise PilotV21111FreshCohortError(str(exc)) from exc
    repository = _real_directory(repo_root, name="V2.11.11 repository")
    launch = Path(scientific_launch_input_path).absolute()
    expected_launch = diagnostics / "scientific_launch_input.json"
    if launch != expected_launch:
        raise PilotV21111FreshCohortError(
            "V2.11.11 scientific launch input must live in diagnostics"
        )
    paid = orch.verify_paid_provenance(
        contract,
        repo_root=repository,
        scientific_launch_input_path=launch,
    )
    orch._persist_release_attestation(raw, paid)
    run_ledger = orch.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    budget_ledger = orch.PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orch._budget_caps(contract),
        tamper_evident=True,
        parent_debit=parent_budget_debit_for_v21111(contract),
    )
    parent_specs = tuple(contract.expand(stage="parent-import"))
    if (
        len(parent_specs) != 1
        or run_ledger.status(parent_specs[0].run_id) != "complete"
        or any(
            run_ledger.status(spec.run_id) != "scheduled"
            for stage_id in SCIENTIFIC_STAGE_IDS
            for spec in contract.expand(stage=stage_id)
        )
    ):
        raise PilotV21111FreshCohortError(
            "V2.11.11 pre-dispatch ITT ledger state drifted"
        )
    orch._verify_v2_stage_receipt(
        contract,
        "parent-import",
        _strict_json(
            raw / "parent-import" / "stage_receipt.json",
            name="V2.11.11 parent stage receipt",
        ),
        raw_root=raw,
        ledger=run_ledger,
        paid=paid,
        authority_repo_root=repository,
    )
    parent_budget_row = budget_ledger.snapshot()["runs"].get(parent_specs[0].run_id)
    if (
        not isinstance(parent_budget_row, Mapping)
        or parent_budget_row.get("status") != "complete"
        or parent_budget_row.get("actual", {}).get("completions") != 0
        or parent_budget_row.get("actual", {}).get("cost_usd") != 0.0
    ):
        raise PilotV21111FreshCohortError(
            "V2.11.11 parent budget row is not terminal zero-provider evidence"
        )
    science_paths = tuple(raw / stage_id for stage_id in SCIENTIFIC_STAGE_IDS)
    if any(path.exists() or path.is_symlink() for path in science_paths):
        raise PilotV21111FreshCohortError(
            "scientific raw namespace is not fresh before acceptance"
        )
    full_fake_binding = {
        "relative_path": (
            f"{V21111_DIAGNOSTICS_NAMESPACE}/"
            f"{V21111_FULL_FAKE_DIRECTORY}/{V21111_FULL_FAKE_RECEIPT_FILENAME}"
        ),
        "file_sha256": _file_sha256(
            diagnostics / V21111_FULL_FAKE_DIRECTORY / V21111_FULL_FAKE_RECEIPT_FILENAME
        ),
        "receipt_sha256": full_fake["receipt_sha256"],
    }
    fault_binding = {
        "relative_path": (
            f"{V21111_DIAGNOSTICS_NAMESPACE}/"
            "provider-free-fault-isolation-acceptance/acceptance_receipt.json"
        ),
        "file_sha256": _file_sha256(fault_path),
        "receipt_sha256": fault["receipt_sha256"],
    }
    target = raw / V21111_ACCEPTANCE_FILENAME
    if receipt_path is not None and Path(receipt_path).absolute() != target:
        raise PilotV21111FreshCohortError(
            "V2.11.11 acceptance receipt must use its exact raw path"
        )
    if target.is_symlink() or target.parent.is_symlink():
        raise PilotV21111FreshCohortError(
            "V2.11.11 acceptance path must not be a symlink"
        )

    # Publication is intentionally receipt-first.  If the process stopped
    # after that immutable write (or after only the run-ledger marker), verify
    # the exact bytes and prefixes, append only the missing marker, and never
    # rewrite either the receipt or an existing marker.
    if target.exists():
        existing = _strict_json(target, name="V2.11.11 scientific acceptance")
        common = _validate_recoverable_acceptance_receipt(
            existing,
            contract=contract,
            paid=paid,
            full_fake_binding=full_fake_binding,
            fault_binding=fault_binding,
        )
        run_present = _acceptance_marker_present(
            run_ledger.snapshot(),
            kind="run",
            common=common,
            receipt=existing,
        )
        budget_present = _acceptance_marker_present(
            budget_ledger.snapshot(),
            kind="budget",
            common=common,
            receipt=existing,
        )
        if not run_present:
            run_ledger.bind_acceptance_receipt(**common)
        if not budget_present:
            budget_ledger.bind_acceptance_receipt(**common)
        verified = verify_scientific_dispatch_acceptance(
            contract=contract,
            raw_root=raw,
            paid=paid,
        )
        return {**verified, "receipt_path": str(target)}

    acceptance = build_provider_free_acceptance(contract)
    acceptance["full_fake_acceptance_binding"] = full_fake_binding
    acceptance["fault_acceptance_binding"] = fault_binding
    run_snapshot = run_ledger.snapshot()
    budget_snapshot = budget_ledger.snapshot()
    run_events = run_snapshot["events"]
    budget_events = budget_snapshot["events"]
    acceptance["ledger_prefixes"] = {
        "run": {
            "event_count": len(run_events),
            "event_chain_head": run_events[-1]["event_sha256"],
            "ledger_sha256": run_snapshot["ledger_sha256"],
        },
        "budget": {
            "event_count": len(budget_events),
            "event_chain_head": budget_events[-1]["event_sha256"],
            "ledger_sha256": budget_snapshot["ledger_sha256"],
        },
    }
    acceptance["release_provenance"] = {
        "git_tag": paid.git_tag,
        "git_commit": paid.head_commit,
        "worktree_clean": paid.worktree_clean,
    }
    # Adding the binding changes the sealed payload, so reseal it exactly once.
    acceptance.pop("integrity", None)
    acceptance = _seal(acceptance)
    _atomic_new_json(target, acceptance)
    common = _acceptance_binding_common(acceptance)
    run_ledger.bind_acceptance_receipt(**common)
    budget_ledger.bind_acceptance_receipt(**common)
    return {**acceptance, "receipt_path": str(target)}


def verify_scientific_dispatch_acceptance(
    *,
    contract: PilotContract,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    """Verify the immutable pre-provider acceptance and both ledger markers."""

    raw = Path(raw_root).absolute()
    target = raw / V21111_ACCEPTANCE_FILENAME
    receipt = _strict_json(target, name="V2.11.11 scientific acceptance")
    _verify_seal(receipt, name="V2.11.11 scientific acceptance")
    boundary = contract.v21111_fresh_cohort_boundary
    if not isinstance(boundary, Mapping):
        raise PilotV21111FreshCohortError("fresh-cohort boundary is absent")
    release = receipt.get("release_provenance")
    prefixes = receipt.get("ledger_prefixes")
    if (
        receipt.get("schema_version") != V21111_ACCEPTANCE_SCHEMA
        or receipt.get("status") != "go"
        or receipt.get("go") is not True
        or receipt.get("contract_id") != contract.contract_id
        or receipt.get("contract_sha256") != contract.canonical_hash
        or receipt.get("registered_cells") != 87
        or receipt.get("scientific_cells") != 86
        or receipt.get("dispatch_refresh_plan")
        != _json_copy(boundary["dispatch_refresh"])
        or receipt.get("provider_boundary", {}).get("provider_calls") != 0
        or receipt.get("scientific_evidence") is not False
        or not isinstance(release, Mapping)
        or release.get("git_tag") != getattr(paid, "git_tag", None)
        or release.get("git_commit") != getattr(paid, "head_commit", None)
        or release.get("worktree_clean") is not True
        or not isinstance(prefixes, Mapping)
        or set(prefixes) != {"run", "budget"}
    ):
        raise PilotV21111FreshCohortError("scientific dispatch acceptance drifted")
    verify_parent_import_receipt(
        raw / "parent-import" / V21111_PARENT_RECEIPT_FILENAME,
        contract=contract,
    )
    from . import pilot_orchestrator as orch

    run = orch.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    ).snapshot()
    budget = orch.PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orch._budget_caps(contract),
        tamper_evident=True,
        parent_debit=parent_budget_debit_for_v21111(contract),
    ).snapshot()
    for kind, ledger in (("run", run), ("budget", budget)):
        prefix = prefixes[kind]
        if not isinstance(prefix, Mapping):
            raise PilotV21111FreshCohortError("acceptance ledger prefix is malformed")
        count = prefix.get("event_count")
        events = ledger.get("events")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or not isinstance(events, list)
            or len(events) <= count
            or events[count - 1].get("event_sha256") != prefix.get("event_chain_head")
            or events[count].get("event_type") != "acceptance_receipt_bound"
            or events[count].get("payload", {}).get("receipt_content_sha256")
            != receipt["integrity"]["content_sha256"]
        ):
            raise PilotV21111FreshCohortError(
                f"scientific acceptance {kind} ledger binding drifted"
            )
    return receipt


@dataclass
class DBranchCoordinator:
    """Pure V2.11.11 D prefix/branch terminalization state machine."""

    seed: int
    prefix_status: str = "scheduled"
    branch_status: dict[str, str] | None = None
    active_branch: str | None = None

    def __post_init__(self) -> None:
        if self.seed not in FRESH_MAIN_SEEDS:
            raise PilotV21111FreshCohortError("D coordinator seed is not registered")
        if self.branch_status is None:
            self.branch_status = {branch: "scheduled" for branch in D_BRANCH_IDS}
        if set(self.branch_status) != set(D_BRANCH_IDS):
            raise PilotV21111FreshCohortError("D branch denominator drifted")
        if self.prefix_status not in {"scheduled", "running", "complete", "failed"}:
            raise PilotV21111FreshCohortError("D prefix status is invalid")
        if (
            self.active_branch is not None
            and self.branch_status.get(self.active_branch) != "running"
        ):
            raise PilotV21111FreshCohortError("D active branch state drifted")

    def start_prefix(self) -> None:
        if self.prefix_status != "scheduled":
            raise PilotV21111FreshCohortError("D prefix cannot be restarted")
        self.prefix_status = "running"

    def finish_prefix(self, *, success: bool) -> None:
        if self.prefix_status != "running":
            raise PilotV21111FreshCohortError("D prefix is not running")
        self.prefix_status = "complete" if success else "failed"
        if not success:
            assert self.branch_status is not None
            for branch in self.branch_status:
                self.branch_status[branch] = "failed-prefix"

    def start_branch(self, branch: str) -> None:
        if self.prefix_status != "complete":
            raise PilotV21111FreshCohortError("D branch requires a complete prefix")
        if branch not in D_BRANCH_IDS:
            raise PilotV21111FreshCohortError("D branch is not registered")
        assert self.branch_status is not None
        if self.branch_status[branch] != "scheduled":
            raise PilotV21111FreshCohortError("D branch cannot be retried")
        if self.active_branch is not None:
            raise PilotV21111FreshCohortError(
                "hosted_max_in_flight=1 forbids concurrent D branches"
            )
        self.branch_status[branch] = "running"
        self.active_branch = branch

    def finish_branch(self, branch: str, *, success: bool) -> None:
        assert self.branch_status is not None
        if self.active_branch != branch or self.branch_status.get(branch) != "running":
            raise PilotV21111FreshCohortError("D branch is not the active branch")
        self.branch_status[branch] = "complete" if success else "failed"
        self.active_branch = None

    def recover_after_interruption(self) -> tuple[str, ...]:
        """Stop only in-flight work; preserve untouched scheduled branches."""

        stopped: list[str] = []
        if self.prefix_status == "running":
            self.prefix_status = "failed"
            assert self.branch_status is not None
            for branch in self.branch_status:
                if self.branch_status[branch] == "scheduled":
                    self.branch_status[branch] = "failed-prefix"
                    stopped.append(branch)
        elif self.active_branch is not None:
            assert self.branch_status is not None
            branch = self.active_branch
            self.branch_status[branch] = "integrity-stopped"
            self.active_branch = None
            stopped.append(branch)
        return tuple(stopped)

    @property
    def untouched_branches(self) -> tuple[str, ...]:
        assert self.branch_status is not None
        return tuple(
            branch
            for branch in D_BRANCH_IDS
            if self.branch_status[branch] == "scheduled"
        )

    def to_dict(self) -> dict[str, Any]:
        return _seal(
            {
                "schema_version": V21111_D_COORDINATOR_SCHEMA,
                "seed": self.seed,
                "prefix_status": self.prefix_status,
                "branch_status": dict(self.branch_status or {}),
                "active_branch": self.active_branch,
                "max_in_flight": 1,
                "prefix_failure_scope": "all-eleven-seed-cells",
                "branch_failure_scope": "single-branch-cell",
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DBranchCoordinator":
        _verify_seal(value, name="V2.11.11 D coordinator")
        if value.get("schema_version") != V21111_D_COORDINATOR_SCHEMA:
            raise PilotV21111FreshCohortError("D coordinator schema drifted")
        if value.get("max_in_flight") != 1:
            raise PilotV21111FreshCohortError("D coordinator concurrency drifted")
        return cls(
            seed=int(value["seed"]),
            prefix_status=str(value["prefix_status"]),
            branch_status={
                str(key): str(status) for key, status in value["branch_status"].items()
            },
            active_branch=(
                None
                if value.get("active_branch") is None
                else str(value["active_branch"])
            ),
        )


__all__ = [
    "D_BRANCH_IDS",
    "DBranchCoordinator",
    "FRESH_MAIN_SEEDS",
    "OPERATIONAL_STAGE_IDS",
    "PilotV21111FreshCohortError",
    "SCIENTIFIC_STAGE_IDS",
    "V21111_ACCEPTANCE_FILENAME",
    "V21111_CONTRACT_ID",
    "V21111_DIAGNOSTICS_NAMESPACE",
    "V21111_PARENT_RECEIPT_FILENAME",
    "accept_scientific_dispatch",
    "build_provider_free_acceptance",
    "call_plan_for_v21111",
    "parent_budget_debit_for_v21111",
    "require_provider_keys_absent",
    "require_exact_diagnostics_namespace",
    "require_fresh_raw_before_parent_import",
    "stage_partition_from_contract",
    "verified_capability_wrapper_for_v21111",
    "verify_parent_import_receipt",
    "verify_full_fake_acceptance",
    "verify_parent_sources",
    "write_parent_import_receipt",
]
