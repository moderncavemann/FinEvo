"""Fail-closed controls for the pilot-v2.2 capability-evaluator amendment.

The amendment does not reinterpret scientific-effect outcomes and does not
dispatch a provider.  It imports two already-terminal capability attempts
through a tracked, self-hashed receipt:

* GPT-5.2 from pilot-v2.1; and
* the controlled local Llama from pilot-v2.

Only the capability proposal-legality diagnostic is corrected.  The receipt
retains compact projections of every affected proposal row and every one of
the 30 per-model usage rows required by the later preflight p95 calculation.
Raw completion text is intentionally neither required nor reconstructed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from .pilot_budget import ParentBudgetDebit
from .pilot_contract import PilotContract, PilotRunSpec, canonical_sha256


EVALUATOR_AMENDMENT_SCHEMA_VERSION = "finevo-pilot-evaluator-amendment-v1"
EVALUATOR_AMENDMENT_ID = (
    "finevo-pilot-v2.2-capability-admission-correction-1"
)
EVALUATOR_RECEIPT_SCHEMA_VERSION = (
    "finevo-pilot-evaluator-amendment-receipt-v1"
)
CAPABILITY_IMPORT_SCHEMA_VERSION = "finevo-capability-import-v1"

EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH = Path(
    "experiments/pilot_v2_1_capability_evaluator_correction.json"
)
EVALUATOR_CORRECTION_RAW_FILENAME = "evaluator_amendment_receipt.json"

V21_CONTRACT_SHA256 = (
    "ac1011e70f3fe85716c4f5c1497812e3c83b3112d7661d234fdaa913f58eadca"
)
V21_RUN_LEDGER_FILE_SHA256 = (
    "73b4e8572730ea91a9ab3838ea36b65b6d17625797333ee2eee53c2186ac736e"
)
V21_RUN_LEDGER_INTERNAL_SHA256 = (
    "ad3e5deb3a01f0ab4b5d079657830755170f9cbe88ab8b8d5cbdf21625738de9"
)
V21_RUN_LEDGER_EVENT_HEAD = (
    "fa0e02bc90fe3a7480277045f17dccda9a8e35bebcc0833018f9bf167a910ef1"
)
V21_BUDGET_LEDGER_FILE_SHA256 = (
    "405a818a1ee0dd4562e3a5452104f6b3411834f1d9d5800a81dcbc515b0144f2"
)
V21_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "8a871a52014a49d50ac8c8701022fea74b57df61aaa2e6a9f4d18c009200d2e1"
)
V21_BUDGET_LEDGER_EVENT_HEAD = (
    "c5ca4eb321ef73c7e623cd80a7d17e40d1b99a24df5bf5a911ec7fc790114877"
)
V22_PARENT_DEBIT_RECORD_SHA256 = (
    "8d1b835decb7c68328ad87974348cc5b7cdb1bcaf8ed984bc30fc7151389464b"
)

GPT52_MODEL_ID = "gpt52_main"
LLAMA33_MODEL_ID = "llama33_local_controlled"
EXPECTED_MODEL_IDS = (GPT52_MODEL_ID, LLAMA33_MODEL_ID)
EXPECTED_TASK_IDS = tuple(
    [f"action-{index:02d}" for index in range(1, 13)]
    + [f"rule-{index:02d}" for index in range(1, 13)]
    + [f"proposal-{index:02d}" for index in range(1, 7)]
)
EXPECTED_PROPOSAL_TASK_IDS = tuple(
    f"proposal-{index:02d}" for index in range(1, 7)
)

# Filled after the tracked receipt is finalized.  These constants make a fully
# rehashed edit fail closed instead of trusting a self-authored replacement.
EXPECTED_EVALUATOR_AMENDMENT_SHA256 = (
    "5c973039795d00c153e84b1bc0b480da322776a06c15aa20ab7e9e886cb8f9eb"
)
EXPECTED_EVALUATOR_RECEIPT_CONTENT_SHA256 = (
    "2e519826bf742a4496f6d094a02c57e7f901c56ca1604d196cc51f6ad2f9c531"
)
EXPECTED_EVALUATOR_RECEIPT_FILE_SHA256 = (
    "6c85a73c5a26d2752e00b7b6f62a5e8d9044ebc489c8ffaee7ddf633905a8e84"
)


class PilotEvaluationAmendmentError(RuntimeError):
    """Raised before import or dispatch when the amendment is not exact."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotEvaluationAmendmentError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PilotEvaluationAmendmentError(f"{name} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise PilotEvaluationAmendmentError(
            f"{name} has wrong fields: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _sha256_hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PilotEvaluationAmendmentError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PilotEvaluationAmendmentError(
            f"{name} must be a hexadecimal SHA-256 digest"
        ) from exc
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotEvaluationAmendmentError(
            f"{name} must be a non-negative integer"
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise PilotEvaluationAmendmentError(
            f"{name} must be a finite non-negative number"
        )
    return float(value)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _strict_json_bytes(raw: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PilotEvaluationAmendmentError(
            f"non-finite JSON constant is forbidden: {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotEvaluationAmendmentError(
                    f"duplicate evaluator receipt key is forbidden: {key!r}"
                )
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8", "strict")
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PilotEvaluationAmendmentError(
            f"cannot parse evaluator amendment receipt: {name}"
        ) from exc
    if not isinstance(value, dict):
        raise PilotEvaluationAmendmentError(
            "evaluator amendment receipt must be an object"
        )
    return value


def _read_regular_bytes_once(path: Path, *, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    elif path.is_symlink():  # pragma: no cover - supported target platforms
        raise PilotEvaluationAmendmentError(f"{name} must not be a symlink")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PilotEvaluationAmendmentError(
                f"{name} must be a regular file"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read()
            after = os.fstat(stream.fileno())
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise PilotEvaluationAmendmentError(
                f"{name} changed while it was read"
            )
        return raw
    except OSError as exc:
        raise PilotEvaluationAmendmentError(
            f"cannot safely read {name}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _contract_mapping(contract: PilotContract | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(contract, Mapping):
        return contract
    return contract.to_dict()


def _amendment_from_contract(
    contract: PilotContract | Mapping[str, Any],
) -> Mapping[str, Any] | None:
    value = _contract_mapping(contract).get("evaluator_amendment")
    if value is None:
        return None
    return _mapping(value, "contract evaluator_amendment")


def _validate_score_map(
    value: Any,
    *,
    name: str,
    expected_proposal_correct: int,
) -> None:
    scores = _mapping(value, name)
    _exact_keys(
        scores,
        {"utility-ranking", "rule-application", "rule-proposal"},
        name,
    )
    for category, row_value in scores.items():
        row = _mapping(row_value, f"{name}.{category}")
        _exact_keys(
            row,
            {"correct", "denominator", "required"},
            f"{name}.{category}",
        )
        correct = _nonnegative_int(row["correct"], f"{name}.{category}.correct")
        denominator = _nonnegative_int(
            row["denominator"], f"{name}.{category}.denominator"
        )
        required = _nonnegative_int(
            row["required"], f"{name}.{category}.required"
        )
        if denominator <= 0 or correct > denominator or required > denominator:
            raise PilotEvaluationAmendmentError(
                f"{name}.{category} is not a valid fixed-denominator score"
            )
    proposal = _mapping(scores["rule-proposal"], f"{name}.rule-proposal")
    if (
        proposal["correct"] != expected_proposal_correct
        or proposal["denominator"] != 6
        or proposal["required"] != 5
    ):
        raise PilotEvaluationAmendmentError(
            f"{name}.rule-proposal differs from the frozen correction"
        )


def _validate_proposal_projection(value: Any, *, attempt_id: str) -> str:
    row = _mapping(value, f"{attempt_id} proposal projection")
    _exact_keys(
        row,
        {
            "task_id",
            "interface_valid",
            "strict_parse",
            "parse_mode",
            "strict_schema_valid",
            "semantic_candidate_accepted",
            "candidate_status",
            "supporting_episode_ids",
            "raw_output_sha256",
        },
        f"{attempt_id} proposal projection",
    )
    task_id = row["task_id"]
    if task_id not in EXPECTED_PROPOSAL_TASK_IDS:
        raise PilotEvaluationAmendmentError(
            f"{attempt_id} has an unknown proposal task"
        )
    if (
        row["interface_valid"] is not True
        or row["strict_parse"] is not True
        or row["parse_mode"] != "exact_json"
        or row["strict_schema_valid"] is not True
        or row["semantic_candidate_accepted"] is not True
        or row["candidate_status"] != "provisional"
    ):
        raise PilotEvaluationAmendmentError(
            f"{attempt_id} proposal {task_id} fails corrected legality inputs"
        )
    episode_ids = _sequence(
        row["supporting_episode_ids"],
        f"{attempt_id} proposal {task_id} supporting_episode_ids",
    )
    if (
        len(episode_ids) != 3
        or len(set(episode_ids)) != 3
        or any(not isinstance(item, str) or not item for item in episode_ids)
    ):
        raise PilotEvaluationAmendmentError(
            f"{attempt_id} proposal {task_id} must cite three unique episodes"
        )
    task_number = int(str(task_id).split("-")[-1])
    expected_episode_ids = [
        (
            f"capability-proposal-{task_number}:s2010922376:"
            f"a{task_number % 4}:t{period}"
        )
        for period in range(1, 4)
    ]
    if list(episode_ids) != expected_episode_ids:
        raise PilotEvaluationAmendmentError(
            f"{attempt_id} proposal {task_id} cites an unregistered episode"
        )
    _sha256_hex(
        row["raw_output_sha256"],
        f"{attempt_id} proposal {task_id} raw_output_sha256",
    )
    return str(task_id)


def _validate_usage_projection(
    value: Any,
    *,
    attempt_id: str,
    expected_model: str,
) -> str:
    row = _mapping(value, f"{attempt_id} usage projection")
    _exact_keys(
        row,
        {
            "task_id",
            "response_model",
            "call_kind",
            "output_contract_id",
            "usage",
        },
        f"{attempt_id} usage projection",
    )
    task_id = row["task_id"]
    if task_id not in EXPECTED_TASK_IDS:
        raise PilotEvaluationAmendmentError(
            f"{attempt_id} has an unknown usage task"
        )
    expected_contract = (
        "semantic-proposal"
        if str(task_id).startswith("proposal-")
        else "actor-action"
    )
    if (
        row["response_model"] != expected_model
        or row["call_kind"] != expected_contract
        or row["output_contract_id"] != expected_contract
    ):
        raise PilotEvaluationAmendmentError(
            f"{attempt_id} usage {task_id} route fields differ"
        )
    usage = _mapping(row["usage"], f"{attempt_id} usage {task_id}")
    _exact_keys(
        usage,
        {"prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"},
        f"{attempt_id} usage {task_id}",
    )
    prompt = _nonnegative_int(
        usage["prompt_tokens"], f"{attempt_id} usage {task_id}.prompt_tokens"
    )
    completion = _nonnegative_int(
        usage["completion_tokens"],
        f"{attempt_id} usage {task_id}.completion_tokens",
    )
    total = _nonnegative_int(
        usage["total_tokens"], f"{attempt_id} usage {task_id}.total_tokens"
    )
    _nonnegative_float(
        usage["cost_usd"], f"{attempt_id} usage {task_id}.cost_usd"
    )
    if total != prompt + completion:
        raise PilotEvaluationAmendmentError(
            f"{attempt_id} usage {task_id} token total is not additive"
        )
    return str(task_id)


def _validate_model_audits(value: Any) -> dict[str, Mapping[str, Any]]:
    projections = _sequence(value, "model_audits")
    if len(projections) != 2:
        raise PilotEvaluationAmendmentError(
            "model_audits requires exactly two source attempts"
        )
    by_model: dict[str, Mapping[str, Any]] = {}
    for item in projections:
        attempt = _mapping(item, "model audit")
        _exact_keys(
            attempt,
            {
                "attempt_id",
                "model_id",
                "response_model",
                "taskset_sha256",
                "source_hashes",
                "old_terminal_status",
                "old_scores",
                "proposal_rows",
            },
            "model audit",
        )
        attempt_id = str(attempt["attempt_id"])
        model_id = str(attempt["model_id"])
        if model_id not in EXPECTED_MODEL_IDS or model_id in by_model:
            raise PilotEvaluationAmendmentError(
                "model_audits model set differs from the frozen pair"
            )
        by_model[model_id] = attempt
        _sha256_hex(attempt["taskset_sha256"], f"{attempt_id} taskset_sha256")
        if attempt["old_terminal_status"] != "capability-no-go":
            raise PilotEvaluationAmendmentError(
                f"{attempt_id} old terminal status differs"
            )
        _validate_score_map(
            attempt["old_scores"],
            name=f"{attempt_id} old_scores",
            expected_proposal_correct=0,
        )
        source_hashes = _mapping(
            attempt["source_hashes"], f"{attempt_id} source_hashes"
        )
        _exact_keys(
            source_hashes,
            {
                "contract_sha256",
                "capability_file_sha256",
                "gate_receipt_file_sha256",
                "summary_file_sha256",
                "summary_content_sha256",
            },
            f"{attempt_id} source_hashes",
        )
        for key, digest in source_hashes.items():
            _sha256_hex(digest, f"{attempt_id} source_hashes.{key}")
        proposal_rows = _sequence(
            attempt["proposal_rows"], f"{attempt_id} proposal_rows"
        )
        proposal_ids = [
            _validate_proposal_projection(row, attempt_id=attempt_id)
            for row in proposal_rows
        ]
        if (
            len(proposal_ids) != 6
            or tuple(proposal_ids) != EXPECTED_PROPOSAL_TASK_IDS
        ):
            raise PilotEvaluationAmendmentError(
                f"{attempt_id} proposal projection denominator differs"
            )
    if set(by_model) != set(EXPECTED_MODEL_IDS):
        raise PilotEvaluationAmendmentError(
            "model_audits model set differs from the frozen pair"
        )
    return by_model


def _validate_usage_projection_groups(
    value: Any,
    *,
    model_audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    groups = _sequence(value, "usage_projection_rows")
    if len(groups) != 2:
        raise PilotEvaluationAmendmentError(
            "usage_projection_rows requires exactly two source attempts"
        )
    by_model: dict[str, Mapping[str, Any]] = {}
    for item in groups:
        group = _mapping(item, "usage projection group")
        _exact_keys(
            group,
            {"attempt_id", "model_id", "response_model", "rows"},
            "usage projection group",
        )
        model_id = str(group["model_id"])
        if model_id not in model_audits or model_id in by_model:
            raise PilotEvaluationAmendmentError(
                "usage projection model set differs from model_audits"
            )
        audit = model_audits[model_id]
        if (
            group["attempt_id"] != audit["attempt_id"]
            or group["response_model"] != audit["response_model"]
        ):
            raise PilotEvaluationAmendmentError(
                f"{model_id} usage projection identity differs"
            )
        attempt_id = str(group["attempt_id"])
        usage_rows = _sequence(group["rows"], f"{attempt_id} usage rows")
        usage_ids = [
            _validate_usage_projection(
                row,
                attempt_id=attempt_id,
                expected_model=str(group["response_model"]),
            )
            for row in usage_rows
        ]
        if len(usage_ids) != 30 or tuple(usage_ids) != EXPECTED_TASK_IDS:
            raise PilotEvaluationAmendmentError(
                f"{attempt_id} usage projection denominator differs"
            )
        by_model[model_id] = group
    if set(by_model) != set(EXPECTED_MODEL_IDS):
        raise PilotEvaluationAmendmentError(
            "usage projection model set differs from the frozen pair"
        )
    return by_model


def _validate_audit(
    value: Any,
    *,
    amendment: Mapping[str, Any],
    model_audits: Sequence[Any],
    usage_projection_rows: Sequence[Any],
) -> None:
    audit = _mapping(value, "evaluator amendment audit")
    _exact_keys(
        audit,
        {
            "schema_version",
            "provider_calls",
            "source_attempt_count",
            "proposal_projection_count",
            "usage_projection_count",
            "evaluator_amendment_sha256",
            "model_audits_sha256",
            "usage_projection_rows_sha256",
        },
        "evaluator amendment audit",
    )
    if (
        audit["schema_version"] != "finevo-pilot-evaluator-amendment-audit-v1"
        or audit["provider_calls"] != 0
        or audit["source_attempt_count"] != 2
        or audit["proposal_projection_count"] != 12
        or audit["usage_projection_count"] != 60
        or audit["evaluator_amendment_sha256"] != canonical_sha256(amendment)
        or audit["model_audits_sha256"] != canonical_sha256(model_audits)
        or audit["usage_projection_rows_sha256"]
        != canonical_sha256(usage_projection_rows)
    ):
        raise PilotEvaluationAmendmentError(
            "evaluator amendment audit summary differs"
        )


def _validate_source_bindings(
    *,
    amendment: Mapping[str, Any],
    model_audits: Mapping[str, Mapping[str, Any]],
) -> None:
    sources = _sequence(amendment["source_attempts"], "source_attempts")
    if len(sources) != 2:
        raise PilotEvaluationAmendmentError(
            "source_attempts must contain exactly two attempts"
        )
    by_model: dict[str, Mapping[str, Any]] = {}
    for value in sources:
        source = _mapping(value, "source attempt")
        model_id = str(source.get("model_id"))
        if model_id not in EXPECTED_MODEL_IDS or model_id in by_model:
            raise PilotEvaluationAmendmentError(
                "source_attempts model set differs from the frozen pair"
            )
        by_model[model_id] = source
    if set(by_model) != set(model_audits):
        raise PilotEvaluationAmendmentError(
            "source_attempts and model_audits model sets differ"
        )
    for model_id, source in by_model.items():
        audit = model_audits[model_id]
        hashes = _mapping(audit["source_hashes"], f"{model_id} source_hashes")
        expected_hashes = {
            "contract_sha256": source.get("contract_sha256"),
            "capability_file_sha256": source.get("capability_sha256"),
            "gate_receipt_file_sha256": source.get("gate_sha256"),
            "summary_file_sha256": source.get("terminal_sha256"),
            "summary_content_sha256": source.get("terminal_content_sha256"),
        }
        if dict(hashes) != expected_hashes:
            raise PilotEvaluationAmendmentError(
                f"{model_id} source hashes differ from the compact amendment"
            )
        source_old_scores = {
            str(category).replace("_", "-"): {
                **dict(_mapping(score, f"{model_id} source old score")),
                "required": 5 if category == "rule_proposal" else 10,
            }
            for category, score in _mapping(
                source.get("old_scores"), f"{model_id} source old_scores"
            ).items()
        }
        if (
            audit["old_terminal_status"] != source.get("old_status")
            or audit["old_scores"] != source_old_scores
        ):
            raise PilotEvaluationAmendmentError(
                f"{model_id} old diagnostic differs from its source attempt"
            )


def validate_evaluator_amendment_mapping(value: Mapping[str, Any]) -> None:
    amendment = _mapping(value, "evaluator_amendment")
    _exact_keys(
        amendment,
        {
            "schema_version",
            "amendment_id",
            "parent",
            "source_attempts",
            "defect",
            "rescore_policy",
            "corrected_results",
            "budget_carry_forward",
        },
        "evaluator_amendment",
    )
    if (
        amendment["schema_version"] != EVALUATOR_AMENDMENT_SCHEMA_VERSION
        or amendment["amendment_id"] != EVALUATOR_AMENDMENT_ID
    ):
        raise PilotEvaluationAmendmentError(
            "unsupported evaluator amendment identity"
        )
    if canonical_sha256(amendment) != EXPECTED_EVALUATOR_AMENDMENT_SHA256:
        raise PilotEvaluationAmendmentError(
            "evaluator amendment differs from the frozen mapping"
        )

    policy = _mapping(amendment["rescore_policy"], "rescore_policy")
    if (
        policy.get("provider_calls") != 0
        or policy.get("scientific_effect_outcomes_inspected") is not False
        or policy.get("capability_outcomes_inspected") is not True
        or policy.get("failed_seed_replacement") != "forbidden"
        or policy.get("semantic_candidate_acceptance_required") is not True
        or policy.get("semantic_match_required") is not False
        or policy.get("provider_redispatch") != "forbidden"
    ):
        raise PilotEvaluationAmendmentError(
            "evaluator amendment inspection or zero-call policy differs"
        )
    defect = _mapping(amendment["defect"], "defect")
    if (
        defect.get("code")
        != "hidden-exact-expected-match-not-preregistered"
        or defect.get("semantic_match_disposition") != "diagnostic-only"
        or defect.get("source_candidate_payload_retained") is not False
        or defect.get("independent_candidate_replay_available") is not False
    ):
        raise PilotEvaluationAmendmentError(
            "evaluator amendment defect scope differs"
        )

    corrected = _sequence(amendment["corrected_results"], "corrected_results")
    if len(corrected) != 2:
        raise PilotEvaluationAmendmentError(
            "corrected_results must contain exactly two models"
        )
    corrected_models: set[str] = set()
    for item in corrected:
        result = _mapping(item, "corrected result")
        model_id = str(result.get("model_id"))
        if model_id not in EXPECTED_MODEL_IDS or model_id in corrected_models:
            raise PilotEvaluationAmendmentError(
                "corrected result model set differs"
            )
        corrected_models.add(model_id)
        scores = _mapping(result.get("scores"), f"{model_id} scores")
        scores_with_required = {
            category: {
                **dict(_mapping(row, f"{model_id} {category}")),
                "required": 5 if category == "rule_proposal" else 10,
            }
            for category, row in scores.items()
        }
        normalized_scores = {
            category.replace("_", "-"): row
            for category, row in scores_with_required.items()
        }
        _validate_score_map(
            normalized_scores,
            name=f"{model_id} corrected_scores",
            expected_proposal_correct=6,
        )
        if result.get("status") != "complete" or result.get("provider_calls") != 0:
            raise PilotEvaluationAmendmentError(
                f"{model_id} corrected result does not pass exactly"
            )

    carry = _mapping(amendment["budget_carry_forward"], "budget_carry_forward")
    parent = _mapping(amendment["parent"], "parent")
    try:
        debit = ParentBudgetDebit(
            parent_contract_sha256=str(carry["source_contract_sha256"]),
            parent_run_ledger_sha256=str(parent["run_ledger_internal_sha256"]),
            parent_budget_ledger_sha256=str(
                parent["budget_ledger_internal_sha256"]
            ),
            stage_bucket=str(carry["source_stage_bucket"]),
            cost_usd=float(carry["cost_usd"]),
            hosted_completions=int(carry["hosted_completions"]),
            storage_bytes=int(carry["storage_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotEvaluationAmendmentError(
            "evaluator amendment parent debit is invalid"
        ) from exc
    if (
        debit.parent_contract_sha256 != V21_CONTRACT_SHA256
        or debit.parent_run_ledger_sha256 != V21_RUN_LEDGER_INTERNAL_SHA256
        or debit.parent_budget_ledger_sha256
        != V21_BUDGET_LEDGER_INTERNAL_SHA256
        or debit.stage_bucket != "capability"
        or not math.isclose(
            debit.cost_usd, 1.53775475, rel_tol=0.0, abs_tol=1e-12
        )
        or debit.hosted_completions != 60
        or debit.storage_bytes != 715_860
        or debit.record_sha256 != V22_PARENT_DEBIT_RECORD_SHA256
    ):
        raise PilotEvaluationAmendmentError(
            "evaluator amendment parent debit differs from V2.1 cumulative spend"
        )


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    contract: PilotContract | Mapping[str, Any] | None,
) -> None:
    receipt = _mapping(value, "evaluator amendment receipt")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "evaluator_amendment",
            "model_audits",
            "usage_projection_rows",
            "audit",
            "limitations",
            "integrity",
        },
        "evaluator amendment receipt",
    )
    if (
        receipt["schema_version"] != EVALUATOR_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != "frozen"
    ):
        raise PilotEvaluationAmendmentError(
            "unsupported or unfrozen evaluator amendment receipt"
        )
    unsigned = dict(receipt)
    integrity = _mapping(unsigned.pop("integrity"), "receipt integrity")
    _exact_keys(
        integrity,
        {"canonicalization", "content_sha256"},
        "receipt integrity",
    )
    actual_content_sha256 = canonical_sha256(unsigned)
    if (
        integrity["canonicalization"] != "json-sort-keys-utf8-v1"
        or integrity["content_sha256"] != actual_content_sha256
        or actual_content_sha256 != EXPECTED_EVALUATOR_RECEIPT_CONTENT_SHA256
    ):
        raise PilotEvaluationAmendmentError(
            "evaluator amendment receipt hash mismatch"
        )
    amendment = _mapping(
        receipt["evaluator_amendment"], "receipt evaluator_amendment"
    )
    validate_evaluator_amendment_mapping(amendment)
    model_audits = _sequence(receipt["model_audits"], "model_audits")
    usage_projection_rows = _sequence(
        receipt["usage_projection_rows"], "usage_projection_rows"
    )
    validated_model_audits = _validate_model_audits(model_audits)
    _validate_source_bindings(
        amendment=amendment,
        model_audits=validated_model_audits,
    )
    _validate_usage_projection_groups(
        usage_projection_rows,
        model_audits=validated_model_audits,
    )
    _validate_audit(
        receipt["audit"],
        amendment=amendment,
        model_audits=model_audits,
        usage_projection_rows=usage_projection_rows,
    )
    limitations = _sequence(receipt["limitations"], "receipt limitations")
    if (
        len(limitations) < 3
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise PilotEvaluationAmendmentError(
            "receipt limitations must state the bounded correction"
        )
    if contract is not None:
        contract_amendment = _amendment_from_contract(contract)
        if contract_amendment is None or dict(contract_amendment) != dict(amendment):
            raise PilotEvaluationAmendmentError(
                "tracked evaluator receipt differs from the loaded contract"
            )


def load_evaluator_amendment_receipt(
    *,
    repo_root: str | Path,
    contract: PilotContract | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    path = (
        Path(repo_root).resolve()
        / EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH
    )
    raw = _read_regular_bytes_once(path, name="tracked evaluator amendment receipt")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_EVALUATOR_RECEIPT_FILE_SHA256:
        raise PilotEvaluationAmendmentError(
            "tracked evaluator amendment receipt file hash mismatch"
        )
    value = _strict_json_bytes(raw, name=str(path))
    _validate_receipt(value, contract=contract)
    return value, path


def evaluator_amendment_control_path(
    *,
    raw_root: str | Path,
) -> Path:
    return Path(raw_root).resolve() / EVALUATOR_CORRECTION_RAW_FILENAME


def persist_evaluator_correction_receipt(
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    contract: PilotContract | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    source = (
        Path(repo_root).resolve()
        / EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH
    )
    raw = _read_regular_bytes_once(source, name="tracked evaluator amendment receipt")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_EVALUATOR_RECEIPT_FILE_SHA256:
        raise PilotEvaluationAmendmentError(
            "tracked evaluator amendment receipt file hash mismatch"
        )
    value = _strict_json_bytes(raw, name=str(source))
    _validate_receipt(value, contract=contract)

    destination = evaluator_amendment_control_path(raw_root=raw_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = _read_regular_bytes_once(
            destination, name="persisted evaluator amendment receipt"
        )
        if existing != raw:
            raise PilotEvaluationAmendmentError(
                "persisted evaluator amendment receipt differs from tracked source"
            )
        return value, destination

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        existing = _read_regular_bytes_once(
            destination, name="persisted evaluator amendment receipt"
        )
        if existing != raw:
            raise PilotEvaluationAmendmentError(
                "persisted evaluator amendment receipt differs from tracked source"
            )
    except OSError as exc:
        raise PilotEvaluationAmendmentError(
            "cannot persist evaluator amendment receipt"
        ) from exc
    return value, destination


def parent_budget_debit_for_evaluator_amendment(
    contract: PilotContract | Mapping[str, Any],
) -> ParentBudgetDebit | None:
    amendment = _amendment_from_contract(contract)
    if amendment is None:
        return None
    validate_evaluator_amendment_mapping(amendment)
    carry = _mapping(amendment["budget_carry_forward"], "budget_carry_forward")
    parent = _mapping(amendment["parent"], "parent")
    try:
        debit = ParentBudgetDebit(
            parent_contract_sha256=str(carry["source_contract_sha256"]),
            parent_run_ledger_sha256=str(parent["run_ledger_internal_sha256"]),
            parent_budget_ledger_sha256=str(
                parent["budget_ledger_internal_sha256"]
            ),
            stage_bucket=str(carry["source_stage_bucket"]),
            cost_usd=float(carry["cost_usd"]),
            hosted_completions=int(carry["hosted_completions"]),
            storage_bytes=int(carry["storage_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover
        raise PilotEvaluationAmendmentError(
            "cannot derive evaluator amendment parent debit"
        ) from exc
    if debit.record_sha256 != V22_PARENT_DEBIT_RECORD_SHA256:
        raise PilotEvaluationAmendmentError(
            "evaluator amendment parent debit record differs"
        )
    return debit


def _audit_attempt_by_model(
    receipt: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    attempts = _sequence(receipt["model_audits"], "model_audits")
    return {
        str(_mapping(item, "source attempt")["model_id"]): _mapping(
            item, "source attempt"
        )
        for item in attempts
    }


def model_import_records(
    contract: PilotContract | Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    _validate_receipt(receipt, contract=contract)
    amendment = _mapping(
        receipt["evaluator_amendment"], "receipt evaluator_amendment"
    )
    attempts = _audit_attempt_by_model(receipt)
    sources = {
        str(_mapping(item, "source attempt")["model_id"]): _mapping(
            item, "source attempt"
        )
        for item in _sequence(
            amendment["source_attempts"], "source_attempts"
        )
    }
    usage_groups = {
        str(_mapping(item, "usage projection group")["model_id"]): _mapping(
            item, "usage projection group"
        )
        for item in _sequence(
            receipt["usage_projection_rows"], "usage_projection_rows"
        )
    }
    records: list[dict[str, Any]] = []
    for result_value in _sequence(
        amendment["corrected_results"], "corrected_results"
    ):
        result = _mapping(result_value, "corrected result")
        model_id = str(result["model_id"])
        attempt = attempts[model_id]
        source = sources[model_id]
        usage_group = usage_groups[model_id]
        normalized_scores = {
            category.replace("_", "-"): {
                **dict(_mapping(score, f"{model_id} score")),
                "required": 5 if category == "rule_proposal" else 10,
            }
            for category, score in _mapping(
                result["scores"], f"{model_id} scores"
            ).items()
        }
        checks = {
            category: score["correct"] >= score["required"]
            for category, score in normalized_scores.items()
        }
        records.append(
            {
                "model_id": model_id,
                "source_attempt_id": attempt["attempt_id"],
                "source_run_id": source["run_id"],
                "source_hashes": _json_copy(attempt["source_hashes"]),
                "taskset_sha256": str(attempt["taskset_sha256"]),
                "proposal_rows": _json_copy(attempt["proposal_rows"]),
                "usage_projection_rows": _json_copy(usage_group["rows"]),
                "old_diagnostic": {
                    "terminal_status": attempt["old_terminal_status"],
                    "scores": _json_copy(attempt["old_scores"]),
                    "semantic_match_disposition": "diagnostic-only",
                },
                "corrected_scores": normalized_scores,
                "checks": checks,
                "capability_assessment": {"status": "pass", "pass": True},
                "pass": True,
                "provider_calls_current_attempt": 0,
            }
        )
    return tuple(records)


def _category_totals(scores: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category, score_value in scores.items():
        score = _mapping(score_value, f"{category} score")
        correct = int(score["correct"])
        denominator = int(score["denominator"])
        result[category] = {
            "correct": correct,
            "denominator": denominator,
            "required": int(score["required"]),
            "registered_correct": correct,
            "registered_total": denominator,
            "evaluable_count": denominator,
            "conditional_correct": correct,
            "conditional_accuracy": correct / denominator,
            "interface_failure_count": 0,
        }
    return result


def build_capability_import(
    contract: PilotContract | Mapping[str, Any],
    spec: PilotRunSpec | Any,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    records = {
        record["model_id"]: record
        for record in model_import_records(contract, receipt)
    }
    model_id = str(getattr(spec, "model_id"))
    if model_id not in records:
        raise PilotEvaluationAmendmentError(
            f"no corrected capability import exists for {model_id}"
        )
    execution_mode = str(getattr(spec, "execution_mode"))
    if execution_mode not in {"capability_probe", "closed_loop_preflight"}:
        raise PilotEvaluationAmendmentError(
            "capability import requires a capability/preflight spec"
        )
    contract_value = _contract_mapping(contract)
    contract_id = str(
        getattr(contract, "contract_id", contract_value.get("contract_id"))
    )
    contract_hash = str(
        getattr(contract, "canonical_hash", canonical_sha256(contract_value))
    )
    record = records[model_id]
    scores = _mapping(record["corrected_scores"], "corrected_scores")
    payload: dict[str, Any] = {
        "schema_version": CAPABILITY_IMPORT_SCHEMA_VERSION,
        "contract_id": contract_id,
        "contract_sha256": contract_hash,
        "target_run_id": str(getattr(spec, "run_id")),
        "target_execution_mode": execution_mode,
        "model_id": model_id,
        "source_attempt_id": record["source_attempt_id"],
        "source_run_id": record["source_run_id"],
        "source_hashes": _json_copy(record["source_hashes"]),
        "taskset_sha256": record["taskset_sha256"],
        "proposal_rows": _json_copy(record["proposal_rows"]),
        "usage_projection_rows": _json_copy(record["usage_projection_rows"]),
        "category_totals": _category_totals(scores),
        "checks": _json_copy(record["checks"]),
        "capability_assessment": _json_copy(record["capability_assessment"]),
        "interface_gate": {"pass": True, "failure_count": 0},
        "pass": True,
        "preflight_go": False,
        "preflight_checks": None,
        "provider_calls_current_attempt": 0,
        "scientific_evidence": False,
        "old_diagnostic": _json_copy(record["old_diagnostic"]),
    }
    payload["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(payload),
    }
    return payload


def validate_capability_import(
    value: Mapping[str, Any],
    contract: PilotContract | Mapping[str, Any],
    spec: PilotRunSpec | Any,
    receipt: Mapping[str, Any],
) -> None:
    imported = _mapping(value, "capability import")
    unsigned = _json_copy(imported)
    integrity = _mapping(unsigned.pop("integrity", None), "import integrity")
    _exact_keys(
        integrity,
        {"canonicalization", "content_sha256"},
        "import integrity",
    )
    if (
        integrity["canonicalization"] != "json-sort-keys-utf8-v1"
        or integrity["content_sha256"] != canonical_sha256(unsigned)
    ):
        raise PilotEvaluationAmendmentError(
            "capability import content hash mismatch"
        )
    expected = build_capability_import(contract, spec, receipt)
    if str(getattr(spec, "execution_mode")) == "closed_loop_preflight":
        preflight_go = imported.get("preflight_go")
        preflight_checks = imported.get("preflight_checks")
        if not isinstance(preflight_go, bool):
            raise PilotEvaluationAmendmentError(
                "closed-loop import preflight_go must be boolean"
            )
        if preflight_checks is not None:
            checks = _mapping(preflight_checks, "preflight_checks")
            if preflight_go and (
                not checks or any(item is not True for item in checks.values())
            ):
                raise PilotEvaluationAmendmentError(
                    "preflight_go requires all recorded checks to pass"
                )
        expected["preflight_go"] = preflight_go
        expected["preflight_checks"] = _json_copy(preflight_checks)
        expected_unsigned = dict(expected)
        expected_unsigned.pop("integrity")
        expected["integrity"] = {
            "canonicalization": "json-sort-keys-utf8-v1",
            "content_sha256": canonical_sha256(expected_unsigned),
        }
    if dict(imported) != expected:
        raise PilotEvaluationAmendmentError(
            "capability import differs from the frozen evaluator receipt"
        )


__all__ = [
    "CAPABILITY_IMPORT_SCHEMA_VERSION",
    "EVALUATOR_AMENDMENT_ID",
    "EVALUATOR_AMENDMENT_SCHEMA_VERSION",
    "EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH",
    "EVALUATOR_RECEIPT_SCHEMA_VERSION",
    "EXPECTED_EVALUATOR_RECEIPT_FILE_SHA256",
    "PilotEvaluationAmendmentError",
    "V22_PARENT_DEBIT_RECORD_SHA256",
    "build_capability_import",
    "evaluator_amendment_control_path",
    "load_evaluator_amendment_receipt",
    "model_import_records",
    "parent_budget_debit_for_evaluator_amendment",
    "persist_evaluator_correction_receipt",
    "validate_capability_import",
    "validate_evaluator_amendment_mapping",
]
