"""Pure paired counterfactual replay with treatment-integrity enforcement.

The replay layer never imports a provider or simulator.  A caller supplies one
immutable decision snapshot and an injected completion callable.  Every treatment
uses the same environment, base prompt, context packet, model, decoding parameters,
and action parser.  The only prompt bytes allowed to differ live inside the explicit
memory-bundle delimiters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

from .actions import (
    ACTION_SCHEMA_VERSION,
    ActionDecision,
    ActionParseError,
    parse_direct_action,
)
from .prompts import (
    MEMORY_END,
    MEMORY_START,
    PROMPT_SCHEMA_VERSION,
    SUPPORTED_PROMPT_SCHEMA_VERSIONS,
)


REPLAY_SCHEMA_VERSION = "paired-counterfactual-replay-v2"
SNAPSHOT_SCHEMA_VERSION = "decision-snapshot-v2"
MEMORY_BUNDLE_SCHEMA_VERSION = "memory-bundle-v1"
TREATMENT_ORDER = (
    "matched",
    "no-memory",
    "shuffled",
    "wrong-context",
    "injected-rule",
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch; missing={missing}, extra={extra}")


class ReplayError(RuntimeError):
    """Base class for fail-closed replay errors."""


class ReplayIntegrityError(ReplayError):
    """Raised when a paired comparison changes a protected field."""


@dataclass(frozen=True, slots=True)
class ReplayFailure:
    schema_version: str
    snapshot_id: str
    treatment: str
    error_stage: str
    error_type: str
    message: str
    provider: str
    model: str
    prompt_hash: str
    memory_hash: str
    provider_request_id: str | None
    provider_metadata_json: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["provider_metadata"] = json.loads(
            result.pop("provider_metadata_json")
        )
        return result

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


class ReplayExecutionError(ReplayError):
    """A provider/parser failure carrying a serializable audit record."""

    def __init__(self, failure: ReplayFailure) -> None:
        self.failure = failure
        super().__init__(
            f"{failure.error_stage} failure for {failure.treatment}: "
            f"{failure.error_type}: {failure.message}"
        )


@dataclass(frozen=True, slots=True)
class MemoryBundle:
    schema_version: str
    treatment: str
    bundle_id: str
    text: str
    memory_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported memory bundle schema {self.schema_version!r}")
        if self.treatment not in TREATMENT_ORDER:
            raise ValueError(f"unsupported treatment {self.treatment!r}")
        if not isinstance(self.text, str):
            raise TypeError("memory bundle text must be a string")
        if MEMORY_START in self.text or MEMORY_END in self.text:
            raise ValueError("memory bundle text contains a reserved delimiter")
        if self.treatment == "no-memory" and self.text:
            raise ValueError("the no-memory bundle must be empty")
        if self.treatment != "no-memory" and not self.text.strip():
            raise ValueError(f"the {self.treatment} bundle must not be empty")
        _hash(self.memory_hash, "memory_hash")
        expected_hash = _sha256(self.text)
        if self.memory_hash != expected_hash:
            raise ValueError("memory_hash does not match bundle text")
        expected_id = f"mem-{self.treatment}-{expected_hash[:16]}"
        if self.bundle_id != expected_id:
            raise ValueError("bundle_id does not match treatment and memory_hash")

    @classmethod
    def create(cls, treatment: str, text: str) -> "MemoryBundle":
        if not isinstance(text, str):
            raise TypeError("memory bundle text must be a string")
        digest = _sha256(text)
        return cls(
            schema_version=MEMORY_BUNDLE_SCHEMA_VERSION,
            treatment=treatment,
            bundle_id=f"mem-{treatment}-{digest[:16]}",
            text=text,
            memory_hash=digest,
        )

    def manifest_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "treatment": self.treatment,
            "bundle_id": self.bundle_id,
            "memory_hash": self.memory_hash,
        }


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """All protected inputs and memory treatments for one paired replay."""

    schema_version: str
    snapshot_id: str
    snapshot_hash: str
    prompt_schema_version: str
    source_full_prompt_hash: str
    environment_state_hash: str
    base_prompt: str
    base_prompt_hash: str
    context_packet_id: str
    context_packet_hash: str
    parser_name: str
    parser_schema_version: str
    max_labor_hours: float
    labor_step: float
    consumption_step: float
    provider: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    decoding_seed: int | None
    memory_bundles: tuple[MemoryBundle, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported snapshot schema {self.schema_version!r}")
        if self.prompt_schema_version not in SUPPORTED_PROMPT_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported source prompt schema {self.prompt_schema_version!r}"
            )
        _hash(self.source_full_prompt_hash, "source_full_prompt_hash")
        _hash(self.environment_state_hash, "environment_state_hash")
        if not isinstance(self.base_prompt, str) or not self.base_prompt.strip():
            raise TypeError("base_prompt must be a non-empty string")
        if MEMORY_START in self.base_prompt or MEMORY_END in self.base_prompt:
            raise ValueError("base_prompt contains a reserved memory delimiter")
        _hash(self.base_prompt_hash, "base_prompt_hash")
        if self.base_prompt_hash != _sha256(self.base_prompt):
            raise ValueError("base_prompt_hash does not match base_prompt")
        _nonempty(self.context_packet_id, "context_packet_id")
        _hash(self.context_packet_hash, "context_packet_hash")
        if self.parser_name != "parse_direct_action":
            raise ValueError("paired replay requires the shared parse_direct_action parser")
        if self.parser_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError("parser_schema_version does not match shared action parser")

        for name in ("max_labor_hours", "labor_step", "consumption_step"):
            number = _finite(getattr(self, name), name)
            if number <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, number)
        if int(self.max_labor_hours // self.labor_step) < 1:
            raise ValueError("labor discretization contains no positive action")
        if int(round(1.0 / self.consumption_step)) < 1:
            raise ValueError("consumption discretization contains no positive action")

        _nonempty(self.provider, "provider")
        _nonempty(self.model, "model")
        temperature = _finite(self.temperature, "temperature")
        top_p = _finite(self.top_p, "top_p")
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.decoding_seed is not None and (
            isinstance(self.decoding_seed, bool)
            or not isinstance(self.decoding_seed, int)
        ):
            raise TypeError("decoding_seed must be an integer or None")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)

        bundles = tuple(self.memory_bundles)
        if not all(isinstance(bundle, MemoryBundle) for bundle in bundles):
            raise TypeError("memory_bundles must contain MemoryBundle values")
        names = tuple(bundle.treatment for bundle in bundles)
        if names != TREATMENT_ORDER:
            raise ValueError(
                f"memory bundles must use deterministic order {TREATMENT_ORDER}, got {names}"
            )
        object.__setattr__(self, "memory_bundles", bundles)

        reconstructed_matched_hash = _sha256(self.build_prompt("matched"))
        if reconstructed_matched_hash != self.source_full_prompt_hash:
            raise ValueError(
                "matched replay prompt does not reconstruct source_full_prompt_hash"
            )

        expected_hash = _sha256(_canonical_json(self._integrity_payload()))
        _hash(self.snapshot_hash, "snapshot_hash")
        if self.snapshot_hash != expected_hash:
            raise ValueError("snapshot_hash does not match snapshot contents")
        if self.snapshot_id != f"decision-{expected_hash[:20]}":
            raise ValueError("snapshot_id does not match snapshot_hash")

    @classmethod
    def create(
        cls,
        *,
        environment_state_hash: str,
        base_prompt: str,
        context_packet_id: str,
        context_packet_hash: str,
        provider: str,
        model: str,
        memory_bundles: Mapping[str, str] | Sequence[MemoryBundle],
        max_labor_hours: float = 168.0,
        labor_step: float = 8.0,
        consumption_step: float = 0.02,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 800,
        decoding_seed: int | None = None,
        prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
        source_full_prompt_hash: str | None = None,
    ) -> "DecisionSnapshot":
        if isinstance(memory_bundles, Mapping):
            if set(memory_bundles) != set(TREATMENT_ORDER):
                raise ValueError(
                    "memory_bundles must contain exactly " + ", ".join(TREATMENT_ORDER)
                )
            bundles = tuple(
                MemoryBundle.create(name, memory_bundles[name])
                for name in TREATMENT_ORDER
            )
        else:
            bundles = tuple(memory_bundles)

        if tuple(bundle.treatment for bundle in bundles) != TREATMENT_ORDER:
            raise ValueError(
                f"memory bundles must use deterministic order {TREATMENT_ORDER}"
            )
        matched_text = bundles[TREATMENT_ORDER.index("matched")].text
        reconstructed_matched = (
            f"{base_prompt}\n\n{MEMORY_START}\n{matched_text}\n{MEMORY_END}"
        )
        reconstructed_hash = _sha256(reconstructed_matched)
        bound_source_hash = source_full_prompt_hash or reconstructed_hash

        common = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "prompt_schema_version": prompt_schema_version,
            "source_full_prompt_hash": bound_source_hash,
            "environment_state_hash": environment_state_hash,
            "base_prompt": base_prompt,
            "base_prompt_hash": _sha256(base_prompt),
            "context_packet_id": context_packet_id,
            "context_packet_hash": context_packet_hash,
            "parser_name": "parse_direct_action",
            "parser_schema_version": ACTION_SCHEMA_VERSION,
            "max_labor_hours": float(max_labor_hours),
            "labor_step": float(labor_step),
            "consumption_step": float(consumption_step),
            "provider": provider,
            "model": model,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": max_tokens,
            "decoding_seed": decoding_seed,
            "memory_bundles": bundles,
        }
        payload = cls._payload_from_values(**common)
        digest = _sha256(_canonical_json(payload))
        return cls(
            snapshot_id=f"decision-{digest[:20]}",
            snapshot_hash=digest,
            **common,
        )

    @staticmethod
    def _payload_from_values(**values: Any) -> dict[str, Any]:
        return {
            "schema_version": values["schema_version"],
            "prompt_schema_version": values["prompt_schema_version"],
            "source_full_prompt_hash": values["source_full_prompt_hash"],
            "environment_state_hash": values["environment_state_hash"],
            "base_prompt": values["base_prompt"],
            "base_prompt_hash": values["base_prompt_hash"],
            "context_packet_id": values["context_packet_id"],
            "context_packet_hash": values["context_packet_hash"],
            "parser_name": values["parser_name"],
            "parser_schema_version": values["parser_schema_version"],
            "max_labor_hours": values["max_labor_hours"],
            "labor_step": values["labor_step"],
            "consumption_step": values["consumption_step"],
            "provider": values["provider"],
            "model": values["model"],
            "temperature": values["temperature"],
            "top_p": values["top_p"],
            "max_tokens": values["max_tokens"],
            "decoding_seed": values["decoding_seed"],
            "memory_bundles": [
                bundle.manifest_dict() for bundle in values["memory_bundles"]
            ],
        }

    def _integrity_payload(self) -> dict[str, Any]:
        return self._payload_from_values(
            schema_version=self.schema_version,
            prompt_schema_version=self.prompt_schema_version,
            source_full_prompt_hash=self.source_full_prompt_hash,
            environment_state_hash=self.environment_state_hash,
            base_prompt=self.base_prompt,
            base_prompt_hash=self.base_prompt_hash,
            context_packet_id=self.context_packet_id,
            context_packet_hash=self.context_packet_hash,
            parser_name=self.parser_name,
            parser_schema_version=self.parser_schema_version,
            max_labor_hours=self.max_labor_hours,
            labor_step=self.labor_step,
            consumption_step=self.consumption_step,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            decoding_seed=self.decoding_seed,
            memory_bundles=self.memory_bundles,
        )

    def common_integrity_payload(self) -> dict[str, Any]:
        payload = self._integrity_payload()
        del payload["memory_bundles"]
        return payload

    @property
    def common_integrity_hash(self) -> str:
        return _sha256(_canonical_json(self.common_integrity_payload()))

    def assert_compatible(self, other: "DecisionSnapshot") -> None:
        if not isinstance(other, DecisionSnapshot):
            raise TypeError("other must be a DecisionSnapshot")
        left = self.common_integrity_payload()
        right = other.common_integrity_payload()
        mismatches = sorted(key for key in left if left[key] != right[key])
        if mismatches:
            raise ReplayIntegrityError(
                "paired snapshots differ in protected fields: " + ", ".join(mismatches)
            )

    def bundle(self, treatment: str) -> MemoryBundle:
        if treatment not in TREATMENT_ORDER:
            raise ValueError(f"unsupported treatment {treatment!r}")
        return self.memory_bundles[TREATMENT_ORDER.index(treatment)]

    def build_prompt(self, treatment: str) -> str:
        bundle = self.bundle(treatment)
        return (
            f"{self.base_prompt}\n\n{MEMORY_START}\n"
            f"{bundle.text}\n{MEMORY_END}"
        )

    def build_prompts(self) -> tuple[tuple[str, str], ...]:
        prompts = tuple((name, self.build_prompt(name)) for name in TREATMENT_ORDER)
        self.verify_treatment_integrity(dict(prompts))
        return prompts

    def verify_treatment_integrity(self, prompts: Mapping[str, str]) -> None:
        if set(prompts) != set(TREATMENT_ORDER):
            raise ReplayIntegrityError("prompt set does not match required treatments")
        invariant_forms: list[str] = []
        for treatment in TREATMENT_ORDER:
            prompt = prompts[treatment]
            if not isinstance(prompt, str):
                raise ReplayIntegrityError("replay prompt must be a string")
            if prompt.count(MEMORY_START) != 1 or prompt.count(MEMORY_END) != 1:
                raise ReplayIntegrityError(
                    f"treatment {treatment} has invalid memory delimiters"
                )
            prefix, remainder = prompt.split(MEMORY_START, 1)
            _, suffix = remainder.split(MEMORY_END, 1)
            expected = self.build_prompt(treatment)
            if prompt != expected:
                raise ReplayIntegrityError(
                    f"treatment {treatment} prompt does not match its hash-bound bundle"
                )
            invariant_forms.append(prefix + MEMORY_START + MEMORY_END + suffix)
        if len(set(invariant_forms)) != 1:
            raise ReplayIntegrityError("prompt bytes outside memory bundle differ")

    def manifest_dict(self) -> dict[str, Any]:
        payload = self._integrity_payload()
        payload.pop("base_prompt")
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "common_integrity_hash": self.common_integrity_hash,
            **payload,
        }


def verify_compatible_snapshots(snapshots: Sequence[DecisionSnapshot]) -> str:
    if not snapshots:
        raise ValueError("at least one DecisionSnapshot is required")
    reference = snapshots[0]
    if not isinstance(reference, DecisionSnapshot):
        raise TypeError("snapshots must contain DecisionSnapshot values")
    for snapshot in snapshots[1:]:
        reference.assert_compatible(snapshot)
    return reference.common_integrity_hash


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    snapshot_id: str
    treatment: str
    prompt: str
    prompt_hash: str
    memory_bundle_id: str
    memory_hash: str
    provider: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    decoding_seed: int | None


@dataclass(frozen=True, slots=True)
class ProviderCompletion:
    content: str
    provider: str
    model: str
    request_id: str | None = None
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("completion content must be a string")
        _nonempty(self.provider, "completion provider")
        _nonempty(self.model, "completion model")
        if self.request_id is not None and not isinstance(self.request_id, str):
            raise TypeError("request_id must be a string or None")
        if not isinstance(self.metadata_json, str):
            raise TypeError("metadata_json must be a string")
        try:
            decoded = json.loads(self.metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError("metadata_json root must be an object")
        canonical = _canonical_json(decoded)
        object.__setattr__(self, "metadata_json", canonical)

    @classmethod
    def create(
        cls,
        content: str,
        *,
        provider: str,
        model: str,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ProviderCompletion":
        return cls(
            content=content,
            provider=provider,
            model=model,
            request_id=request_id,
            metadata_json=_canonical_json(dict(metadata or {})),
        )

    def metadata(self) -> dict[str, Any]:
        return dict(json.loads(self.metadata_json))


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    schema_version: str
    snapshot_id: str
    snapshot_hash: str
    common_integrity_hash: str
    prompt_schema_version: str
    source_full_prompt_hash: str
    environment_state_hash: str
    base_prompt_hash: str
    context_packet_id: str
    context_packet_hash: str
    treatment_index: int
    treatment: str
    prompt_hash: str
    memory_bundle_id: str
    memory_hash: str
    provider: str
    model: str
    provider_request_id: str | None
    provider_metadata_json: str
    raw_output: str
    raw_output_hash: str
    action: ActionDecision
    proposed_work_delta_vs_matched: float
    proposed_consumption_delta_vs_matched: float
    labor_action_index_delta_vs_matched: int
    executed_labor_hours_delta_vs_matched: float
    consumption_action_index_delta_vs_matched: int
    executed_consumption_rate_delta_vs_matched: float
    integrity_verified: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["provider_metadata"] = json.loads(result.pop("provider_metadata_json"))
        return result

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class PairedReplayResult:
    snapshot: DecisionSnapshot
    records: tuple[ReplayRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, DecisionSnapshot):
            raise TypeError("snapshot must be a DecisionSnapshot")
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        self.validate_integrity()

    def validate_integrity(self) -> None:
        """Recompute every record binding before serialization or interpretation."""

        if not all(isinstance(record, ReplayRecord) for record in self.records):
            raise TypeError("records must contain ReplayRecord values")
        if tuple(record.treatment for record in self.records) != TREATMENT_ORDER:
            raise ReplayIntegrityError("records are not in deterministic treatment order")
        matched = self.records[0].action
        protected = {
            "snapshot_id": self.snapshot.snapshot_id,
            "snapshot_hash": self.snapshot.snapshot_hash,
            "common_integrity_hash": self.snapshot.common_integrity_hash,
            "prompt_schema_version": self.snapshot.prompt_schema_version,
            "source_full_prompt_hash": self.snapshot.source_full_prompt_hash,
            "environment_state_hash": self.snapshot.environment_state_hash,
            "base_prompt_hash": self.snapshot.base_prompt_hash,
            "context_packet_id": self.snapshot.context_packet_id,
            "context_packet_hash": self.snapshot.context_packet_hash,
            "provider": self.snapshot.provider,
            "model": self.snapshot.model,
        }
        for index, record in enumerate(self.records):
            treatment = TREATMENT_ORDER[index]
            bundle = self.snapshot.bundle(treatment)
            if record.schema_version != REPLAY_SCHEMA_VERSION:
                raise ReplayIntegrityError("record schema does not match replay schema")
            mismatches = [
                field
                for field, expected in protected.items()
                if getattr(record, field) != expected
            ]
            if mismatches:
                raise ReplayIntegrityError(
                    "record differs from protected snapshot fields: "
                    + ", ".join(mismatches)
                )
            if record.treatment_index != index:
                raise ReplayIntegrityError("record treatment_index is not deterministic")
            if (
                record.memory_bundle_id != bundle.bundle_id
                or record.memory_hash != bundle.memory_hash
            ):
                raise ReplayIntegrityError(
                    "record memory identity does not match its snapshot bundle"
                )
            expected_prompt_hash = _sha256(self.snapshot.build_prompt(treatment))
            if record.prompt_hash != expected_prompt_hash:
                raise ReplayIntegrityError(
                    "record prompt_hash does not match its reconstructed prompt"
                )
            if not isinstance(record.action, ActionDecision):
                raise TypeError("record action must be an ActionDecision")
            action = record.action
            if action.schema_version != ACTION_SCHEMA_VERSION:
                raise ReplayIntegrityError("record action schema is unsupported")
            work = _finite(action.proposed_work_fraction, "proposed_work_fraction")
            consumption = _finite(
                action.proposed_consumption_fraction,
                "proposed_consumption_fraction",
            )
            if not 0.0 <= work <= 1.0 or not 0.0 <= consumption <= 1.0:
                raise ReplayIntegrityError("record action fractions must lie in [0, 1]")
            max_labor_index = int(
                self.snapshot.max_labor_hours // self.snapshot.labor_step
            )
            max_consumption_index = int(round(1.0 / self.snapshot.consumption_step))
            expected_labor_index = min(
                max_labor_index, int(math.floor(work * max_labor_index + 0.5))
            )
            expected_consumption_index = min(
                max_consumption_index,
                int(
                    math.floor(
                        consumption / self.snapshot.consumption_step + 0.5
                    )
                ),
            )
            if (
                isinstance(action.labor_action_index, bool)
                or not isinstance(action.labor_action_index, int)
                or action.labor_action_index != expected_labor_index
                or isinstance(action.consumption_action_index, bool)
                or not isinstance(action.consumption_action_index, int)
                or action.consumption_action_index != expected_consumption_index
                or not math.isclose(
                    _finite(action.executed_labor_hours, "executed_labor_hours"),
                    expected_labor_index * self.snapshot.labor_step,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    _finite(
                        action.executed_consumption_rate,
                        "executed_consumption_rate",
                    ),
                    expected_consumption_index * self.snapshot.consumption_step,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ReplayIntegrityError(
                    "record action does not match the protected discretization"
                )
            if (
                not isinstance(action.reflection, str)
                or isinstance(action.repair_attempts, bool)
                or not isinstance(action.repair_attempts, int)
                or action.repair_attempts < 0
                or not isinstance(action.clipped, bool)
            ):
                raise ReplayIntegrityError("record action audit fields are invalid")
            if not isinstance(record.raw_output, str):
                raise TypeError("record raw_output must be a string")
            _hash(record.raw_output_hash, "record.raw_output_hash")
            if (
                record.raw_output_hash != _sha256(record.raw_output)
                or record.raw_output_hash != record.action.raw_output_hash
            ):
                raise ReplayIntegrityError(
                    "record raw_output_hash does not match raw output and parsed action"
                )
            try:
                reparsed_action = parse_direct_action(
                    record.raw_output,
                    max_labor_hours=self.snapshot.max_labor_hours,
                    labor_step=self.snapshot.labor_step,
                    consumption_step=self.snapshot.consumption_step,
                )
            except (TypeError, ValueError) as exc:
                raise ReplayIntegrityError(
                    "record raw output cannot be reparsed under the protected grid"
                ) from exc
            if reparsed_action != record.action:
                raise ReplayIntegrityError(
                    "record action does not equal the reparsed raw output"
                )
            if record.provider_request_id is not None and not isinstance(
                record.provider_request_id, str
            ):
                raise TypeError("provider_request_id must be a string or None")
            if not isinstance(record.provider_metadata_json, str):
                raise TypeError("provider_metadata_json must be a string")
            try:
                metadata = json.loads(record.provider_metadata_json)
            except json.JSONDecodeError as exc:
                raise ReplayIntegrityError("provider metadata is not valid JSON") from exc
            if not isinstance(metadata, Mapping):
                raise ReplayIntegrityError("provider metadata root must be an object")
            if record.provider_metadata_json != _canonical_json(metadata):
                raise ReplayIntegrityError("provider metadata is not canonical JSON")
            if record.integrity_verified is not True:
                raise ReplayIntegrityError("result contains an unverified treatment")

            float_deltas = {
                "proposed_work_delta_vs_matched": (
                    record.action.proposed_work_fraction
                    - matched.proposed_work_fraction
                ),
                "proposed_consumption_delta_vs_matched": (
                    record.action.proposed_consumption_fraction
                    - matched.proposed_consumption_fraction
                ),
                "executed_labor_hours_delta_vs_matched": (
                    record.action.executed_labor_hours
                    - matched.executed_labor_hours
                ),
                "executed_consumption_rate_delta_vs_matched": (
                    record.action.executed_consumption_rate
                    - matched.executed_consumption_rate
                ),
            }
            for field, expected in float_deltas.items():
                actual = _finite(getattr(record, field), field)
                if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ReplayIntegrityError(
                        f"record {field} does not match the matched-action delta"
                    )
            integer_deltas = {
                "labor_action_index_delta_vs_matched": (
                    record.action.labor_action_index - matched.labor_action_index
                ),
                "consumption_action_index_delta_vs_matched": (
                    record.action.consumption_action_index
                    - matched.consumption_action_index
                ),
            }
            for field, expected in integer_deltas.items():
                actual = getattr(record, field)
                if isinstance(actual, bool) or not isinstance(actual, int):
                    raise TypeError(f"{field} must be an integer")
                if actual != expected:
                    raise ReplayIntegrityError(
                        f"record {field} does not match the matched-action delta"
                    )

    def validate_provider_metadata(self) -> None:
        """Require one applied seed and served-model identity before sealing."""

        self.validate_integrity()
        if self.snapshot.decoding_seed is None:
            raise ReplayIntegrityError(
                "a sealed paired replay requires an explicit decoding seed"
            )
        metadata = [json.loads(record.provider_metadata_json) for record in self.records]
        if any(
            row.get("decoding_seed_applied") != self.snapshot.decoding_seed
            for row in metadata
        ):
            raise ReplayIntegrityError(
                "provider metadata does not confirm the protected decoding seed"
            )
        response_models = {row.get("response_model") for row in metadata}
        if (
            len(response_models) != 1
            or None in response_models
            or not all(isinstance(value, str) and value for value in response_models)
        ):
            raise ReplayIntegrityError(
                "provider metadata does not report one non-empty served model"
            )
        fingerprints = {
            row.get("system_fingerprint")
            for row in metadata
            if row.get("system_fingerprint") is not None
        }
        if len(fingerprints) > 1:
            raise ReplayIntegrityError(
                "provider metadata reports multiple system fingerprints"
            )
        price_bindings = {
            (
                row.get("request_price_snapshot_source"),
                row.get("request_price_snapshot_captured_at"),
            )
            for row in metadata
        }
        if len(price_bindings) != 1:
            raise ReplayIntegrityError(
                "provider metadata reports multiple price snapshots"
            )
        price_source, price_captured_at = next(iter(price_bindings))
        if (price_source is None) != (price_captured_at is None):
            raise ReplayIntegrityError(
                "provider metadata has an incomplete price snapshot"
            )
        if any(row.get("request_profile_id") is not None for row in metadata) and (
            not isinstance(price_source, str)
            or not price_source
            or not isinstance(price_captured_at, str)
            or not price_captured_at
        ):
            raise ReplayIntegrityError(
                "strict provider replay lacks a frozen price snapshot"
            )

    def to_jsonl(self) -> str:
        self.validate_integrity()
        return "".join(f"{record.to_json()}\n" for record in self.records)

    def manifest_dict(self) -> dict[str, Any]:
        self.validate_integrity()
        core = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "snapshot": self.snapshot.manifest_dict(),
            "treatment_order": list(TREATMENT_ORDER),
            "record_count": len(self.records),
            "record_hashes": [_sha256(record.to_json()) for record in self.records],
            "prompt_hashes": {
                record.treatment: record.prompt_hash for record in self.records
            },
            "memory_hashes": {
                record.treatment: record.memory_hash for record in self.records
            },
            "integrity_verified": True,
            "failure_count": 0,
        }
        return {**core, "manifest_hash": _sha256(_canonical_json(core))}

    def manifest_json(self) -> str:
        return _canonical_json(self.manifest_dict())


CompletionCallable = Callable[[ReplayRequest], ProviderCompletion | str]


class PairedReplayRunner:
    """Run all memory interventions through one injected completion callable."""

    def __init__(self, completion: CompletionCallable) -> None:
        if not callable(completion):
            raise TypeError("completion must be callable")
        self._completion = completion

    @staticmethod
    def _failure(
        snapshot: DecisionSnapshot,
        treatment: str,
        stage: str,
        exc: BaseException,
        prompt_hash: str,
        completion: ProviderCompletion | None = None,
    ) -> ReplayExecutionError:
        message = str(exc).strip() or exc.__class__.__name__
        return ReplayExecutionError(
            ReplayFailure(
                schema_version=REPLAY_SCHEMA_VERSION,
                snapshot_id=snapshot.snapshot_id,
                treatment=treatment,
                error_stage=stage,
                error_type=exc.__class__.__name__,
                message=message,
                provider=snapshot.provider,
                model=snapshot.model,
                prompt_hash=prompt_hash,
                memory_hash=snapshot.bundle(treatment).memory_hash,
                provider_request_id=(completion.request_id if completion else None),
                provider_metadata_json=(
                    completion.metadata_json if completion else "{}"
                ),
            )
        )

    def run(self, snapshot: DecisionSnapshot) -> PairedReplayResult:
        if not isinstance(snapshot, DecisionSnapshot):
            raise TypeError("snapshot must be a DecisionSnapshot")
        prompt_pairs = snapshot.build_prompts()
        prompts = dict(prompt_pairs)
        snapshot.verify_treatment_integrity(prompts)

        completions: dict[str, ProviderCompletion] = {}
        decisions: dict[str, ActionDecision] = {}
        prompt_hashes: dict[str, str] = {}
        for treatment in TREATMENT_ORDER:
            prompt = prompts[treatment]
            bundle = snapshot.bundle(treatment)
            prompt_hash = _sha256(prompt)
            prompt_hashes[treatment] = prompt_hash
            request = ReplayRequest(
                snapshot_id=snapshot.snapshot_id,
                treatment=treatment,
                prompt=prompt,
                prompt_hash=prompt_hash,
                memory_bundle_id=bundle.bundle_id,
                memory_hash=bundle.memory_hash,
                provider=snapshot.provider,
                model=snapshot.model,
                temperature=snapshot.temperature,
                top_p=snapshot.top_p,
                max_tokens=snapshot.max_tokens,
                decoding_seed=snapshot.decoding_seed,
            )
            try:
                response = self._completion(request)
                if isinstance(response, str):
                    completion = ProviderCompletion.create(
                        response, provider=snapshot.provider, model=snapshot.model
                    )
                elif isinstance(response, ProviderCompletion):
                    completion = response
                else:
                    raise TypeError(
                        "completion callable must return ProviderCompletion or str"
                    )
            except ReplayExecutionError:
                raise
            except Exception as exc:
                raise self._failure(
                    snapshot, treatment, "provider", exc, prompt_hash
                ) from exc

            if completion.provider != snapshot.provider or completion.model != snapshot.model:
                raise ReplayIntegrityError(
                    "completion provider/model differs from protected replay settings"
                )
            try:
                decision = parse_direct_action(
                    completion.content,
                    max_labor_hours=snapshot.max_labor_hours,
                    labor_step=snapshot.labor_step,
                    consumption_step=snapshot.consumption_step,
                )
            except ActionParseError as exc:
                raise self._failure(
                    snapshot, treatment, "parser", exc, prompt_hash, completion
                ) from exc
            except Exception as exc:
                raise self._failure(
                    snapshot, treatment, "parser", exc, prompt_hash, completion
                ) from exc
            completions[treatment] = completion
            decisions[treatment] = decision

        matched = decisions["matched"]
        records = []
        for index, treatment in enumerate(TREATMENT_ORDER):
            bundle = snapshot.bundle(treatment)
            completion = completions[treatment]
            decision = decisions[treatment]
            records.append(
                ReplayRecord(
                    schema_version=REPLAY_SCHEMA_VERSION,
                    snapshot_id=snapshot.snapshot_id,
                    snapshot_hash=snapshot.snapshot_hash,
                    common_integrity_hash=snapshot.common_integrity_hash,
                    prompt_schema_version=snapshot.prompt_schema_version,
                    source_full_prompt_hash=snapshot.source_full_prompt_hash,
                    environment_state_hash=snapshot.environment_state_hash,
                    base_prompt_hash=snapshot.base_prompt_hash,
                    context_packet_id=snapshot.context_packet_id,
                    context_packet_hash=snapshot.context_packet_hash,
                    treatment_index=index,
                    treatment=treatment,
                    prompt_hash=prompt_hashes[treatment],
                    memory_bundle_id=bundle.bundle_id,
                    memory_hash=bundle.memory_hash,
                    provider=completion.provider,
                    model=completion.model,
                    provider_request_id=completion.request_id,
                    provider_metadata_json=completion.metadata_json,
                    raw_output=completion.content,
                    raw_output_hash=decision.raw_output_hash,
                    action=decision,
                    proposed_work_delta_vs_matched=(
                        decision.proposed_work_fraction
                        - matched.proposed_work_fraction
                    ),
                    proposed_consumption_delta_vs_matched=(
                        decision.proposed_consumption_fraction
                        - matched.proposed_consumption_fraction
                    ),
                    labor_action_index_delta_vs_matched=(
                        decision.labor_action_index - matched.labor_action_index
                    ),
                    executed_labor_hours_delta_vs_matched=(
                        decision.executed_labor_hours - matched.executed_labor_hours
                    ),
                    consumption_action_index_delta_vs_matched=(
                        decision.consumption_action_index
                        - matched.consumption_action_index
                    ),
                    executed_consumption_rate_delta_vs_matched=(
                        decision.executed_consumption_rate
                        - matched.executed_consumption_rate
                    ),
                    integrity_verified=True,
                )
            )
        return PairedReplayResult(snapshot=snapshot, records=tuple(records))


__all__ = [
    "DecisionSnapshot",
    "MEMORY_END",
    "MEMORY_START",
    "MemoryBundle",
    "PairedReplayResult",
    "PairedReplayRunner",
    "ProviderCompletion",
    "REPLAY_SCHEMA_VERSION",
    "ReplayError",
    "ReplayExecutionError",
    "ReplayFailure",
    "ReplayIntegrityError",
    "ReplayRecord",
    "ReplayRequest",
    "TREATMENT_ORDER",
    "verify_compatible_snapshots",
]
