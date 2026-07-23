"""M1: causal context construction and prompt/retrieval channel routing.

The router deliberately has no fitting code.  It either exposes deterministic
rolling features directly or applies an already-frozen linear projection whose
schema and training provenance are validated when it is loaded.  Every input
observation must be timestamped at or before ``observed_through``; accidental
future rows are rejected rather than silently filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CONTEXT_SCHEMA_VERSION = "m1-context-packet-v2"
FEATURE_SCHEMA_VERSION = "m1-causal-rolling-v2"
PROJECTION_SCHEMA_VERSION = "m1-frozen-linear-projection-v1"
ROUTER_SCHEMA_VERSION = "m1-context-router-v2"
ROUTE_SCHEMA_VERSION = "m1-context-route-v2"

CONTEXT_MODES = frozenset(
    {"no-context", "prompt-only", "retrieval-only", "full"}
)

DEFAULT_BASE_FEATURES = (
    "price",
    "interest_rate",
    "unemployment_rate",
    "inflation",
    "sentiment",
)

_TRAINING_METADATA_TYPES = {
    "training_run_id": str,
    "training_seeds": list,
    "target_definition": str,
    "fit_method": str,
    "uses_future_inputs": bool,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_keys(
    value: Mapping[str, Any], expected: set[str], object_name: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"invalid {object_name} keys: {', '.join(details)}")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _timestamp(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer timestamp")
    return value


def _string_tuple(values: Sequence[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of feature names")
    result = tuple(_nonempty_string(item, f"{name} item") for item in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate names")
    return result


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized not in CONTEXT_MODES:
        raise ValueError(
            f"unknown context mode {mode!r}; expected one of {sorted(CONTEXT_MODES)}"
        )
    return normalized


@dataclass(frozen=True)
class FrozenLinearProjection:
    """An inference-only linear projection with auditable training metadata."""

    projection_id: str
    input_features: tuple[str, ...]
    output_features: tuple[str, ...]
    feature_config: Mapping[str, Any]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    training_metadata: Mapping[str, Any]
    schema_version: str = PROJECTION_SCHEMA_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROJECTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported projection schema_version {self.schema_version!r}"
            )
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "projection feature_schema_version does not match "
                f"{FEATURE_SCHEMA_VERSION!r}"
            )
        _nonempty_string(self.projection_id, "projection_id")
        input_features = _string_tuple(self.input_features, "input_features")
        output_features = _string_tuple(self.output_features, "output_features")
        object.__setattr__(self, "input_features", input_features)
        object.__setattr__(self, "output_features", output_features)

        feature_config = dict(self.feature_config)
        _strict_keys(
            feature_config,
            {"window_size", "base_feature_names", "event_feature_names"},
            "projection feature_config",
        )
        if (
            isinstance(feature_config["window_size"], bool)
            or not isinstance(feature_config["window_size"], int)
            or feature_config["window_size"] < 1
        ):
            raise ValueError("feature_config.window_size must be a positive integer")
        base_names = _string_tuple(
            feature_config["base_feature_names"],
            "feature_config.base_feature_names",
        )
        event_values = feature_config["event_feature_names"]
        if isinstance(event_values, (str, bytes)):
            raise ValueError("feature_config.event_feature_names must be a sequence")
        event_names = tuple(
            _nonempty_string(name, "feature_config.event_feature_names item")
            for name in event_values
        )
        if len(set(event_names)) != len(event_names):
            raise ValueError("feature_config.event_feature_names contains duplicates")
        feature_config["base_feature_names"] = list(base_names)
        feature_config["event_feature_names"] = list(event_names)
        object.__setattr__(self, "feature_config", feature_config)

        if len(self.weights) != len(output_features):
            raise ValueError("weights must have one row per output feature")
        normalized_weights = []
        for row_index, row in enumerate(self.weights):
            if len(row) != len(input_features):
                raise ValueError(
                    f"weights[{row_index}] must have {len(input_features)} columns"
                )
            normalized_weights.append(
                tuple(
                    _finite_float(value, f"weights[{row_index}][{column_index}]")
                    for column_index, value in enumerate(row)
                )
            )
        object.__setattr__(self, "weights", tuple(normalized_weights))

        if len(self.bias) != len(output_features):
            raise ValueError("bias must have one value per output feature")
        object.__setattr__(
            self,
            "bias",
            tuple(
                _finite_float(value, f"bias[{index}]")
                for index, value in enumerate(self.bias)
            ),
        )

        metadata = dict(self.training_metadata)
        for key, expected_type in _TRAINING_METADATA_TYPES.items():
            if key not in metadata:
                raise ValueError(f"training_metadata is missing {key!r}")
            if not isinstance(metadata[key], expected_type):
                raise ValueError(
                    f"training_metadata[{key!r}] must be {expected_type.__name__}"
                )
        _nonempty_string(metadata["training_run_id"], "training_run_id")
        _nonempty_string(metadata["target_definition"], "target_definition")
        _nonempty_string(metadata["fit_method"], "fit_method")
        if not metadata["training_seeds"]:
            raise ValueError("training_metadata['training_seeds'] must not be empty")
        for seed in metadata["training_seeds"]:
            if isinstance(seed, bool) or not isinstance(seed, (int, str)):
                raise ValueError("training_seeds must contain only integer or string IDs")
        if metadata["uses_future_inputs"] is not False:
            raise ValueError("a causal projection must declare uses_future_inputs=false")
        try:
            _canonical_json(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("training_metadata must be finite JSON data") from exc
        object.__setattr__(self, "training_metadata", metadata)

    @property
    def projection_hash(self) -> str:
        return _sha256(self.to_dict())

    def transform(self, features: Sequence[float]) -> tuple[float, ...]:
        if len(features) != len(self.input_features):
            raise ValueError(
                f"projection expected {len(self.input_features)} features, "
                f"received {len(features)}"
            )
        vector = tuple(
            _finite_float(value, f"projection input[{index}]")
            for index, value in enumerate(features)
        )
        return tuple(
            self.bias[row_index]
            + sum(weight * value for weight, value in zip(row, vector))
            for row_index, row in enumerate(self.weights)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_schema_version": self.feature_schema_version,
            "projection_id": self.projection_id,
            "input_features": list(self.input_features),
            "output_features": list(self.output_features),
            "feature_config": dict(self.feature_config),
            "weights": [list(row) for row in self.weights],
            "bias": list(self.bias),
            "training_metadata": dict(self.training_metadata),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        expected_input_features: Optional[Sequence[str]] = None,
        expected_feature_config: Optional[Mapping[str, Any]] = None,
    ) -> "FrozenLinearProjection":
        expected_keys = {
            "schema_version",
            "feature_schema_version",
            "projection_id",
            "input_features",
            "output_features",
            "feature_config",
            "weights",
            "bias",
            "training_metadata",
        }
        _strict_keys(value, expected_keys, "projection")
        projection = cls(
            schema_version=value["schema_version"],
            feature_schema_version=value["feature_schema_version"],
            projection_id=value["projection_id"],
            input_features=tuple(value["input_features"]),
            output_features=tuple(value["output_features"]),
            feature_config=dict(value["feature_config"]),
            weights=tuple(tuple(row) for row in value["weights"]),
            bias=tuple(value["bias"]),
            training_metadata=dict(value["training_metadata"]),
        )
        if expected_input_features is not None:
            expected = tuple(expected_input_features)
            if projection.input_features != expected:
                raise ValueError(
                    "projection input feature schema/order mismatch: "
                    f"expected {expected!r}, found {projection.input_features!r}"
                )
        if expected_feature_config is not None:
            expected_config = {
                "window_size": expected_feature_config["window_size"],
                "base_feature_names": list(
                    expected_feature_config["base_feature_names"]
                ),
                "event_feature_names": list(
                    expected_feature_config["event_feature_names"]
                ),
            }
            if projection.feature_config != expected_config:
                raise ValueError(
                    "projection rolling feature configuration mismatch: "
                    f"expected {expected_config!r}, "
                    f"found {projection.feature_config!r}"
                )
        return projection

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        *,
        expected_input_features: Optional[Sequence[str]] = None,
        expected_feature_config: Optional[Mapping[str, Any]] = None,
    ) -> "FrozenLinearProjection":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise ValueError("projection JSON root must be an object")
        return cls.from_dict(
            value,
            expected_input_features=expected_input_features,
            expected_feature_config=expected_feature_config,
        )


@dataclass(frozen=True)
class ContextPacket:
    """Immutable, hash-addressed context produced from causally available inputs."""

    context_id: str
    context_hash: str
    source_hash: str
    decision_t: int
    observed_through: int
    history_start: int
    observation_count: int
    feature_names: tuple[str, ...]
    raw_features: tuple[float, ...]
    vector_feature_names: tuple[str, ...]
    context_vector: tuple[float, ...]
    encoder_version: str
    prompt_summary: str
    projection_metadata: Optional[Mapping[str, Any]] = None
    schema_version: str = CONTEXT_SCHEMA_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTEXT_SCHEMA_VERSION:
            raise ValueError(f"unsupported context schema {self.schema_version!r}")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported feature schema {self.feature_schema_version!r}"
            )
        _timestamp(self.decision_t, "decision_t")
        _timestamp(self.observed_through, "observed_through")
        _timestamp(self.history_start, "history_start")
        if self.observed_through > self.decision_t:
            raise ValueError("observed_through cannot be later than decision_t")
        if self.history_start > self.observed_through:
            raise ValueError("history_start cannot be later than observed_through")
        if isinstance(self.observation_count, bool) or self.observation_count < 1:
            raise ValueError("observation_count must be a positive integer")

        feature_names = _string_tuple(self.feature_names, "feature_names")
        vector_names = _string_tuple(
            self.vector_feature_names, "vector_feature_names"
        )
        if len(feature_names) != len(self.raw_features):
            raise ValueError("feature_names and raw_features lengths differ")
        if len(vector_names) != len(self.context_vector):
            raise ValueError(
                "vector_feature_names and context_vector lengths differ"
            )
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "vector_feature_names", vector_names)
        object.__setattr__(
            self,
            "raw_features",
            tuple(
                _finite_float(value, f"raw_features[{index}]")
                for index, value in enumerate(self.raw_features)
            ),
        )
        object.__setattr__(
            self,
            "context_vector",
            tuple(
                _finite_float(value, f"context_vector[{index}]")
                for index, value in enumerate(self.context_vector)
            ),
        )
        _nonempty_string(self.encoder_version, "encoder_version")
        _nonempty_string(self.prompt_summary, "prompt_summary")
        if self.projection_metadata is not None:
            metadata = dict(self.projection_metadata)
            _canonical_json(metadata)
            object.__setattr__(self, "projection_metadata", metadata)

        expected_hash = _sha256(self._integrity_payload())
        if self.context_hash != expected_hash:
            raise ValueError("context_hash does not match packet contents")
        if self.context_id != f"ctx-{expected_hash[:20]}":
            raise ValueError("context_id does not match context_hash")

    def _integrity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_schema_version": self.feature_schema_version,
            "source_hash": self.source_hash,
            "decision_t": self.decision_t,
            "observed_through": self.observed_through,
            "history_start": self.history_start,
            "observation_count": self.observation_count,
            "feature_names": list(self.feature_names),
            "raw_features": list(self.raw_features),
            "vector_feature_names": list(self.vector_feature_names),
            "context_vector": list(self.context_vector),
            "encoder_version": self.encoder_version,
            "prompt_summary": self.prompt_summary,
            "projection_metadata": self.projection_metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "context_hash": self.context_hash,
            **self._integrity_payload(),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContextPacket":
        expected_keys = {
            "context_id",
            "context_hash",
            "schema_version",
            "feature_schema_version",
            "source_hash",
            "decision_t",
            "observed_through",
            "history_start",
            "observation_count",
            "feature_names",
            "raw_features",
            "vector_feature_names",
            "context_vector",
            "encoder_version",
            "prompt_summary",
            "projection_metadata",
        }
        _strict_keys(value, expected_keys, "context packet")
        return cls(
            context_id=value["context_id"],
            context_hash=value["context_hash"],
            schema_version=value["schema_version"],
            feature_schema_version=value["feature_schema_version"],
            source_hash=value["source_hash"],
            decision_t=value["decision_t"],
            observed_through=value["observed_through"],
            history_start=value["history_start"],
            observation_count=value["observation_count"],
            feature_names=tuple(value["feature_names"]),
            raw_features=tuple(value["raw_features"]),
            vector_feature_names=tuple(value["vector_feature_names"]),
            context_vector=tuple(value["context_vector"]),
            encoder_version=value["encoder_version"],
            prompt_summary=value["prompt_summary"],
            projection_metadata=value["projection_metadata"],
        )

    @classmethod
    def from_json(cls, value: str) -> "ContextPacket":
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("context packet JSON root must be an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class ContextRoute:
    """The two independent channels selected for a context packet."""

    mode: str
    context_id: str
    retrieval_vector: Optional[tuple[float, ...]]
    prompt_summary: str
    schema_version: str = ROUTE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROUTE_SCHEMA_VERSION:
            raise ValueError(f"unsupported route schema {self.schema_version!r}")
        normalized = _normalize_mode(self.mode)
        object.__setattr__(self, "mode", normalized)
        _nonempty_string(self.context_id, "context_id")
        if self.retrieval_vector is not None:
            object.__setattr__(
                self,
                "retrieval_vector",
                tuple(
                    _finite_float(value, f"retrieval_vector[{index}]")
                    for index, value in enumerate(self.retrieval_vector)
                ),
            )

    @property
    def to_retrieval(self) -> bool:
        return self.retrieval_vector is not None

    @property
    def to_prompt(self) -> bool:
        return bool(self.prompt_summary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "context_id": self.context_id,
            "retrieval_vector": (
                list(self.retrieval_vector)
                if self.retrieval_vector is not None
                else None
            ),
            "prompt_summary": self.prompt_summary,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContextRoute":
        expected_keys = {
            "schema_version",
            "mode",
            "context_id",
            "retrieval_vector",
            "prompt_summary",
        }
        _strict_keys(value, expected_keys, "context route")
        vector = value["retrieval_vector"]
        return cls(
            schema_version=value["schema_version"],
            mode=value["mode"],
            context_id=value["context_id"],
            retrieval_vector=None if vector is None else tuple(vector),
            prompt_summary=value["prompt_summary"],
        )

    @classmethod
    def from_json(cls, value: str) -> "ContextRoute":
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("context route JSON root must be an object")
        return cls.from_dict(decoded)


class CausalContextRouter:
    """Build causal rolling context and independently route its two channels."""

    def __init__(
        self,
        *,
        base_feature_names: Sequence[str] = DEFAULT_BASE_FEATURES,
        event_feature_names: Sequence[str] = (),
        window_size: int = 6,
        mode: str = "retrieval-only",
        projection: Optional[FrozenLinearProjection] = None,
    ) -> None:
        self.base_feature_names = _string_tuple(
            base_feature_names, "base_feature_names"
        )
        if isinstance(event_feature_names, (str, bytes)):
            raise ValueError("event_feature_names must be a sequence")
        self.event_feature_names = tuple(
            _nonempty_string(name, "event_feature_names item")
            for name in event_feature_names
        )
        if len(set(self.event_feature_names)) != len(self.event_feature_names):
            raise ValueError("event_feature_names contains duplicate names")
        if set(self.event_feature_names) & set(self.base_feature_names):
            raise ValueError("base and event feature names must be disjoint")
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise ValueError("window_size must be an integer")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.mode = _normalize_mode(mode)
        self.projection = projection
        if projection is not None and projection.input_features != self.feature_names:
            raise ValueError(
                "projection input feature schema/order mismatch: "
                f"expected {self.feature_names!r}, "
                f"found {projection.input_features!r}"
            )
        if projection is not None and projection.feature_config != self.feature_config:
            raise ValueError(
                "projection rolling feature configuration mismatch: "
                f"expected {self.feature_config!r}, "
                f"found {projection.feature_config!r}"
            )

    @property
    def feature_names(self) -> tuple[str, ...]:
        rolling = tuple(
            f"{feature}.{statistic}"
            for feature in self.base_feature_names
            for statistic in ("last", "mean", "slope")
        )
        event = ("event.present",) + tuple(
            f"event.{feature}" for feature in self.event_feature_names
        ) if self.event_feature_names else ()
        return rolling + event

    @property
    def feature_config(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "base_feature_names": list(self.base_feature_names),
            "event_feature_names": list(self.event_feature_names),
        }

    @property
    def vector_feature_names(self) -> tuple[str, ...]:
        if self.projection is None:
            return self.feature_names
        return self.projection.output_features

    def _validated_window(
        self,
        history: Sequence[Mapping[str, Any]],
        observed_through: int,
    ) -> tuple[Mapping[str, Any], ...]:
        if isinstance(history, (str, bytes)) or not history:
            raise ValueError("history must contain at least one observation")
        validated = []
        previous_timestamp: Optional[int] = None
        for index, row in enumerate(history):
            if not isinstance(row, Mapping):
                raise ValueError(f"history[{index}] must be an object")
            if "timestamp" not in row:
                raise ValueError(f"history[{index}] is missing 'timestamp'")
            row_timestamp = _timestamp(row["timestamp"], f"history[{index}].timestamp")
            if previous_timestamp is not None and row_timestamp <= previous_timestamp:
                raise ValueError("history timestamps must be strictly increasing")
            if row_timestamp > observed_through:
                raise ValueError(
                    f"history contains future observation t={row_timestamp} beyond "
                    f"observed_through={observed_through}"
                )
            for feature in self.base_feature_names:
                if feature not in row:
                    raise ValueError(f"history[{index}] is missing feature {feature!r}")
                _finite_float(row[feature], f"history[{index}][{feature!r}]")
            validated.append(row)
            previous_timestamp = row_timestamp
        if previous_timestamp != observed_through:
            raise ValueError(
                "the last history timestamp must equal observed_through; pass an "
                "explicit carried-forward observation when macro fields are sparse"
            )
        return tuple(validated[-self.window_size :])

    def _validated_event(
        self,
        event: Optional[Mapping[str, Any]],
        observed_through: int,
    ) -> tuple[float, ...]:
        if not self.event_feature_names:
            if event is not None:
                raise ValueError(
                    "event was provided but this router has no event_feature_names"
                )
            return ()
        if event is None:
            return (0.0,) + (0.0,) * len(self.event_feature_names)
        if not isinstance(event, Mapping):
            raise ValueError("event must be an object")
        if "timestamp" not in event:
            raise ValueError("event is missing 'timestamp'")
        event_timestamp = _timestamp(event["timestamp"], "event.timestamp")
        if event_timestamp > observed_through:
            raise ValueError(
                f"event timestamp {event_timestamp} is later than "
                f"observed_through={observed_through}"
            )
        values = []
        for feature in self.event_feature_names:
            if feature not in event:
                raise ValueError(f"event is missing feature {feature!r}")
            values.append(_finite_float(event[feature], f"event[{feature!r}]"))
        return (1.0, *values)

    def encode(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        decision_t: int,
        observed_through: Optional[int] = None,
        event: Optional[Mapping[str, Any]] = None,
    ) -> ContextPacket:
        decision_t = _timestamp(decision_t, "decision_t")
        observed_through = (
            decision_t
            if observed_through is None
            else _timestamp(observed_through, "observed_through")
        )
        if observed_through > decision_t:
            raise ValueError("observed_through cannot be later than decision_t")
        window = self._validated_window(history, observed_through)
        event_values = self._validated_event(event, observed_through)

        feature_set = set(self.base_feature_names)
        raw_features = []
        for feature in self.base_feature_names:
            availability_feature = f"{feature}_available"
            if (
                not feature.endswith("_available")
                and availability_feature in feature_set
            ):
                # A paired availability feature makes the numeric value in a
                # missing row a storage placeholder, not an observation.  Keep
                # the fixed three-statistic schema, but calculate it from only
                # the causally available (timestamp, value) pairs.  Endpoint
                # slope is per timestamp unit; zero or one available point has
                # the deterministic neutral slope 0.0.
                observed = [
                    (
                        _timestamp(row["timestamp"], "history timestamp"),
                        _finite_float(
                            row[feature], f"history feature {feature!r}"
                        ),
                    )
                    for row in window
                    if _finite_float(
                        row[availability_feature],
                        f"history feature {availability_feature!r}",
                    )
                    > 0.0
                ]
                if not observed:
                    statistics = (0.0, 0.0, 0.0)
                else:
                    first_t, first_value = observed[0]
                    last_t, last_value = observed[-1]
                    slope = (
                        0.0
                        if len(observed) == 1
                        else (last_value - first_value) / (last_t - first_t)
                    )
                    statistics = (
                        last_value,
                        sum(value for _, value in observed) / len(observed),
                        slope,
                    )
            else:
                # Unmasked values, including the availability-mask features
                # themselves, retain the ordinary rolling-statistic semantics.
                values = [
                    _finite_float(row[feature], f"history feature {feature!r}")
                    for row in window
                ]
                statistics = (
                    values[-1],
                    sum(values) / len(values),
                    (values[-1] - values[0]) / max(len(values) - 1, 1),
                )
            raw_features.extend(
                statistics
            )
        raw_features.extend(event_values)
        raw_vector = tuple(raw_features)

        if self.projection is None:
            context_vector = raw_vector
            vector_feature_names = self.feature_names
            encoder_version = "identity-rolling-v2"
            projection_metadata = None
        else:
            context_vector = self.projection.transform(raw_vector)
            vector_feature_names = self.projection.output_features
            encoder_version = self.projection.projection_id
            projection_metadata = {
                "projection_id": self.projection.projection_id,
                "projection_hash": self.projection.projection_hash,
                "training_metadata": dict(self.projection.training_metadata),
            }

        last = window[-1]
        # A numeric placeholder plus an explicit mask is useful for a fixed-size
        # vector, but the placeholder must never be narrated as an observation.
        # Features following ``<name>``/``<name>_available`` therefore render as
        # either the observed value or one unambiguous ``unavailable`` token.
        rendered_fields = []
        for feature in self.base_feature_names:
            if (
                feature.endswith("_available")
                and feature[: -len("_available")] in feature_set
            ):
                continue
            availability_feature = f"{feature}_available"
            if (
                availability_feature in feature_set
                and float(last[availability_feature]) <= 0.0
            ):
                rendered_fields.append(f"{feature}=unavailable")
            else:
                rendered_fields.append(f"{feature}={float(last[feature]):.6g}")
        summary_fields = ", ".join(rendered_fields)
        if self.event_feature_names:
            if event is None:
                summary_fields += ", event_present=0"
            else:
                event_summary = ", ".join(
                    f"event_{feature}={float(event[feature]):.6g}"
                    for feature in self.event_feature_names
                )
                summary_fields += f", event_present=1, {event_summary}"
        prompt_summary = (
            f"Observed context through t={observed_through} "
            f"(rolling_n={len(window)}): {summary_fields}."
        )

        causal_source = {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "window_size": self.window_size,
            "observed_through": observed_through,
            "rows": [
                {
                    "timestamp": row["timestamp"],
                    **{
                        feature: _finite_float(row[feature], feature)
                        for feature in self.base_feature_names
                    },
                }
                for row in window
            ],
            "event": (
                None
                if event is None
                else {
                    "timestamp": event["timestamp"],
                    **{
                        feature: _finite_float(event[feature], feature)
                        for feature in self.event_feature_names
                    },
                }
            ),
        }
        source_hash = _sha256(causal_source)
        packet_payload = {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "source_hash": source_hash,
            "decision_t": decision_t,
            "observed_through": observed_through,
            "history_start": window[0]["timestamp"],
            "observation_count": len(window),
            "feature_names": list(self.feature_names),
            "raw_features": list(raw_vector),
            "vector_feature_names": list(vector_feature_names),
            "context_vector": list(context_vector),
            "encoder_version": encoder_version,
            "prompt_summary": prompt_summary,
            "projection_metadata": projection_metadata,
        }
        context_hash = _sha256(packet_payload)
        return ContextPacket(
            context_id=f"ctx-{context_hash[:20]}",
            context_hash=context_hash,
            source_hash=source_hash,
            decision_t=decision_t,
            observed_through=observed_through,
            history_start=window[0]["timestamp"],
            observation_count=len(window),
            feature_names=self.feature_names,
            raw_features=raw_vector,
            vector_feature_names=vector_feature_names,
            context_vector=context_vector,
            encoder_version=encoder_version,
            prompt_summary=prompt_summary,
            projection_metadata=projection_metadata,
        )

    def route(
        self, packet: ContextPacket, *, mode: Optional[str] = None
    ) -> ContextRoute:
        selected_mode = self.mode if mode is None else _normalize_mode(mode)
        retrieval_enabled = selected_mode in {"retrieval-only", "full"}
        prompt_enabled = selected_mode in {"prompt-only", "full"}
        return ContextRoute(
            mode=selected_mode,
            context_id=packet.context_id,
            retrieval_vector=(packet.context_vector if retrieval_enabled else None),
            prompt_summary=(packet.prompt_summary if prompt_enabled else ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "base_feature_names": list(self.base_feature_names),
            "event_feature_names": list(self.event_feature_names),
            "window_size": self.window_size,
            "mode": self.mode,
            "projection": self.projection.to_dict() if self.projection else None,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CausalContextRouter":
        expected_keys = {
            "schema_version",
            "feature_schema_version",
            "base_feature_names",
            "event_feature_names",
            "window_size",
            "mode",
            "projection",
        }
        _strict_keys(value, expected_keys, "context router")
        if value["schema_version"] != ROUTER_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported router schema {value['schema_version']!r}"
            )
        if value["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported feature schema {value['feature_schema_version']!r}"
            )
        base_features = tuple(value["base_feature_names"])
        event_features = tuple(value["event_feature_names"])
        probe = cls(
            base_feature_names=base_features,
            event_feature_names=event_features,
            window_size=value["window_size"],
            mode=value["mode"],
        )
        projection_value = value["projection"]
        projection = (
            None
            if projection_value is None
            else FrozenLinearProjection.from_dict(
                projection_value,
                expected_input_features=probe.feature_names,
                expected_feature_config=probe.feature_config,
            )
        )
        return cls(
            base_feature_names=base_features,
            event_feature_names=event_features,
            window_size=value["window_size"],
            mode=value["mode"],
            projection=projection,
        )

    @classmethod
    def from_json(cls, value: str) -> "CausalContextRouter":
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("context router JSON root must be an object")
        return cls.from_dict(decoded)

    @classmethod
    def with_projection_json(
        cls,
        path: str | Path,
        *,
        base_feature_names: Sequence[str] = DEFAULT_BASE_FEATURES,
        event_feature_names: Sequence[str] = (),
        window_size: int = 6,
        mode: str = "retrieval-only",
    ) -> "CausalContextRouter":
        probe = cls(
            base_feature_names=base_feature_names,
            event_feature_names=event_feature_names,
            window_size=window_size,
            mode=mode,
        )
        projection = FrozenLinearProjection.load_json(
            path,
            expected_input_features=probe.feature_names,
            expected_feature_config=probe.feature_config,
        )
        return cls(
            base_feature_names=base_feature_names,
            event_feature_names=event_feature_names,
            window_size=window_size,
            mode=mode,
            projection=projection,
        )
