"""M2: evidence-linked episodic memory with correctly aligned transitions.

The legacy implementation wrote ``action_t`` beside the wealth delta observed
on entry to month ``t``.  This module makes the temporal contract explicit:
``begin_episode`` is called before the environment step and
``finalize_episode`` is called only after ``state_{t+1}`` and the realized
outcome are available.  Retrieval never sees pending transitions.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "m2-episodic-v2"


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible copy."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _json_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    """Return a finite JSON-object copy or fail with a field-local error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    try:
        result = _jsonable(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field_name} must be JSON-compatible and contain only finite numbers"
        ) from exc
    if not isinstance(result, dict):  # Defensive: Mapping should encode as an object.
        raise ValueError(f"{field_name} must be a JSON object")
    return result


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _finite_vector(values: Any, field_name: str) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of finite numbers")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of finite numbers") from exc
    normalized = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must contain only finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must contain only finite numbers")
        normalized.append(number)
    return tuple(normalized)


def _optional_unit_float(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be null or a finite number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be null or a finite number in [0, 1]")
    return number


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return number


def _identifier(value: Any, field_name: str) -> str:
    # Empty strings remain permitted for backwards-compatible optional labels;
    # typed identity fields must nevertheless stay JSON string scalars.
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _identifier_tuple(
    values: Any,
    field_name: str,
    *,
    deduplicate: bool,
) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of string IDs")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of string IDs") from exc
    normalized = tuple(_identifier(value, field_name) for value in items)
    unique = tuple(dict.fromkeys(normalized))
    if not deduplicate and len(unique) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate IDs")
    return unique if deduplicate else normalized


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = math.sqrt(sum(float(value) ** 2 for value in left))
    norm_right = math.sqrt(sum(float(value) ** 2 for value in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_left * norm_right)))


STATE_SCALES: Dict[str, float] = {
    "price": 100.0,
    "interest_rate": 0.05,
    "low_labor_rate": 0.10,
    "unemployment_rate": 0.10,
    "inflation": 0.10,
    "wealth": 100_000.0,
    "income": 10_000.0,
    "employed": 1.0,
}


def state_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Bounded RBF similarity over shared, explicitly scaled numeric fields."""
    squared = []
    for key, scale in STATE_SCALES.items():
        if key not in left or key not in right:
            continue
        try:
            a = float(left[key])
            b = float(right[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(a) or not math.isfinite(b):
            continue
        squared.append(((a - b) / scale) ** 2)
    if not squared:
        return 0.0
    return math.exp(-math.sqrt(sum(squared) / len(squared)))


def _utility_statistics(
    previous_utilities: Sequence[float], utility_value: float
) -> tuple[float, float]:
    """Return the causal utility advantage and prompt importance.

    M3 consumes both values as evidence, so they must be reproducible from the
    append order of a per-agent causal ledger rather than depend on whichever
    older transition happened to be finalized first.
    """

    baseline = statistics.median(previous_utilities) if previous_utilities else 0.0
    utility_advantage = utility_value - baseline
    scale = (
        statistics.median([abs(value - baseline) for value in previous_utilities])
        if previous_utilities
        else 1.0
    )
    scale = max(float(scale), 1e-6)
    importance = 1.0 + min(1.0, abs(utility_advantage) / scale)
    return utility_advantage, importance


@dataclass(frozen=True)
class PendingEpisode:
    decision_id: str
    episode_id: str
    run_id: str
    seed: int
    agent_id: int
    decision_t: int
    pre_state: Dict[str, Any]
    context_id: str
    context_vector: Tuple[float, ...]
    retrieved_episode_ids: Tuple[str, ...]
    selected_rule_ids: Tuple[str, ...]
    proposed_action: Dict[str, Any]
    executed_action: Dict[str, Any]
    rng_draw: Optional[float]
    reflection: str

    def verify_structure(self) -> None:
        """Validate a pending transition without registering or mutating it."""

        _identifier(self.decision_id, "decision_id")
        _identifier(self.episode_id, "episode_id")
        _identifier(self.run_id, "run_id")
        _integer(self.seed, "seed")
        _nonnegative_integer(self.agent_id, "agent_id")
        _nonnegative_integer(self.decision_t, "decision_t")
        _json_mapping(self.pre_state, "pre_state")
        _identifier(self.context_id, "context_id")
        _finite_vector(self.context_vector, "context_vector")
        _identifier_tuple(
            self.retrieved_episode_ids,
            "retrieved_episode_ids",
            deduplicate=False,
        )
        _identifier_tuple(
            self.selected_rule_ids,
            "selected_rule_ids",
            deduplicate=False,
        )
        _json_mapping(self.proposed_action, "proposed_action")
        _json_mapping(self.executed_action, "executed_action")
        _optional_unit_float(self.rng_draw, "rng_draw")
        if not isinstance(self.reflection, str):
            raise ValueError("reflection must be a string")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PendingEpisode":
        """Restore a canonical pending record without calling ``begin_episode``."""

        if not isinstance(payload, Mapping):
            raise ValueError("restored pending episode must be an object")
        expected_keys = set(cls.__dataclass_fields__)
        if set(payload) != expected_keys:
            missing = sorted(expected_keys - set(payload))
            extra = sorted(set(payload) - expected_keys)
            raise ValueError(
                "pending episode keys are not exact: "
                f"missing={missing}, extra={extra}"
            )
        values = dict(payload)
        # Canonicalize exactly the fields that ``begin_episode`` copies or
        # normalizes. This creates no ledger/pending mutation and therefore
        # cannot consume a decision ID on a failed restore.
        values["decision_t"] = _nonnegative_integer(
            values["decision_t"], "decision_t"
        )
        values["pre_state"] = _json_mapping(values["pre_state"], "pre_state")
        values["context_id"] = _identifier(values["context_id"], "context_id")
        values["context_vector"] = _finite_vector(
            values["context_vector"], "context_vector"
        )
        values["retrieved_episode_ids"] = _identifier_tuple(
            values["retrieved_episode_ids"],
            "retrieved_episode_ids",
            deduplicate=False,
        )
        values["selected_rule_ids"] = _identifier_tuple(
            values["selected_rule_ids"],
            "selected_rule_ids",
            deduplicate=False,
        )
        values["proposed_action"] = _json_mapping(
            values["proposed_action"], "proposed_action"
        )
        values["executed_action"] = _json_mapping(
            values["executed_action"], "executed_action"
        )
        values["rng_draw"] = _optional_unit_float(values["rng_draw"], "rng_draw")
        pending = cls(**values)
        pending.verify_structure()
        return pending


@dataclass(frozen=True)
class EpisodeRecord:
    schema_version: str
    decision_id: str
    episode_id: str
    run_id: str
    seed: int
    agent_id: int
    decision_t: int
    outcome_t: int
    pre_state: Dict[str, Any]
    next_state: Dict[str, Any]
    context_id: str
    context_vector: Tuple[float, ...]
    retrieved_episode_ids: Tuple[str, ...]
    selected_rule_ids: Tuple[str, ...]
    proposed_action: Dict[str, Any]
    executed_action: Dict[str, Any]
    rng_draw: Optional[float]
    reflection: str
    outcome: Dict[str, Any]
    reward: float
    flow_utility: float
    utility_advantage: float
    importance: float
    record_hash: str = field(compare=False)

    def integrity_payload(self) -> Dict[str, Any]:
        """Return the exact hash payload used when the record was finalized."""

        values = asdict(self)
        values.pop("record_hash")
        return values

    def verify_integrity(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported episode-record schema version")
        _identifier(self.decision_id, "decision_id")
        _identifier(self.episode_id, "episode_id")
        _identifier(self.run_id, "run_id")
        _integer(self.seed, "seed")
        _nonnegative_integer(self.agent_id, "agent_id")
        _nonnegative_integer(self.decision_t, "decision_t")
        _nonnegative_integer(self.outcome_t, "outcome_t")
        if self.outcome_t != self.decision_t + 1:
            raise ValueError("episode outcome_t must equal decision_t + 1")
        _json_mapping(self.pre_state, "pre_state")
        _json_mapping(self.next_state, "next_state")
        _identifier(self.context_id, "context_id")
        _finite_vector(self.context_vector, "context_vector")
        _identifier_tuple(
            self.retrieved_episode_ids,
            "retrieved_episode_ids",
            deduplicate=False,
        )
        _identifier_tuple(
            self.selected_rule_ids,
            "selected_rule_ids",
            deduplicate=False,
        )
        _json_mapping(self.proposed_action, "proposed_action")
        _json_mapping(self.executed_action, "executed_action")
        _optional_unit_float(self.rng_draw, "rng_draw")
        if not isinstance(self.reflection, str):
            raise ValueError("reflection must be a string")
        _json_mapping(self.outcome, "outcome")
        _finite_number(self.reward, "reward")
        _finite_number(self.flow_utility, "flow_utility")
        _finite_number(self.utility_advantage, "utility_advantage")
        importance = _finite_number(self.importance, "importance")
        if not 1.0 <= importance <= 2.0:
            raise ValueError("importance must be in [1, 2]")
        if not isinstance(self.record_hash, str):
            raise ValueError("record_hash must be a string")
        expected_hash = _stable_hash(self.integrity_payload())
        if self.record_hash != expected_hash:
            raise ValueError(f"episode record hash mismatch: {self.episode_id}")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in (
            "context_vector",
            "retrieved_episode_ids",
            "selected_rule_ids",
        ):
            result[key] = list(result[key])
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeRecord":
        if not isinstance(payload, Mapping):
            raise ValueError("restored episode record must be an object")
        expected_keys = set(cls.__dataclass_fields__)
        if set(payload) != expected_keys:
            missing = sorted(expected_keys - set(payload))
            extra = sorted(set(payload) - expected_keys)
            raise ValueError(
                "episode record keys are not exact: "
                f"missing={missing}, extra={extra}"
            )
        values = dict(payload)
        for name in (
            "pre_state",
            "next_state",
            "proposed_action",
            "executed_action",
            "outcome",
        ):
            values[name] = _json_mapping(values[name], name)
        values["context_vector"] = _finite_vector(
            values["context_vector"], "context_vector"
        )
        values["retrieved_episode_ids"] = _identifier_tuple(
            values["retrieved_episode_ids"],
            "retrieved_episode_ids",
            deduplicate=False,
        )
        values["selected_rule_ids"] = _identifier_tuple(
            values["selected_rule_ids"],
            "selected_rule_ids",
            deduplicate=False,
        )
        record = cls(**values)
        record.verify_integrity()
        return record

    def to_prompt_text(self) -> str:
        work = float(self.executed_action.get("labor_hours", 0.0))
        consumption = float(self.executed_action.get("consumption_fraction", 0.0))
        wealth_change = float(self.outcome.get("wealth_change", 0.0))
        return (
            f"[{self.episode_id}] state at month {self.decision_t}; "
            f"worked {work:.1f}h, consumed {consumption:.0%}; "
            f"realized utility {self.flow_utility:.4f}, "
            f"wealth change {wealth_change:+.2f}."
        )


@dataclass(frozen=True)
class RetrievalHit:
    episode: EpisodeRecord
    score: float
    components: Dict[str, float]

    def to_trace(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode.episode_id,
            "score": self.score,
            "components": dict(self.components),
        }


class EvidenceLinkedEpisodicTrack:
    """Bounded prompt buffer backed by an append-only evidence ledger."""

    def __init__(
        self,
        *,
        run_id: str,
        seed: int,
        agent_id: int,
        prompt_capacity: int = 24,
        time_decay_rate: float = 0.95,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        seed_value = _integer(seed, "seed")
        agent_id_value = _nonnegative_integer(agent_id, "agent_id")
        prompt_capacity_value = _nonnegative_integer(
            prompt_capacity, "prompt_capacity"
        )
        if prompt_capacity_value < 1:
            raise ValueError("prompt_capacity must be positive")
        time_decay_rate_value = _finite_number(
            time_decay_rate, "time_decay_rate"
        )
        if not 0.0 < time_decay_rate_value <= 1.0:
            raise ValueError("time_decay_rate must be in (0, 1]")
        self.run_id = run_id
        self.seed = seed_value
        self.agent_id = agent_id_value
        self.prompt_capacity = prompt_capacity_value
        self.time_decay_rate = time_decay_rate_value
        self._pending: Dict[str, PendingEpisode] = {}
        self._ledger: "OrderedDict[str, EpisodeRecord]" = OrderedDict()
        # This deque is only the bounded, serializable prompt-buffer view.  It
        # must never define the retrieval candidate set: long-horizon evidence
        # remains eligible through the append-only finalized ledger below.
        self._prompt_ids: Deque[str] = deque(maxlen=prompt_capacity)

    @staticmethod
    def make_episode_id(run_id: str, seed: int, agent_id: int, decision_t: int) -> str:
        safe_run = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_id)
        return f"{safe_run}:s{int(seed)}:a{int(agent_id)}:t{int(decision_t)}"

    def begin_episode(
        self,
        *,
        decision_t: int,
        pre_state: Mapping[str, Any],
        context_id: str,
        context_vector: Sequence[float],
        retrieved_episode_ids: Iterable[str],
        selected_rule_ids: Iterable[str],
        proposed_action: Mapping[str, Any],
        executed_action: Mapping[str, Any],
        rng_draw: Optional[float] = None,
        reflection: str = "",
    ) -> str:
        decision_t_value = _nonnegative_integer(decision_t, "decision_t")
        episode_id = self.make_episode_id(
            self.run_id, self.seed, self.agent_id, decision_t_value
        )
        decision_id = f"D:{episode_id}"
        if decision_id in self._pending or episode_id in self._ledger:
            raise ValueError(f"duplicate decision/episode id: {decision_id}")
        known_decision_times = [
            *(record.decision_t for record in self._ledger.values()),
            *(item.decision_t for item in self._pending.values()),
        ]
        if known_decision_times and decision_t_value <= max(known_decision_times):
            raise ValueError(
                "episodes must begin in strictly increasing per-agent decision order"
            )
        vector = _finite_vector(context_vector, "context_vector")
        pending = PendingEpisode(
            decision_id=decision_id,
            episode_id=episode_id,
            run_id=self.run_id,
            seed=self.seed,
            agent_id=self.agent_id,
            decision_t=decision_t_value,
            pre_state=_json_mapping(pre_state, "pre_state"),
            context_id=str(context_id),
            context_vector=vector,
            retrieved_episode_ids=_identifier_tuple(
                retrieved_episode_ids,
                "retrieved_episode_ids",
                deduplicate=True,
            ),
            selected_rule_ids=_identifier_tuple(
                selected_rule_ids,
                "selected_rule_ids",
                deduplicate=True,
            ),
            proposed_action=_json_mapping(proposed_action, "proposed_action"),
            executed_action=_json_mapping(executed_action, "executed_action"),
            rng_draw=_optional_unit_float(rng_draw, "rng_draw"),
            reflection=str(reflection),
        )
        pending.verify_structure()
        self._pending[decision_id] = pending
        return decision_id

    def finalize_episode(
        self,
        decision_id: str,
        *,
        outcome_t: int,
        next_state: Mapping[str, Any],
        outcome: Mapping[str, Any],
        reward: float,
        flow_utility: float,
    ) -> EpisodeRecord:
        if decision_id not in self._pending:
            raise KeyError(f"unknown or already finalized decision: {decision_id}")
        pending = self._pending[decision_id]
        # Pending objects normally come from ``begin_episode`` or validated
        # restore. Re-check here so even accidental/private-state corruption
        # cannot be sealed into the evidence ledger.
        pending.verify_structure()
        outcome_t_value = _nonnegative_integer(outcome_t, "outcome_t")
        if outcome_t_value != pending.decision_t + 1:
            raise ValueError(
                f"outcome_t must equal decision_t + 1; got {outcome_t} for {pending.decision_t}"
            )
        reward_value = float(reward)
        utility_value = float(flow_utility)
        if not math.isfinite(reward_value) or not math.isfinite(utility_value):
            raise ValueError("reward and flow_utility must be finite")

        earlier_pending = [
            item.decision_t
            for candidate_id, item in self._pending.items()
            if candidate_id != decision_id and item.decision_t < pending.decision_t
        ]
        if earlier_pending:
            raise ValueError(
                "cannot finalize an episode before earlier pending decisions"
            )
        if self._ledger:
            latest = next(reversed(self._ledger.values()))
            if pending.decision_t <= latest.decision_t:
                raise ValueError(
                    "episodes must finalize in strictly increasing per-agent decision order"
                )

        # Utility normalization is part of the evidence consumed by M3.  The
        # monotonic ledger invariant above makes the complete causal prefix
        # reproducible at restore time; future outcomes can never enter it.
        previous_utilities = [
            record.flow_utility
            for record in self._ledger.values()
            if record.outcome_t <= pending.decision_t
        ]
        utility_advantage, importance = _utility_statistics(
            previous_utilities, utility_value
        )

        core = {
            "schema_version": SCHEMA_VERSION,
            **asdict(pending),
            "outcome_t": outcome_t_value,
            "next_state": _json_mapping(next_state, "next_state"),
            "outcome": _json_mapping(outcome, "outcome"),
            "reward": reward_value,
            "flow_utility": utility_value,
            "utility_advantage": utility_advantage,
            "importance": importance,
        }
        record = EpisodeRecord(**core, record_hash=_stable_hash(core))
        record.verify_integrity()

        # Commit only after every conversion, statistic, and hash check succeeds.
        # In particular, malformed next_state/outcome JSON must leave the pending
        # transition available for a corrected retry.
        self._pending.pop(decision_id)
        self._ledger[record.episode_id] = record
        self._prompt_ids.append(record.episode_id)
        return EpisodeRecord.from_dict(record.to_dict())

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_episodes(self) -> Tuple[PendingEpisode, ...]:
        return tuple(
            PendingEpisode.from_dict(asdict(item))
            for item in self._pending.values()
        )

    @property
    def finalized_count(self) -> int:
        return len(self._ledger)

    @property
    def finalized_episodes(self) -> Tuple[EpisodeRecord, ...]:
        return tuple(
            EpisodeRecord.from_dict(record.to_dict())
            for record in self._ledger.values()
        )

    def get(self, episode_id: str) -> Optional[EpisodeRecord]:
        record = self._ledger.get(episode_id)
        return None if record is None else EpisodeRecord.from_dict(record.to_dict())

    def retrieve(
        self,
        *,
        current_t: int,
        current_state: Mapping[str, Any],
        context_vector: Optional[Sequence[float]],
        use_context: bool,
        k: int = 5,
        recency_weight: float = 0.25,
        state_weight: float = 0.35,
        importance_weight: float = 0.15,
        context_weight: float = 0.25,
    ) -> Tuple[RetrievalHit, ...]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if any(weight < 0.0 for weight in (
            recency_weight, state_weight, importance_weight, context_weight
        )):
            raise ValueError("retrieval weights must be non-negative")
        query_context = tuple(float(value) for value in (context_vector or ()))
        hits = []
        # Score the complete append-only evidence ledger. ``prompt_capacity``
        # bounds only how many top-ranked records can enter the eventual prompt;
        # using the bounded ``_prompt_ids`` buffer here would silently turn M2
        # into a rolling-window memory and make old evidence irretrievable.
        for episode in self._ledger.values():
            # A finalized object can still be from the caller's future after an
            # out-of-order restore or direct-track use. Retrieval is causal at the
            # outcome boundary, not merely "finalized somewhere in the ledger".
            if episode.outcome_t > int(current_t):
                continue
            age = int(current_t) - episode.decision_t
            recency = self.time_decay_rate ** age
            similarity = state_similarity(current_state, episode.pre_state)
            importance = min(1.0, episode.importance / 2.0)
            context = 0.0
            if use_context and query_context and episode.context_vector:
                context = (_cosine_similarity(query_context, episode.context_vector) + 1.0) / 2.0
            components = {
                "recency": recency_weight * recency,
                "state_similarity": state_weight * similarity,
                "importance": importance_weight * importance,
                "context_similarity": context_weight * context if use_context else 0.0,
            }
            score = sum(components.values())
            hits.append(
                RetrievalHit(
                    episode=EpisodeRecord.from_dict(episode.to_dict()),
                    score=score,
                    components=components,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, -hit.episode.decision_t, hit.episode.episode_id))
        return tuple(hits[:min(k, self.prompt_capacity)])

    def validate_references(self) -> None:
        for episode in self._ledger.values():
            episode.verify_integrity()
            missing = [
                episode_id
                for episode_id in episode.retrieved_episode_ids
                if episode_id not in self._ledger
            ]
            # A run can reference an older episode that was imported from a
            # different ledger only if the caller explicitly carries it over.
            if missing:
                raise ValueError(
                    f"episode {episode.episode_id} references missing evidence: {missing}"
                )
            future = [
                episode_id
                for episode_id in episode.retrieved_episode_ids
                if self._ledger[episode_id].outcome_t > episode.decision_t
            ]
            if future:
                raise ValueError(
                    f"episode {episode.episode_id} references future evidence: {future}"
                )

        for pending in self._pending.values():
            missing = [
                episode_id
                for episode_id in pending.retrieved_episode_ids
                if episode_id not in self._ledger
            ]
            if missing:
                raise ValueError(
                    f"pending decision {pending.decision_id} references missing evidence: {missing}"
                )
            future = [
                episode_id
                for episode_id in pending.retrieved_episode_ids
                if self._ledger[episode_id].outcome_t > pending.decision_t
            ]
            if future:
                raise ValueError(
                    f"pending decision {pending.decision_id} references future evidence: {future}"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "agent_id": self.agent_id,
            "prompt_capacity": self.prompt_capacity,
            "time_decay_rate": self.time_decay_rate,
            "prompt_ids": list(self._prompt_ids),
            "episodes": [record.to_dict() for record in self._ledger.values()],
            "pending": [asdict(item) for item in self._pending.values()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceLinkedEpisodicTrack":
        if not isinstance(payload, Mapping):
            raise ValueError("M2 snapshot must be an object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported M2 schema version")
        expected_keys = {
            "schema_version",
            "run_id",
            "seed",
            "agent_id",
            "prompt_capacity",
            "time_decay_rate",
            "prompt_ids",
            "episodes",
            "pending",
        }
        if set(payload) != expected_keys:
            missing = sorted(expected_keys - set(payload))
            extra = sorted(set(payload) - expected_keys)
            raise ValueError(
                f"M2 snapshot keys are not exact: missing={missing}, extra={extra}"
            )
        if not isinstance(payload["run_id"], str) or not payload["run_id"].strip():
            raise ValueError("M2 run_id must be a non-empty string")
        seed = _integer(payload["seed"], "M2 seed")
        agent_id = _nonnegative_integer(payload["agent_id"], "M2 agent_id")
        prompt_capacity = _nonnegative_integer(
            payload["prompt_capacity"], "M2 prompt_capacity"
        )
        if prompt_capacity < 1:
            raise ValueError("M2 prompt_capacity must be positive")
        time_decay_rate = _finite_number(
            payload["time_decay_rate"], "M2 time_decay_rate"
        )
        if not 0.0 < time_decay_rate <= 1.0:
            raise ValueError("M2 time_decay_rate must lie in (0, 1]")
        for name in ("prompt_ids", "episodes", "pending"):
            if not isinstance(payload[name], list):
                raise ValueError(f"M2 {name} must be an array")
        track = cls(
            run_id=payload["run_id"],
            seed=seed,
            agent_id=agent_id,
            prompt_capacity=prompt_capacity,
            time_decay_rate=time_decay_rate,
        )
        episode_items = list(payload["episodes"])
        previous_decision_t: Optional[int] = None
        for item in episode_items:
            if not isinstance(item, Mapping):
                raise ValueError("restored episode must be an object")
            decision_t = item.get("decision_t")
            if isinstance(decision_t, bool) or not isinstance(decision_t, int):
                raise ValueError("restored episode decision_t must be an integer")
            if previous_decision_t is not None and decision_t <= previous_decision_t:
                raise ValueError(
                    "restored episodes are not in strictly increasing decision order"
                )
            previous_decision_t = decision_t
        for item in episode_items:
            record = EpisodeRecord.from_dict(item)
            expected_episode_id = track.make_episode_id(
                track.run_id, track.seed, track.agent_id, record.decision_t
            )
            if (
                record.run_id != track.run_id
                or record.seed != track.seed
                or record.agent_id != track.agent_id
                or record.episode_id != expected_episode_id
                or record.decision_id != f"D:{expected_episode_id}"
            ):
                raise ValueError(
                    f"episode identity does not match restored track: {record.episode_id}"
                )
            if record.episode_id in track._ledger:
                raise ValueError(f"duplicate episode ID: {record.episode_id}")
            if track._ledger:
                previous = next(reversed(track._ledger.values()))
                if record.decision_t <= previous.decision_t:
                    raise ValueError(
                        "restored episodes are not in strictly increasing decision order"
                    )
            expected_advantage, expected_importance = _utility_statistics(
                [previous.flow_utility for previous in track._ledger.values()],
                record.flow_utility,
            )
            if not math.isclose(
                record.utility_advantage,
                expected_advantage,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"episode utility_advantage is inconsistent with its causal "
                    f"ledger prefix: {record.episode_id}"
                )
            if not math.isclose(
                record.importance,
                expected_importance,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"episode importance is inconsistent with its causal ledger "
                    f"prefix: {record.episode_id}"
                )
            track._ledger[record.episode_id] = record
        prompt_ids = list(payload["prompt_ids"])
        if any(not isinstance(episode_id, str) for episode_id in prompt_ids):
            raise ValueError("prompt buffer IDs must be strings")
        if len(prompt_ids) > track.prompt_capacity:
            raise ValueError("prompt buffer exceeds configured capacity")
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("prompt buffer contains duplicate episode IDs")
        if any(episode_id not in track._ledger for episode_id in prompt_ids):
            raise ValueError("prompt buffer references a missing episode")
        expected_prompt_ids = list(track._ledger)[-track.prompt_capacity :]
        if prompt_ids != expected_prompt_ids:
            raise ValueError(
                "prompt buffer must equal the canonical finalized-ledger suffix"
            )
        track._prompt_ids.extend(prompt_ids)
        for item in payload["pending"]:
            pending = PendingEpisode.from_dict(item)
            expected_episode_id = track.make_episode_id(
                track.run_id, track.seed, track.agent_id, pending.decision_t
            )
            if (
                pending.run_id != track.run_id
                or pending.seed != track.seed
                or pending.agent_id != track.agent_id
                or pending.episode_id != expected_episode_id
                or pending.decision_id != f"D:{expected_episode_id}"
            ):
                raise ValueError(
                    f"pending identity does not match restored track: {pending.decision_id}"
                )
            if pending.decision_id in track._pending:
                raise ValueError(f"duplicate pending decision ID: {pending.decision_id}")
            if pending.episode_id in track._ledger:
                raise ValueError(
                    f"pending decision collides with finalized episode: {pending.episode_id}"
                )
            known_decision_times = [
                *(record.decision_t for record in track._ledger.values()),
                *(existing.decision_t for existing in track._pending.values()),
            ]
            if known_decision_times and pending.decision_t <= max(known_decision_times):
                raise ValueError(
                    "restored pending decisions are not in strictly increasing "
                    "per-agent decision order"
                )
            track._pending[pending.decision_id] = pending
        track.validate_references()
        return track
