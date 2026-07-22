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


SCHEMA_VERSION = "m2-episodic-v1"


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible copy."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


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
        values = dict(payload)
        values["context_vector"] = tuple(float(v) for v in values.get("context_vector", []))
        values["retrieved_episode_ids"] = tuple(values.get("retrieved_episode_ids", []))
        values["selected_rule_ids"] = tuple(values.get("selected_rule_ids", []))
        return cls(**values)

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
        if prompt_capacity < 1:
            raise ValueError("prompt_capacity must be positive")
        if not 0.0 < time_decay_rate <= 1.0:
            raise ValueError("time_decay_rate must be in (0, 1]")
        self.run_id = str(run_id)
        self.seed = int(seed)
        self.agent_id = int(agent_id)
        self.prompt_capacity = int(prompt_capacity)
        self.time_decay_rate = float(time_decay_rate)
        self._pending: Dict[str, PendingEpisode] = {}
        self._ledger: "OrderedDict[str, EpisodeRecord]" = OrderedDict()
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
        if int(decision_t) < 0:
            raise ValueError("decision_t must be non-negative")
        episode_id = self.make_episode_id(
            self.run_id, self.seed, self.agent_id, int(decision_t)
        )
        decision_id = f"D:{episode_id}"
        if decision_id in self._pending or episode_id in self._ledger:
            raise ValueError(f"duplicate decision/episode id: {decision_id}")
        if rng_draw is not None and not 0.0 <= float(rng_draw) <= 1.0:
            raise ValueError("rng_draw must be in [0, 1]")
        vector = tuple(float(value) for value in context_vector)
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("context_vector must contain only finite values")
        self._pending[decision_id] = PendingEpisode(
            decision_id=decision_id,
            episode_id=episode_id,
            run_id=self.run_id,
            seed=self.seed,
            agent_id=self.agent_id,
            decision_t=int(decision_t),
            pre_state=_jsonable(pre_state),
            context_id=str(context_id),
            context_vector=vector,
            retrieved_episode_ids=tuple(dict.fromkeys(retrieved_episode_ids)),
            selected_rule_ids=tuple(dict.fromkeys(selected_rule_ids)),
            proposed_action=_jsonable(proposed_action),
            executed_action=_jsonable(executed_action),
            rng_draw=None if rng_draw is None else float(rng_draw),
            reflection=str(reflection),
        )
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
        if int(outcome_t) != pending.decision_t + 1:
            raise ValueError(
                f"outcome_t must equal decision_t + 1; got {outcome_t} for {pending.decision_t}"
            )
        reward_value = float(reward)
        utility_value = float(flow_utility)
        if not math.isfinite(reward_value) or not math.isfinite(utility_value):
            raise ValueError("reward and flow_utility must be finite")

        # Consume the staged transition only after every basic finalization
        # invariant has passed, so a malformed caller can correct and retry.
        self._pending.pop(decision_id)

        previous_utilities = [record.flow_utility for record in self._ledger.values()]
        baseline = statistics.median(previous_utilities) if previous_utilities else 0.0
        utility_advantage = utility_value - baseline
        scale = statistics.median(
            [abs(value - baseline) for value in previous_utilities]
        ) if previous_utilities else 1.0
        scale = max(float(scale), 1e-6)
        importance = 1.0 + min(1.0, abs(utility_advantage) / scale)

        core = {
            "schema_version": SCHEMA_VERSION,
            **asdict(pending),
            "outcome_t": int(outcome_t),
            "next_state": _jsonable(next_state),
            "outcome": _jsonable(outcome),
            "reward": reward_value,
            "flow_utility": utility_value,
            "utility_advantage": utility_advantage,
            "importance": importance,
        }
        record = EpisodeRecord(**core, record_hash=_stable_hash(core))
        self._ledger[record.episode_id] = record
        self._prompt_ids.append(record.episode_id)
        return record

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def finalized_count(self) -> int:
        return len(self._ledger)

    @property
    def finalized_episodes(self) -> Tuple[EpisodeRecord, ...]:
        return tuple(self._ledger.values())

    def get(self, episode_id: str) -> Optional[EpisodeRecord]:
        return self._ledger.get(episode_id)

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
        for episode_id in self._prompt_ids:
            episode = self._ledger[episode_id]
            age = max(0, int(current_t) - episode.decision_t)
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
            hits.append(RetrievalHit(episode=episode, score=score, components=components))
        hits.sort(key=lambda hit: (-hit.score, -hit.episode.decision_t, hit.episode.episode_id))
        return tuple(hits[:k])

    def validate_references(self) -> None:
        for episode in self._ledger.values():
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
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported M2 schema version")
        track = cls(
            run_id=str(payload["run_id"]),
            seed=int(payload["seed"]),
            agent_id=int(payload["agent_id"]),
            prompt_capacity=int(payload["prompt_capacity"]),
            time_decay_rate=float(payload["time_decay_rate"]),
        )
        for item in payload.get("episodes", []):
            record = EpisodeRecord.from_dict(item)
            track._ledger[record.episode_id] = record
        prompt_ids = payload.get("prompt_ids", [])
        if any(episode_id not in track._ledger for episode_id in prompt_ids):
            raise ValueError("prompt buffer references a missing episode")
        track._prompt_ids.extend(prompt_ids)
        for item in payload.get("pending", []):
            values = dict(item)
            values["context_vector"] = tuple(values.get("context_vector", []))
            values["retrieved_episode_ids"] = tuple(values.get("retrieved_episode_ids", []))
            values["selected_rule_ids"] = tuple(values.get("selected_rule_ids", []))
            pending = PendingEpisode(**values)
            track._pending[pending.decision_id] = pending
        return track
