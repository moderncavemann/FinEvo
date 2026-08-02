"""Orchestration facade for M1 + M2 + M3 verified dual-track memory."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .m1_context import CausalContextRouter, ContextPacket, ContextRoute
from .m2_episodic import (
    EpisodeRecord,
    EvidenceLinkedEpisodicTrack,
    RetrievalHit,
)
from .m3_semantic import VerifiedRule, VerifiedSemanticRuleTrack


SYSTEM_SCHEMA_VERSION = "verified-dual-track-system-v2"


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _semantic_state_digest(value: Mapping[str, Any]) -> str:
    """Match M3's UTF-8 canonical state hash for empty-query markers."""

    text = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryBundle:
    decision_t: int
    context_packet: ContextPacket
    context_route: ContextRoute
    episodic_hits: Tuple[RetrievalHit, ...]
    active_rules: Tuple[VerifiedRule, ...]
    protected_context_prompt: str
    memory_prompt: str
    prompt: str
    bundle_hash: str

    @property
    def retrieved_episode_ids(self) -> Tuple[str, ...]:
        return tuple(hit.episode.episode_id for hit in self.episodic_hits)

    @property
    def selected_rule_ids(self) -> Tuple[str, ...]:
        return tuple(rule.rule_id for rule in self.active_rules)

    def to_trace(self) -> Dict[str, Any]:
        return {
            "schema_version": SYSTEM_SCHEMA_VERSION,
            "decision_t": self.decision_t,
            "context_id": self.context_packet.context_id,
            "context_hash": self.context_packet.context_hash,
            "context_mode": self.context_route.mode,
            "context_to_retrieval": self.context_route.to_retrieval,
            "context_to_prompt": self.context_route.to_prompt,
            "retrieved_episode_ids": list(self.retrieved_episode_ids),
            "retrieval_scores": [hit.score for hit in self.episodic_hits],
            "score_components": [hit.components for hit in self.episodic_hits],
            "selected_rule_ids": list(self.selected_rule_ids),
            "memory_bundle_hash": self.bundle_hash,
            "protected_context_prompt_hash": hashlib.sha256(
                self.protected_context_prompt.encode("utf-8")
            ).hexdigest(),
            "memory_prompt_hash": hashlib.sha256(
                self.memory_prompt.encode("utf-8")
            ).hexdigest(),
            "combined_prompt_hash": hashlib.sha256(
                self.prompt.encode("utf-8")
            ).hexdigest(),
        }


class VerifiedDualTrackMemory:
    """One-agent facade that preserves the episodic/semantic dual-track logic.

    M1 is a router, not a third memory store. M2 and M3 remain the two memory
    tracks. The facade enforces one prepare/begin/finalize sequence per month.
    """

    def __init__(
        self,
        *,
        run_id: str,
        seed: int,
        agent_id: int,
        context_router: Optional[CausalContextRouter] = None,
        context_mode: str = "retrieval-only",
        episodic_capacity: int = 24,
        enable_episodic_retrieval: bool = True,
        enable_semantic: bool = True,
        semantic_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(agent_id, bool)
            or not isinstance(agent_id, int)
            or agent_id < 0
        ):
            raise ValueError("agent_id must be a non-negative integer")
        if context_router is not None and not isinstance(
            context_router, CausalContextRouter
        ):
            raise ValueError("context_router must be a CausalContextRouter or None")
        if not isinstance(context_mode, str):
            raise ValueError("context_mode must be a string")
        if (
            isinstance(episodic_capacity, bool)
            or not isinstance(episodic_capacity, int)
            or episodic_capacity < 1
        ):
            raise ValueError("episodic_capacity must be a positive integer")
        if not isinstance(enable_episodic_retrieval, bool):
            raise ValueError("enable_episodic_retrieval must be boolean")
        if not isinstance(enable_semantic, bool):
            raise ValueError("enable_semantic must be boolean")
        if semantic_config is not None and not isinstance(semantic_config, Mapping):
            raise ValueError("semantic_config must be an object or None")

        self.run_id = run_id
        self.seed = seed
        self.agent_id = agent_id
        self.context_router = context_router or CausalContextRouter(mode=context_mode)
        self.context_mode = context_mode
        # Validate mode via the router instead of duplicating its normalization.
        self.context_router.mode = self.context_router.route(
            self._bootstrap_packet_for_mode_validation(), mode=context_mode
        ).mode
        self.enable_episodic_retrieval = enable_episodic_retrieval
        self.enable_semantic = enable_semantic
        self.episodic = EvidenceLinkedEpisodicTrack(
            run_id=self.run_id,
            seed=self.seed,
            agent_id=self.agent_id,
            prompt_capacity=episodic_capacity,
        )
        self.semantic = (
            VerifiedSemanticRuleTrack(self.episodic, **dict(semantic_config or {}))
            if self.enable_semantic
            else None
        )
        self._history: list[Dict[str, Any]] = []
        # M1 events are a separate causal input channel.  Keep one exact,
        # canonical event (or ``None``) beside every history row so a restored
        # v2 checkpoint can reproduce the original ContextPacket rather than
        # silently rebuilding it with ``event=None``.
        self._decision_events: list[Optional[Dict[str, Any]]] = []
        # Preserve the exact M2/M3 query contract for each decision.  In
        # particular, ``pre_state`` is recorded later by M2 and is not a valid
        # substitute for the state that was actually passed to retrieval.
        self._decision_retrievals: list[Dict[str, Any]] = []
        self._prepared: Dict[int, MemoryBundle] = {}
        self._decision_ids: Dict[int, str] = {}

    def _bootstrap_packet_for_mode_validation(self) -> ContextPacket:
        """Build one private throwaway packet so M1 remains the mode authority."""
        row = {"timestamp": 0}
        for name in self.context_router.base_feature_names:
            row[name] = 0.0
        event = None
        if self.context_router.event_feature_names:
            event = {"timestamp": 0}
            for name in self.context_router.event_feature_names:
                event[name] = 0.0
        return self.context_router.encode([row], decision_t=0, event=event)

    @property
    def history(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(_copy_json(row) for row in self._history)

    @property
    def decision_events(self) -> Tuple[Optional[Dict[str, Any]], ...]:
        return tuple(_copy_json(event) for event in self._decision_events)

    @property
    def decision_retrievals(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(_copy_json(item) for item in self._decision_retrievals)

    @staticmethod
    def _canonical_decision_retrieval(value: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("decision retrieval input must be an object")
        expected_keys = {
            "retrieval_state",
            "retrieval_k",
            "rule_budget",
            "semantic_event_cursor",
        }
        if set(value) != expected_keys:
            raise ValueError("decision retrieval input keys are not exact")
        retrieval_state = value["retrieval_state"]
        if not isinstance(retrieval_state, Mapping):
            raise ValueError("retrieval_state must be a JSON object")
        try:
            canonical_state = _copy_json(dict(retrieval_state))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "retrieval_state must be JSON-compatible and contain only "
                "finite numbers"
            ) from exc
        if not isinstance(canonical_state, dict):
            raise ValueError("retrieval_state must be a JSON object")
        result: Dict[str, Any] = {"retrieval_state": canonical_state}
        for name in ("retrieval_k", "rule_budget"):
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            result[name] = item
        cursor = value["semantic_event_cursor"]
        if cursor is not None and (
            isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0
        ):
            raise ValueError(
                "semantic_event_cursor must be a non-negative integer or null"
            )
        result["semantic_event_cursor"] = cursor
        return result

    def _canonical_decision_event(
        self, event: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Return the exact event shape consumed by this router.

        M1 itself ignores undeclared event fields.  Checkpoints cannot do that:
        retaining ignored fields would create multiple serialized states for
        the same causal packet, while dropping them only during restore would
        make round trips non-exact.  Normalize to the registered schema before
        packet construction and storage.
        """

        feature_names = tuple(self.context_router.event_feature_names)
        if not feature_names:
            if event is not None:
                raise ValueError(
                    "event was provided but this router has no event_feature_names"
                )
            return None
        if event is None:
            return None
        if not isinstance(event, Mapping):
            raise ValueError("event must be an object")
        expected_keys = {"timestamp", *feature_names}
        if set(event) != expected_keys:
            raise ValueError("event does not match the registered event feature schema")
        timestamp = event["timestamp"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError("event.timestamp must be an integer")
        result: Dict[str, Any] = {"timestamp": timestamp}
        for feature in feature_names:
            value = event[feature]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"event[{feature!r}] must be a finite number")
            result[feature] = float(value)
        return result

    def prepare_decision(
        self,
        *,
        decision_t: int,
        context_observation: Mapping[str, Any],
        retrieval_state: Mapping[str, Any],
        event: Optional[Mapping[str, Any]] = None,
        retrieval_k: int = 5,
        rule_budget: int = 3,
    ) -> MemoryBundle:
        if decision_t in self._prepared or decision_t in self._decision_ids:
            raise ValueError(f"decision month {decision_t} was already prepared")
        semantic_event_cursor = (
            len(self.semantic.events) if self.semantic is not None else None
        )
        retrieval_input = self._canonical_decision_retrieval(
            {
                "retrieval_state": retrieval_state,
                "retrieval_k": retrieval_k,
                "rule_budget": rule_budget,
                "semantic_event_cursor": semantic_event_cursor,
            }
        )
        canonical_retrieval_state = retrieval_input["retrieval_state"]
        observation = _copy_json(context_observation)
        if observation.get("timestamp") != int(decision_t):
            raise ValueError("context observation timestamp must equal decision_t")
        if self._history and int(observation["timestamp"]) <= int(self._history[-1]["timestamp"]):
            raise ValueError("context observations must be strictly increasing")
        canonical_event = self._canonical_decision_event(event)
        candidate_history = [*self._history, observation]
        packet = self.context_router.encode(
            candidate_history,
            decision_t=int(decision_t),
            observed_through=int(decision_t),
            event=canonical_event,
        )
        route = self.context_router.route(packet, mode=self.context_mode)
        hits: Tuple[RetrievalHit, ...] = ()
        if self.enable_episodic_retrieval:
            hits = self.episodic.retrieve(
                current_t=int(decision_t),
                current_state=canonical_retrieval_state,
                context_vector=route.retrieval_vector,
                use_context=route.to_retrieval,
                k=retrieval_input["retrieval_k"],
            )
        rules: Tuple[VerifiedRule, ...] = ()
        if self.semantic is not None:
            rules = self.semantic.retrieve(
                canonical_retrieval_state,
                current_t=int(decision_t),
                limit=retrieval_input["rule_budget"],
                log_empty_selection=True,
            )

        protected_context_prompt = route.prompt_summary if route.to_prompt else ""
        memory_parts = []
        if hits:
            memory_parts.append("Finalized experience evidence:")
            memory_parts.extend(f"- {hit.episode.to_prompt_text()}" for hit in hits)
        if rules:
            memory_parts.append("Verified active rules:")
            memory_parts.extend(f"- {rule.to_prompt_text()}" for rule in rules)
        memory_prompt = " ".join(memory_parts)
        prompt_parts = []
        if protected_context_prompt:
            prompt_parts.append(
                f"Causal context summary: {protected_context_prompt}"
            )
        if memory_prompt:
            prompt_parts.append(memory_prompt)
        prompt = " ".join(prompt_parts)
        bundle_payload = {
            "decision_t": int(decision_t),
            "context_id": packet.context_id,
            "context_mode": route.mode,
            "episode_ids": [hit.episode.episode_id for hit in hits],
            "episode_scores": [hit.score for hit in hits],
            "rule_ids": [rule.rule_id for rule in rules],
            "protected_context_prompt": protected_context_prompt,
            "memory_prompt": memory_prompt,
            "prompt": prompt,
        }
        bundle = MemoryBundle(
            decision_t=int(decision_t),
            context_packet=packet,
            context_route=route,
            episodic_hits=hits,
            active_rules=rules,
            protected_context_prompt=protected_context_prompt,
            memory_prompt=memory_prompt,
            prompt=prompt,
            bundle_hash=_digest(bundle_payload),
        )
        # Commit history and prepared state only after packet construction,
        # retrieval, and prompt hashing all succeed. A rejected event/feature can
        # therefore be corrected and retried for the same decision month.
        self._history.append(observation)
        self._decision_events.append(canonical_event)
        self._decision_retrievals.append(retrieval_input)
        self._prepared[int(decision_t)] = bundle
        return bundle

    def begin_episode(
        self,
        *,
        decision_t: int,
        pre_state: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
        executed_action: Mapping[str, Any],
        reflection: str = "",
        rng_draw: Optional[float] = None,
    ) -> str:
        if decision_t not in self._prepared:
            raise ValueError("prepare_decision must run before begin_episode")
        bundle = self._prepared[decision_t]
        decision_id = self.episodic.begin_episode(
            decision_t=decision_t,
            pre_state=pre_state,
            context_id=bundle.context_packet.context_id,
            context_vector=bundle.context_packet.context_vector,
            retrieved_episode_ids=bundle.retrieved_episode_ids,
            selected_rule_ids=bundle.selected_rule_ids,
            proposed_action=proposed_action,
            executed_action=executed_action,
            rng_draw=rng_draw,
            reflection=reflection,
        )
        # Do not consume the prepared bundle until M2 has accepted every input.
        self._prepared.pop(decision_t)
        self._decision_ids[decision_t] = decision_id
        return decision_id

    def finalize_episode(
        self,
        *,
        decision_t: int,
        next_state: Mapping[str, Any],
        outcome: Mapping[str, Any],
        reward: float,
        flow_utility: float,
    ) -> EpisodeRecord:
        if decision_t not in self._decision_ids:
            raise ValueError("begin_episode must run before finalize_episode")
        decision_id = self._decision_ids[decision_t]
        record = self.episodic.finalize_episode(
            decision_id,
            outcome_t=int(decision_t) + 1,
            next_state=next_state,
            outcome=outcome,
            reward=reward,
            flow_utility=flow_utility,
        )
        if self.semantic is not None:
            for rule in tuple(self.semantic.rules):
                self.semantic.observe_episode(
                    rule.rule_id,
                    record.episode_id,
                    current_t=int(decision_t) + 1,
                )
        # Retain the facade mapping whenever M2 rejects malformed finalization
        # input, so callers can correct the input and retry the same month.
        self._decision_ids.pop(decision_t)
        return record

    def build_rule_proposal_prompt(self, *, max_episodes: int = 6) -> str:
        if self.semantic is None:
            raise RuntimeError("semantic track is disabled")
        return self.semantic.build_proposal_prompt(max_episodes=max_episodes)

    def submit_rule_proposal(
        self,
        raw_response: str,
        *,
        current_t: int,
        generator_id: str,
        semantic_policy: str = "evidence-grounded",
    ) -> VerifiedRule:
        if self.semantic is None:
            raise RuntimeError("semantic track is disabled")
        if semantic_policy == "unverified-immediate":
            return self.semantic.propose_unverified_immediate(
                raw_response,
                current_t=current_t,
                generator_id=generator_id,
            )
        if semantic_policy != "evidence-grounded":
            raise ValueError(f"unsupported semantic policy: {semantic_policy!r}")
        return self.semantic.propose(
            raw_response,
            current_t=current_t,
            generator_id=generator_id,
        )

    def validate(self) -> None:
        self._validate_runtime_integrity()
        if self._prepared or self._decision_ids or self.episodic.pending_count:
            raise ValueError("memory system contains an unfinished decision")

    def _validate_runtime_integrity(self) -> None:
        if (
            self.episodic.run_id != self.run_id
            or self.episodic.seed != self.seed
            or self.episodic.agent_id != self.agent_id
        ):
            raise ValueError("system and episodic track identities differ")

        pending_by_t: Dict[int, str] = {}
        for pending in self.episodic.pending_episodes:
            if pending.decision_t in pending_by_t:
                raise ValueError(
                    f"multiple pending decisions for month {pending.decision_t}"
                )
            pending_by_t[pending.decision_t] = pending.decision_id
        if pending_by_t != self._decision_ids:
            raise ValueError("facade decision IDs do not match M2 pending decisions")

        previous_timestamp: Optional[int] = None
        expected_history_keys = {
            "timestamp",
            *self.context_router.base_feature_names,
        }
        if len(self._decision_events) != len(self._history):
            raise ValueError(
                "decision_events must contain exactly one entry per context history row"
            )
        if len(self._decision_retrievals) != len(self._history):
            raise ValueError(
                "decision_retrievals must contain exactly one entry per context "
                "history row"
            )
        for index, row in enumerate(self._history):
            timestamp = row.get("timestamp") if isinstance(row, Mapping) else None
            if not isinstance(row, Mapping) or set(row) != expected_history_keys:
                raise ValueError(
                    f"history[{index}] does not match the registered feature schema"
                )
            if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                raise ValueError(f"history[{index}] has an invalid timestamp")
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("restored history timestamps are not strictly increasing")
            previous_timestamp = timestamp
            for feature in self.context_router.base_feature_names:
                feature_value = row[feature]
                if (
                    isinstance(feature_value, bool)
                    or not isinstance(feature_value, (int, float))
                    or not math.isfinite(float(feature_value))
                ):
                    raise ValueError(
                        f"history[{index}].{feature} must be a finite number"
                    )

        self.episodic.validate_references()
        if self.semantic is not None:
            self.semantic.validate_referential_integrity()
            rules_by_id = {
                rule.rule_id: rule for rule in self.semantic.rules
            }
            known_rule_ids = set(rules_by_id)
        else:
            rules_by_id = {}
            known_rule_ids = set()
        episodes = (*self.episodic.finalized_episodes, *self.episodic.pending_episodes)
        episode_times = sorted(episode.decision_t for episode in episodes)
        history_times = [int(row["timestamp"]) for row in self._history]
        if history_times != episode_times:
            raise ValueError(
                "context history timestamps do not match finalized/pending decisions"
            )
        episodes_by_t = {episode.decision_t: episode for episode in episodes}
        if len(episodes_by_t) != len(episodes):
            raise ValueError("multiple episodes share the same decision month")
        ordered_episodes = tuple(episodes_by_t[timestamp] for timestamp in history_times)
        retrieval_inputs_by_t: Dict[int, Dict[str, Any]] = {}
        for index, (row, stored_event, stored_retrieval) in enumerate(
            zip(
                self._history,
                self._decision_events,
                self._decision_retrievals,
                strict=True,
            )
        ):
            canonical_event = self._canonical_decision_event(stored_event)
            if canonical_event != stored_event:
                raise ValueError(
                    f"decision_events[{index}] is not in canonical event form"
                )
            canonical_retrieval = self._canonical_decision_retrieval(
                stored_retrieval
            )
            if canonical_retrieval != stored_retrieval:
                raise ValueError(
                    f"decision_retrievals[{index}] is not in canonical form"
                )
            semantic_event_cursor = canonical_retrieval["semantic_event_cursor"]
            if self.semantic is None:
                if semantic_event_cursor is not None:
                    raise ValueError(
                        "semantic_event_cursor must be null when M3 is disabled"
                    )
            elif semantic_event_cursor is None:
                raise ValueError(
                    "semantic_event_cursor is required when M3 is enabled"
                )
            decision_t = int(row["timestamp"])
            retrieval_inputs_by_t[decision_t] = canonical_retrieval
            expected_packet = self.context_router.encode(
                self._history[: index + 1],
                decision_t=decision_t,
                observed_through=decision_t,
                event=canonical_event,
            )
            episode = episodes_by_t[decision_t]
            if (
                episode.context_id != expected_packet.context_id
                or tuple(episode.context_vector)
                != tuple(expected_packet.context_vector)
            ):
                raise ValueError(
                    "episode context identity/vector does not match the "
                    f"reconstructed causal M1 packet at decision_t={decision_t}"
                )
            route = self.context_router.route(
                expected_packet, mode=self.context_mode
            )
            expected_retrieved_episode_ids: Tuple[str, ...] = ()
            if self.enable_episodic_retrieval:
                expected_retrieved_episode_ids = tuple(
                    hit.episode.episode_id
                    for hit in self.episodic.retrieve(
                        current_t=decision_t,
                        current_state=canonical_retrieval["retrieval_state"],
                        context_vector=route.retrieval_vector,
                        use_context=route.to_retrieval,
                        k=canonical_retrieval["retrieval_k"],
                    )
                )
            if tuple(episode.retrieved_episode_ids) != (
                expected_retrieved_episode_ids
            ):
                raise ValueError(
                    "episode retrieved_episode_ids do not reproduce the exact "
                    f"M2 top-k query at decision_t={decision_t}"
                )
        selected_rule_retrievals: Counter[tuple[int, str]] = Counter()
        for episode in episodes:
            missing_rules = sorted(set(episode.selected_rule_ids) - known_rule_ids)
            if missing_rules:
                raise ValueError(
                    f"episode/decision {episode.decision_id} references missing rules: "
                    f"{missing_rules}"
                )
            selected_rule_retrievals.update(
                (episode.decision_t, rule_id)
                for rule_id in episode.selected_rule_ids
            )

        logged_active_retrievals: Counter[tuple[int, str]] = Counter()
        logged_retrieval_order: Dict[int, list[str]] = {}
        semantic_state_by_cursor: Dict[
            int, tuple[Dict[str, str], Dict[str, Mapping[str, Any]], Dict[str, int]]
        ] = {}
        semantic_retrieval_event_indices: set[int] = set()
        if self.semantic is not None:
            semantic_events = self.semantic.events
            active_rule_ids: set[str] = set()
            for event_index, event in enumerate(semantic_events):
                rule_id = event.rule_id
                if event.event_type == "active_rule_retrieved":
                    # M3's own ledger validation checks lifecycle continuity.
                    # Rebuild the lifecycle here as well so the facade's
                    # episode-selection binding explicitly requires the rule
                    # to have been active at the retrieval event, rather than
                    # merely active in the final restored snapshot.
                    if rule_id is None or rule_id not in active_rule_ids:
                        raise ValueError(
                            "active-rule retrieval event references a rule that "
                            "was not active at that point in the lifecycle"
                        )
                    logged_active_retrievals[(event.timestamp, rule_id)] += 1
                    logged_retrieval_order.setdefault(event.timestamp, []).append(
                        rule_id
                    )
                    semantic_retrieval_event_indices.add(event_index)
                elif event.event_type == "active_rule_retrieval_empty":
                    semantic_retrieval_event_indices.add(event_index)
                if rule_id is not None and event.to_status is not None:
                    if event.to_status == "active":
                        active_rule_ids.add(rule_id)
                    else:
                        active_rule_ids.discard(rule_id)

            requested_cursors = sorted(
                {
                    item["semantic_event_cursor"]
                    for item in self._decision_retrievals
                }
            )
            if requested_cursors and requested_cursors[-1] > len(semantic_events):
                raise ValueError("semantic_event_cursor exceeds the M3 event ledger")

            previous_cursor: Optional[int] = None
            for decision_index, episode in enumerate(ordered_episodes):
                cursor = retrieval_inputs_by_t[episode.decision_t][
                    "semantic_event_cursor"
                ]
                if previous_cursor is not None and cursor < previous_cursor:
                    raise ValueError(
                        "semantic_event_cursor values must be non-decreasing in "
                        "decision order"
                    )
                if (
                    cursor > 0
                    and semantic_events[cursor - 1].timestamp > episode.decision_t
                ):
                    raise ValueError(
                        "semantic_event_cursor includes an M3 event from after "
                        f"decision_t={episode.decision_t}"
                    )
                if (
                    cursor < len(semantic_events)
                    and semantic_events[cursor].timestamp < episode.decision_t
                ):
                    raise ValueError(
                        "semantic_event_cursor skips an M3 event from before "
                        f"decision_t={episode.decision_t}"
                    )
                block_end = cursor + max(1, len(episode.selected_rule_ids))
                next_cursor = (
                    retrieval_inputs_by_t[
                        ordered_episodes[decision_index + 1].decision_t
                    ]["semantic_event_cursor"]
                    if decision_index + 1 < len(ordered_episodes)
                    else len(semantic_events)
                )
                if block_end > next_cursor:
                    raise ValueError(
                        "M3 retrieval block extends beyond the next decision's "
                        "semantic_event_cursor"
                    )
                previous_cursor = cursor

            historical_status: Dict[str, str] = {}
            historical_metrics: Dict[str, Mapping[str, Any]] = {}
            historical_updated_at: Dict[str, int] = {}
            event_index = 0
            mutating_event_types = {
                "candidate_verified",
                "candidate_rejected",
                "experimental_rule_injected_active",
                "rule_activated",
                "rule_retired",
            }
            for cursor in requested_cursors:
                while event_index < cursor:
                    event = semantic_events[event_index]
                    rule_id = event.rule_id
                    if rule_id is not None:
                        historical_metrics[rule_id] = dict(event.metrics)
                        if (
                            event.event_type in mutating_event_types
                            or event.event_type.endswith("_evidence_added")
                        ):
                            historical_updated_at[rule_id] = event.timestamp
                        if event.to_status is not None:
                            historical_status[rule_id] = event.to_status
                    event_index += 1
                semantic_state_by_cursor[cursor] = (
                    dict(historical_status),
                    dict(historical_metrics),
                    dict(historical_updated_at),
                )

        if selected_rule_retrievals != logged_active_retrievals:
            mismatches = sorted(
                set(selected_rule_retrievals) | set(logged_active_retrievals)
            )
            if mismatches:
                decision_t, rule_id = mismatches[0]
                raise ValueError(
                    f"episode selection for rule {rule_id} at decision_t={decision_t} "
                    "must bind to exactly one same-time active-rule retrieval event"
                )
            raise ValueError(
                "episode rule selections do not exactly match active-rule "
                "retrieval events"
            )

        bound_semantic_retrieval_event_indices: set[int] = set()
        for episode in ordered_episodes:
            selected_rule_ids = tuple(episode.selected_rule_ids)
            if tuple(logged_retrieval_order.get(episode.decision_t, ())) != (
                selected_rule_ids
            ):
                raise ValueError(
                    "episode selected_rule_ids do not preserve the same-time "
                    f"M3 retrieval order at decision_t={episode.decision_t}"
                )
            retrieval_input = retrieval_inputs_by_t[episode.decision_t]
            if self.semantic is not None:
                cursor = retrieval_input["semantic_event_cursor"]
                historical_status, historical_metrics, historical_updated_at = (
                    semantic_state_by_cursor[cursor]
                )
                eligible_rules = [
                    rule
                    for rule in rules_by_id.values()
                    if (
                        historical_status.get(rule.rule_id) == "active"
                        and rule.context_scope.matches(
                            retrieval_input["retrieval_state"]
                        )
                        and rule.condition.matches(
                            retrieval_input["retrieval_state"]
                        )
                    )
                ]
                try:
                    eligible_rules.sort(
                        key=lambda rule: (
                            -float(historical_metrics[rule.rule_id]["confidence"]),
                            -float(historical_metrics[rule.rule_id]["margin"]),
                            -historical_updated_at[rule.rule_id],
                            rule.rule_id,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "M3 event prefix cannot reproduce historical rule ranking"
                    ) from exc
                expected_rule_ids = tuple(
                    rule.rule_id
                    for rule in eligible_rules[
                        : retrieval_input["rule_budget"]
                    ]
                )
                if selected_rule_ids != expected_rule_ids:
                    raise ValueError(
                        "episode selected_rule_ids do not reproduce the exact M3 "
                        f"eligibility/ranking query at decision_t={episode.decision_t}"
                    )
                if selected_rule_ids:
                    retrieval_events = semantic_events[
                        cursor : cursor + len(selected_rule_ids)
                    ]
                    if tuple(
                        event.rule_id
                        for event in retrieval_events
                        if (
                            event.event_type == "active_rule_retrieved"
                            and event.timestamp == episode.decision_t
                        )
                    ) != selected_rule_ids or len(retrieval_events) != len(
                        selected_rule_ids
                    ):
                        raise ValueError(
                            "semantic_event_cursor does not point to the exact M3 "
                            f"retrieval block at decision_t={episode.decision_t}"
                        )
                    bound_semantic_retrieval_event_indices.update(
                        range(cursor, cursor + len(selected_rule_ids))
                    )
                else:
                    if cursor >= len(semantic_events):
                        raise ValueError(
                            "empty M3 retrieval is missing its exact ledger marker"
                        )
                    marker = semantic_events[cursor]
                    expected_provenance = {
                        "limit": retrieval_input["rule_budget"],
                        "state_hash": _semantic_state_digest(
                            retrieval_input["retrieval_state"]
                        ),
                    }
                    if (
                        marker.event_type != "active_rule_retrieval_empty"
                        or marker.timestamp != episode.decision_t
                        or dict(marker.provenance) != expected_provenance
                    ):
                        raise ValueError(
                            "semantic_event_cursor does not point to the exact "
                            f"empty M3 retrieval marker at decision_t={episode.decision_t}"
                        )
                    bound_semantic_retrieval_event_indices.add(cursor)

        if (
            bound_semantic_retrieval_event_indices
            != semantic_retrieval_event_indices
        ):
            raise ValueError(
                "M3 retrieval ledger contains an event or empty marker not bound "
                "to exactly one system decision"
            )

    def to_dict(self) -> Dict[str, Any]:
        if self._prepared:
            raise ValueError("cannot serialize while a decision is prepared but not begun")
        self._validate_runtime_integrity()
        return {
            "schema_version": SYSTEM_SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "agent_id": self.agent_id,
            "context_mode": self.context_mode,
            "enable_episodic_retrieval": self.enable_episodic_retrieval,
            "enable_semantic": self.enable_semantic,
            "context_router": self.context_router.to_dict(),
            "history": _copy_json(self._history),
            "decision_events": _copy_json(self._decision_events),
            "decision_retrievals": _copy_json(self._decision_retrievals),
            "episodic": self.episodic.to_dict(),
            "semantic": self.semantic.to_dict() if self.semantic is not None else None,
            "decision_ids": dict(self._decision_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerifiedDualTrackMemory":
        if not isinstance(value, Mapping):
            raise ValueError("verified-memory system snapshot must be an object")
        if value.get("schema_version") != SYSTEM_SCHEMA_VERSION:
            raise ValueError("unsupported verified-memory system schema")
        expected_keys = {
            "schema_version",
            "run_id",
            "seed",
            "agent_id",
            "context_mode",
            "enable_episodic_retrieval",
            "enable_semantic",
            "context_router",
            "history",
            "decision_events",
            "decision_retrievals",
            "episodic",
            "semantic",
            "decision_ids",
        }
        if set(value) != expected_keys:
            missing = sorted(expected_keys - set(value))
            extra = sorted(set(value) - expected_keys)
            raise ValueError(
                "verified-memory system snapshot keys are not exact: "
                f"missing={missing}, extra={extra}"
            )
        if not isinstance(value["run_id"], str) or not value["run_id"]:
            raise ValueError("system run_id must be a non-empty string")
        if (
            isinstance(value["seed"], bool)
            or not isinstance(value["seed"], int)
            or isinstance(value["agent_id"], bool)
            or not isinstance(value["agent_id"], int)
            or value["agent_id"] < 0
        ):
            raise ValueError("system seed/agent_id have invalid native types")
        if not isinstance(value["context_mode"], str):
            raise ValueError("system context_mode must be a string")
        for name in ("enable_episodic_retrieval", "enable_semantic"):
            if not isinstance(value[name], bool):
                raise ValueError(f"system {name} must be boolean")
        if not isinstance(value["context_router"], Mapping):
            raise ValueError("system context_router must be an object")
        if not isinstance(value["episodic"], Mapping):
            raise ValueError("system episodic state must be an object")
        if not isinstance(value["history"], list):
            raise ValueError("system history must be an array")
        if not isinstance(value["decision_events"], list):
            raise ValueError("system decision_events must be an array")
        if not isinstance(value["decision_retrievals"], list):
            raise ValueError("system decision_retrievals must be an array")
        if not isinstance(value["decision_ids"], Mapping):
            raise ValueError("system decision_ids must be an object")
        router = CausalContextRouter.from_dict(value["context_router"])
        restored = cls(
            run_id=value["run_id"],
            seed=value["seed"],
            agent_id=value["agent_id"],
            context_router=router,
            context_mode=value["context_mode"],
            episodic_capacity=value["episodic"].get("prompt_capacity"),
            enable_episodic_retrieval=value["enable_episodic_retrieval"],
            enable_semantic=value["enable_semantic"],
        )
        restored._history = _copy_json(value["history"])
        restored._decision_events = _copy_json(value["decision_events"])
        restored._decision_retrievals = _copy_json(value["decision_retrievals"])
        restored.episodic = EvidenceLinkedEpisodicTrack.from_dict(value["episodic"])
        if restored.enable_semantic:
            if value["semantic"] is None:
                raise ValueError("semantic track is enabled but serialized state is missing")
            if not isinstance(value["semantic"], Mapping):
                raise ValueError("enabled semantic state must be an object")
            restored.semantic = VerifiedSemanticRuleTrack.from_dict(
                value["semantic"], episodic_track=restored.episodic
            )
        else:
            if value["semantic"] is not None:
                raise ValueError("semantic track is disabled but serialized state is present")
            restored.semantic = None
        restored._decision_ids = {}
        for key, item in value["decision_ids"].items():
            if isinstance(key, bool):
                raise ValueError("decision month keys must be canonical integers")
            if isinstance(key, int):
                normalized_key = key
            elif (
                isinstance(key, str)
                and key.isdigit()
                and key == str(int(key))
            ):
                normalized_key = int(key)
            else:
                raise ValueError("decision month keys must be canonical integers")
            if normalized_key < 0 or not isinstance(item, str) or not item:
                raise ValueError("decision IDs must map nonnegative months to strings")
            if normalized_key in restored._decision_ids:
                raise ValueError("duplicate decision month after integer normalization")
            restored._decision_ids[normalized_key] = item
        restored._validate_runtime_integrity()
        return restored


__all__ = [
    "MemoryBundle",
    "SYSTEM_SCHEMA_VERSION",
    "VerifiedDualTrackMemory",
]
