"""Orchestration facade for M1 + M2 + M3 verified dual-track memory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .m1_context import CausalContextRouter, ContextPacket, ContextRoute
from .m2_episodic import (
    EpisodeRecord,
    EvidenceLinkedEpisodicTrack,
    RetrievalHit,
)
from .m3_semantic import VerifiedRule, VerifiedSemanticRuleTrack


SYSTEM_SCHEMA_VERSION = "verified-dual-track-system-v1"


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryBundle:
    decision_t: int
    context_packet: ContextPacket
    context_route: ContextRoute
    episodic_hits: Tuple[RetrievalHit, ...]
    active_rules: Tuple[VerifiedRule, ...]
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
            "memory_prompt_hash": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
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
        self.run_id = str(run_id)
        self.seed = int(seed)
        self.agent_id = int(agent_id)
        self.context_router = context_router or CausalContextRouter(mode=context_mode)
        self.context_mode = context_mode
        # Validate mode via the router instead of duplicating its normalization.
        self.context_router.mode = self.context_router.route(
            self._bootstrap_packet_for_mode_validation(), mode=context_mode
        ).mode
        self.enable_episodic_retrieval = bool(enable_episodic_retrieval)
        self.enable_semantic = bool(enable_semantic)
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
        observation = _copy_json(context_observation)
        if observation.get("timestamp") != int(decision_t):
            raise ValueError("context observation timestamp must equal decision_t")
        if self._history and int(observation["timestamp"]) <= int(self._history[-1]["timestamp"]):
            raise ValueError("context observations must be strictly increasing")
        self._history.append(observation)
        packet = self.context_router.encode(
            self._history,
            decision_t=int(decision_t),
            observed_through=int(decision_t),
            event=event,
        )
        route = self.context_router.route(packet, mode=self.context_mode)
        hits: Tuple[RetrievalHit, ...] = ()
        if self.enable_episodic_retrieval:
            hits = self.episodic.retrieve(
                current_t=int(decision_t),
                current_state=retrieval_state,
                context_vector=route.retrieval_vector,
                use_context=route.to_retrieval,
                k=retrieval_k,
            )
        rules: Tuple[VerifiedRule, ...] = ()
        if self.semantic is not None:
            rules = self.semantic.retrieve(
                retrieval_state,
                current_t=int(decision_t),
                limit=rule_budget,
            )

        prompt_parts = []
        if route.to_prompt:
            prompt_parts.append(f"Causal context summary: {route.prompt_summary}")
        if hits:
            prompt_parts.append("Finalized experience evidence:")
            prompt_parts.extend(f"- {hit.episode.to_prompt_text()}" for hit in hits)
        if rules:
            prompt_parts.append("Verified active rules:")
            prompt_parts.extend(f"- {rule.to_prompt_text()}" for rule in rules)
        prompt = " ".join(prompt_parts)
        bundle_payload = {
            "decision_t": int(decision_t),
            "context_id": packet.context_id,
            "context_mode": route.mode,
            "episode_ids": [hit.episode.episode_id for hit in hits],
            "episode_scores": [hit.score for hit in hits],
            "rule_ids": [rule.rule_id for rule in rules],
            "prompt": prompt,
        }
        bundle = MemoryBundle(
            decision_t=int(decision_t),
            context_packet=packet,
            context_route=route,
            episodic_hits=hits,
            active_rules=rules,
            prompt=prompt,
            bundle_hash=_digest(bundle_payload),
        )
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
        bundle = self._prepared.pop(decision_t)
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
        decision_id = self._decision_ids.pop(decision_t)
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
    ) -> VerifiedRule:
        if self.semantic is None:
            raise RuntimeError("semantic track is disabled")
        return self.semantic.propose(
            raw_response,
            current_t=current_t,
            generator_id=generator_id,
        )

    def validate(self) -> None:
        if self._prepared or self._decision_ids or self.episodic.pending_count:
            raise ValueError("memory system contains an unfinished decision")
        self.episodic.validate_references()
        if self.semantic is not None:
            self.semantic.validate_referential_integrity()

    def to_dict(self) -> Dict[str, Any]:
        if self._prepared:
            raise ValueError("cannot serialize while a decision is prepared but not begun")
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
            "episodic": self.episodic.to_dict(),
            "semantic": self.semantic.to_dict() if self.semantic is not None else None,
            "decision_ids": dict(self._decision_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerifiedDualTrackMemory":
        if value.get("schema_version") != SYSTEM_SCHEMA_VERSION:
            raise ValueError("unsupported verified-memory system schema")
        router = CausalContextRouter.from_dict(value["context_router"])
        restored = cls(
            run_id=value["run_id"],
            seed=int(value["seed"]),
            agent_id=int(value["agent_id"]),
            context_router=router,
            context_mode=value["context_mode"],
            episodic_capacity=int(value["episodic"]["prompt_capacity"]),
            enable_episodic_retrieval=bool(value["enable_episodic_retrieval"]),
            enable_semantic=bool(value["enable_semantic"]),
        )
        restored._history = _copy_json(value.get("history", []))
        restored.episodic = EvidenceLinkedEpisodicTrack.from_dict(value["episodic"])
        if restored.enable_semantic:
            restored.semantic = VerifiedSemanticRuleTrack.from_dict(
                value["semantic"], episodic_track=restored.episodic
            )
        else:
            restored.semantic = None
        restored._decision_ids = {
            int(key): str(item) for key, item in value.get("decision_ids", {}).items()
        }
        return restored


__all__ = [
    "MemoryBundle",
    "SYSTEM_SCHEMA_VERSION",
    "VerifiedDualTrackMemory",
]
