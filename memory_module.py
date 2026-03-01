"""
Dual-Track Memory Module - ECCV 2026

Implements episodic memory + semantic memory dual-track structure (GAP 3 Fix)

Features:
- Episodic Memory: Recent experiences with importance scoring
- Semantic Memory: Consolidated strategy rules
- Memory retrieval based on recency, similarity, and importance
- Quarterly consolidation from episodic to semantic
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class EpisodicMemory:
    """Episodic memory item - specific experience"""
    timestamp: int
    economic_state: Dict
    personal_state: Dict
    decision: Dict
    outcome: Dict
    sentiment: float
    reflection: str = ""
    importance: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "economic_state": self.economic_state,
            "personal_state": self.personal_state,
            "decision": self.decision,
            "outcome": self.outcome,
            "sentiment": self.sentiment,
            "reflection": self.reflection,
            "importance": self.importance,
        }

    def to_prompt_text(self) -> str:
        """Convert to text embeddable in prompt"""
        return (
            f"[Month {self.timestamp}] "
            f"Price ${self.economic_state.get('price', 0):.2f}, "
            f"Interest {self.economic_state.get('interest_rate', 0)*100:.1f}%. "
            f"Your wealth ${self.personal_state.get('wealth', 0):.2f}. "
            f"Decision: work {self.decision.get('work', 0):.0%}, "
            f"consume {self.decision.get('consumption', 0):.0%}. "
            f"Result: wealth change ${self.outcome.get('wealth_change', 0):.2f}."
        )


@dataclass
class SemanticMemory:
    """Semantic memory item - consolidated strategy rule"""
    rule_id: str
    condition: str
    strategy: str
    confidence: float
    success_count: int = 0
    failure_count: int = 0
    last_updated: int = 0
    source_episodes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "condition": self.condition,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_updated": self.last_updated,
        }

    def to_prompt_text(self) -> str:
        return f"When {self.condition}, {self.strategy} (confidence: {self.confidence:.0%})"

    def update_confidence(self, success: bool):
        """Update confidence based on outcome"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = self.success_count / total


class DualTrackMemory:
    """
    Dual-Track Memory System

    Combines episodic memory (recent experiences) with semantic memory (learned rules)
    to provide context-aware decision support.
    """

    def __init__(
        self,
        agent_id: int,
        episodic_capacity: int = 24,
        semantic_capacity: int = 10,
        consolidation_interval: int = 3,
        time_decay_rate: float = 0.95,
    ):
        """
        Initialize dual-track memory system

        Args:
            agent_id: Unique agent identifier
            episodic_capacity: Maximum number of episodic memories
            semantic_capacity: Maximum number of semantic rules
            consolidation_interval: Months between consolidation
            time_decay_rate: Decay rate for recency scoring
        """
        self.agent_id = agent_id
        self.episodic_capacity = episodic_capacity
        self.semantic_capacity = semantic_capacity
        self.consolidation_interval = consolidation_interval
        self.time_decay_rate = time_decay_rate

        self.episodic_memories: deque = deque(maxlen=episodic_capacity)
        self.semantic_memories: Dict[str, SemanticMemory] = {}
        self.strategy_evolution: List[Dict] = []

    def add_episodic_memory(
        self,
        timestamp: int,
        economic_state: Dict,
        personal_state: Dict,
        decision: Dict,
        outcome: Dict,
        sentiment: float,
        reflection: str = "",
    ):
        """Add new episodic memory with automatic importance scoring"""
        # Calculate importance based on wealth change magnitude
        wealth_change = outcome.get('wealth_change', 0)
        income = personal_state.get('income', 1)
        importance = min(2.0, 1.0 + abs(wealth_change) / (income + 1e-8))

        memory = EpisodicMemory(
            timestamp=timestamp,
            economic_state=economic_state,
            personal_state=personal_state,
            decision=decision,
            outcome=outcome,
            sentiment=sentiment,
            reflection=reflection,
            importance=importance,
        )
        self.episodic_memories.append(memory)

    def retrieve_episodic_memories(
        self,
        current_state: Dict,
        k: int = 3,
        recency_weight: float = 0.4,
        similarity_weight: float = 0.4,
        importance_weight: float = 0.2,
    ) -> List[EpisodicMemory]:
        """
        Retrieve most relevant episodic memories

        Args:
            current_state: Current economic state
            k: Number of memories to retrieve
            recency_weight: Weight for recency scoring
            similarity_weight: Weight for similarity scoring
            importance_weight: Weight for importance scoring

        Returns:
            List of top-k relevant memories
        """
        if not self.episodic_memories:
            return []

        current_timestamp = current_state.get('timestamp', 0)
        scores = []

        for memory in self.episodic_memories:
            # Recency score (exponential decay)
            age = current_timestamp - memory.timestamp
            recency_score = self.time_decay_rate ** age

            # Similarity score (cosine similarity of state features)
            similarity_score = self._compute_state_similarity(current_state, memory.economic_state)

            # Importance score (normalized)
            importance_score = memory.importance / 2.0

            # Combined score
            total_score = (
                recency_weight * recency_score +
                similarity_weight * similarity_score +
                importance_weight * importance_score
            )
            scores.append((memory, total_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scores[:k]]

    def _compute_state_similarity(self, state1: Dict, state2: Dict) -> float:
        """Compute cosine similarity between two economic states"""
        features1 = self._extract_features(state1)
        features2 = self._extract_features(state2)

        dot_product = sum(a * b for a, b in zip(features1, features2))
        norm1 = np.sqrt(sum(a ** 2 for a in features1))
        norm2 = np.sqrt(sum(a ** 2 for a in features2))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _extract_features(self, state: Dict) -> List[float]:
        """Extract normalized features from economic state"""
        return [
            state.get('price', 1) / 100,
            state.get('interest_rate', 0.03),
            state.get('unemployment_rate', 0.05),
            state.get('inflation', 0) + 0.5,
            state.get('sentiment', 0) + 0.5,
        ]

    def consolidate_to_semantic(self, current_timestamp: int):
        """
        Consolidate recent episodic memories into semantic rules

        Analyzes patterns in recent experiences and updates strategy rules.
        """
        if len(self.episodic_memories) < 3:
            return

        recent_memories = list(self.episodic_memories)[-6:]

        # Analyze high inflation pattern
        high_inflation_memories = [
            m for m in recent_memories
            if m.economic_state.get('inflation', 0) > 0.03
        ]
        if high_inflation_memories:
            avg_consumption = np.mean([
                m.decision.get('consumption', 0.5) for m in high_inflation_memories
            ])
            avg_outcome = np.mean([
                m.outcome.get('wealth_change', 0) for m in high_inflation_memories
            ])

            rule_id = "high_inflation_strategy"
            if avg_outcome < 0:
                strategy = f"reduce consumption to {max(0.2, avg_consumption - 0.1):.0%}"
            else:
                strategy = f"maintain consumption around {avg_consumption:.0%}"

            self._update_semantic_rule(
                rule_id, "inflation is high (>3%)", strategy,
                avg_outcome > 0, current_timestamp,
                [m.timestamp for m in high_inflation_memories]
            )

        # Analyze high unemployment pattern
        high_unemployment_memories = [
            m for m in recent_memories
            if m.economic_state.get('unemployment_rate', 0) > 0.08
        ]
        if high_unemployment_memories:
            avg_work = np.mean([
                m.decision.get('work', 0.5) for m in high_unemployment_memories
            ])
            employed_count = sum(
                1 for m in high_unemployment_memories
                if m.personal_state.get('employed', False)
            )

            rule_id = "high_unemployment_strategy"
            success = employed_count > len(high_unemployment_memories) / 2

            if success:
                strategy = f"maintain high work propensity ({avg_work:.0%})"
            else:
                strategy = "be flexible in job search"

            self._update_semantic_rule(
                rule_id, "unemployment is high (>8%)", strategy,
                success, current_timestamp,
                [m.timestamp for m in high_unemployment_memories]
            )

    def _update_semantic_rule(
        self,
        rule_id: str,
        condition: str,
        strategy: str,
        success: bool,
        timestamp: int,
        source_episodes: List[int]
    ):
        """Update or create semantic memory rule"""
        if rule_id in self.semantic_memories:
            rule = self.semantic_memories[rule_id]
            rule.update_confidence(success)
            rule.strategy = strategy
            rule.last_updated = timestamp
            rule.source_episodes.extend(source_episodes)
        else:
            # Evict lowest confidence rule if at capacity
            if len(self.semantic_memories) >= self.semantic_capacity:
                min_rule = min(
                    self.semantic_memories.values(),
                    key=lambda r: r.confidence
                )
                del self.semantic_memories[min_rule.rule_id]

            self.semantic_memories[rule_id] = SemanticMemory(
                rule_id=rule_id,
                condition=condition,
                strategy=strategy,
                confidence=0.5,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                last_updated=timestamp,
                source_episodes=source_episodes,
            )

        # Track strategy evolution
        self.strategy_evolution.append({
            "timestamp": timestamp,
            "rule_id": rule_id,
            "strategy": strategy,
            "confidence": self.semantic_memories[rule_id].confidence,
        })

    def get_relevant_strategies(self, current_state: Dict) -> List[SemanticMemory]:
        """Get semantic strategies relevant to current state"""
        relevant = []

        inflation = current_state.get('inflation', 0)
        unemployment = current_state.get('unemployment_rate', 0.05)
        sentiment = current_state.get('sentiment', 0)

        for rule_id, rule in self.semantic_memories.items():
            if "high_inflation" in rule_id and inflation > 0.03:
                relevant.append(rule)
            elif "high_unemployment" in rule_id and unemployment > 0.08:
                relevant.append(rule)
            elif "optimistic" in rule_id and sentiment > 0.2:
                relevant.append(rule)
            elif "pessimistic" in rule_id and sentiment < -0.2:
                relevant.append(rule)

        relevant.sort(key=lambda r: r.confidence, reverse=True)
        return relevant

    def generate_memory_prompt(self, current_state: Dict) -> str:
        """Generate memory context prompt for LLM"""
        prompt_parts = []

        # Add relevant episodic memories
        relevant_episodes = self.retrieve_episodic_memories(current_state, k=2)
        if relevant_episodes:
            prompt_parts.append("Recent experiences:")
            for ep in relevant_episodes:
                prompt_parts.append(ep.to_prompt_text())

        # Add relevant strategies
        relevant_strategies = self.get_relevant_strategies(current_state)
        if relevant_strategies:
            prompt_parts.append("Learned strategies:")
            for strategy in relevant_strategies[:2]:
                prompt_parts.append(f"- {strategy.to_prompt_text()}")

        return " ".join(prompt_parts) if prompt_parts else ""

    def to_dict(self) -> Dict:
        """Serialize memory system to dict"""
        return {
            "agent_id": self.agent_id,
            "episodic_memories": [m.to_dict() for m in self.episodic_memories],
            "semantic_memories": {k: v.to_dict() for k, v in self.semantic_memories.items()},
            "strategy_evolution": self.strategy_evolution,
        }
