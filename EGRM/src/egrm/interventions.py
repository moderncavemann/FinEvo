"""Hash-bound manifests for matched memory interventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Mapping


SCHEMA_VERSION = "egrm-matched-intervention-v1"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ProtectedDecision:
    checkpoint_hash: str
    environment_state_hash: str
    base_prompt_hash: str
    model_profile_hash: str
    decoding_profile_hash: str
    rng_state_hash: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class Treatment:
    name: str
    memory_payload_hash: str
    semantic_policy: str

    def __post_init__(self) -> None:
        if not self.name or not self.semantic_policy:
            raise ValueError("treatment name and semantic policy must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.memory_payload_hash):
            raise ValueError("memory_payload_hash must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class MatchedInterventionManifest:
    decision: ProtectedDecision
    treatments: tuple[Treatment, ...]

    def __post_init__(self) -> None:
        if len(self.treatments) < 2:
            raise ValueError("a matched intervention needs at least two treatments")
        names = [treatment.name for treatment in self.treatments]
        if len(set(names)) != len(names):
            raise ValueError("treatment names must be unique")

    @classmethod
    def create(
        cls,
        *,
        protected_fields: Mapping[str, str],
        treatment_payloads: Mapping[str, tuple[str, str]],
    ) -> "MatchedInterventionManifest":
        decision = ProtectedDecision(**dict(protected_fields))
        treatments = tuple(
            Treatment(
                name=name,
                memory_payload_hash=_digest(payload),
                semantic_policy=policy,
            )
            for name, (payload, policy) in sorted(treatment_payloads.items())
        )
        return cls(decision=decision, treatments=treatments)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "protected_decision": asdict(self.decision),
            "treatments": [asdict(treatment) for treatment in self.treatments],
        }
        payload["manifest_hash"] = _digest(payload)
        return payload

    def assert_compatible(self, other: "MatchedInterventionManifest") -> None:
        if self.decision != other.decision:
            raise ValueError("matched interventions differ in a protected field")
