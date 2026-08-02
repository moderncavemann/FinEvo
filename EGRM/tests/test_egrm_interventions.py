import hashlib

import pytest

from egrm.interventions import MatchedInterventionManifest, Treatment


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def protected(label: str = "same"):
    return {
        "checkpoint_hash": digest(f"{label}-checkpoint"),
        "environment_state_hash": digest(f"{label}-environment"),
        "base_prompt_hash": digest(f"{label}-prompt"),
        "model_profile_hash": digest(f"{label}-model"),
        "decoding_profile_hash": digest(f"{label}-decoding"),
        "rng_state_hash": digest(f"{label}-rng"),
    }


def test_manifest_binds_one_protected_decision_to_distinct_treatments():
    manifest = MatchedInterventionManifest.create(
        protected_fields=protected(),
        treatment_payloads={
            "matched": ("episode and rule", "egrm"),
            "no-memory": ("", "no-memory"),
        },
    )
    payload = manifest.to_dict()
    assert len(payload["manifest_hash"]) == 64
    assert len({row["memory_payload_hash"] for row in payload["treatments"]}) == 2


def test_compatibility_rejects_protected_field_drift():
    left = MatchedInterventionManifest.create(
        protected_fields=protected("left"),
        treatment_payloads={"a": ("a", "egrm"), "b": ("b", "egrm")},
    )
    right = MatchedInterventionManifest.create(
        protected_fields=protected("right"),
        treatment_payloads={"a": ("a", "egrm"), "b": ("b", "egrm")},
    )
    with pytest.raises(ValueError, match="protected field"):
        left.assert_compatible(right)


def test_protected_and_treatment_hashes_must_be_lowercase_sha256():
    changed = protected()
    changed["checkpoint_hash"] = "z" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        MatchedInterventionManifest.create(
            protected_fields=changed,
            treatment_payloads={"a": ("a", "egrm"), "b": ("b", "egrm")},
        )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        Treatment(name="a", memory_payload_hash="A" * 64, semantic_policy="egrm")
