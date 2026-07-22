import json

from verified_memory.scripted_provider import ScriptedDiagnosticProvider


def test_scripted_action_is_timestamped_and_explicitly_synthetic() -> None:
    provider = ScriptedDiagnosticProvider()
    result = provider.get_structured_completion(
        [{"role": "user", "content": "This is monthly decision t=2."}]
    )
    payload = json.loads(result.text)
    assert payload["work"] == 0.75
    assert payload["consumption"] == 0.30
    assert "Synthetic" in payload["reflection"]
    assert result.provider == "diagnostic"
    assert result.usage.cost_usd == 0


def test_scripted_proposal_copies_real_episode_ids() -> None:
    provider = ScriptedDiagnosticProvider()
    evidence = [
        {"episode_id": "E0", "pre_state": {}, "executed_action": {}, "outcome": {}},
        {"episode_id": "E1", "pre_state": {}, "executed_action": {}, "outcome": {}},
    ]
    result = provider.get_structured_completion(
        [
            {
                "role": "user",
                "content": (
                    "Propose one semantic decision rule using only evidence.\n"
                    f"Evidence:\n{json.dumps(evidence)}"
                ),
            }
        ]
    )
    payload = json.loads(result.text)
    assert payload["supporting_episode_ids"] == ["E0", "E1"]
    assert set(payload) == {
        "condition",
        "action_guidance",
        "outcome_criterion",
        "rationale",
        "supporting_episode_ids",
    }
