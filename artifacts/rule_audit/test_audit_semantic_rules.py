import unittest

from artifacts.rule_audit.audit_semantic_rules import (
    _guidance_adherence,
    _infer_rule_snapshots,
    _summary_row,
)


def episode(timestamp, consumption, wealth_change, inflation=0.04):
    return {
        "timestamp": timestamp,
        "economic_state": {"inflation": inflation, "unemployment_rate": 0.01},
        "personal_state": {"employed": True},
        "decision": {"consumption": consumption, "work": 1.0},
        "outcome": {"wealth_change": wealth_change},
        "reflection": "observed reflection",
    }


class RuleAuditTest(unittest.TestCase):
    def test_create_merge_carry_and_reuse(self):
        episodes = {
            (0, 1): episode(1, 0.20, 10),
            (0, 2): episode(2, 0.20, 20),
            (0, 3): episode(3, 0.20, 30),
            (0, 4): episode(4, 0.20, 40),
            (0, 5): episode(5, 0.20, 50),
        }
        base = {
            "seed": 13,
            "model": "test",
            "agent_id": 0,
            "rule_id": "high_inflation_strategy",
            "condition": "inflation is high (>3%)",
            "action_guidance": "maintain consumption around 20%",
            "confidence": 0.5,
            "validity_note": "supported_by_episode",
            "rationale": "",
            "coded_category": [],
        }
        rows = [
            {**base, "month": 2, "source_episode_ids": ["E1", "E2"]},
            {
                **base,
                "month": 5,
                "confidence": 1.0,
                "source_episode_ids": ["E1", "E2", "E2", "E3", "E4", "E5"],
            },
            {
                **base,
                "month": 8,
                "confidence": 1.0,
                "source_episode_ids": ["E1", "E2", "E2", "E3", "E4", "E5"],
            },
        ]

        snapshots, updates = _infer_rule_snapshots("run", "test", 13, rows, episodes)
        self.assertEqual([row["event_type"] for row in snapshots], ["create", "merge_update", "carried_snapshot"])
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[1]["new_source_reused_reference_count"], 1)
        self.assertTrue(all(row["support_status"] == "verified_supported" for row in updates))

    def test_strategy_mismatch_is_contradicted(self):
        episodes = {(0, 1): episode(1, 0.20, 10)}
        rows = [{
            "month": 1,
            "seed": 13,
            "model": "test",
            "agent_id": 0,
            "rule_id": "high_inflation_strategy",
            "condition": "inflation is high (>3%)",
            "action_guidance": "reduce consumption to 20%",
            "confidence": 0.5,
            "source_episode_ids": ["E1"],
        }]
        _, updates = _infer_rule_snapshots("run", "test", 13, rows, episodes)
        self.assertEqual(updates[0]["support_status"], "mechanically_contradicted")
        self.assertEqual(updates[0]["strategy_match"], "false")

    def test_zero_rule_summary_preserves_no_denominator(self):
        inventory = {
            "run_id": "zero",
            "model": "test",
            "seed": 13,
            "num_agents": 10,
            "num_months": 24,
        }
        summary = _summary_row(inventory, [], [], [])
        self.assertEqual(summary["accepted_update_events_observed"], 0)
        self.assertIsNone(summary["mechanical_contradiction_rate"])
        self.assertIsNone(summary["proposed_candidate_events"])
        self.assertIsNone(summary["rejected_candidate_events"])

    def test_guidance_adherence(self):
        self.assertTrue(_guidance_adherence("reduce consumption to 20%", 0.22))
        self.assertFalse(_guidance_adherence("reduce consumption to 20%", 0.24))
        self.assertTrue(_guidance_adherence("maintain consumption around 10%", 0.12))
        self.assertFalse(_guidance_adherence("maintain consumption around 10%", 0.14))


if __name__ == "__main__":
    unittest.main()
