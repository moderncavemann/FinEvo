import unittest

from verified_memory.m0_utility import UtilityConfig
from verified_memory.prompts import (
    MEMORY_END,
    MEMORY_START,
    DecisionPromptState,
    build_base_decision_prompt,
    compose_decision_prompt,
)


class VerifiedPromptTest(unittest.TestCase):
    def state(self):
        return DecisionPromptState(
            decision_t=2,
            agent_id=0,
            name="Ada",
            age=35,
            city="Sydney",
            job="Analyst",
            offer="Analyst",
            wealth=100.0,
            skill=2.0,
            price=1.2,
            interest_rate=0.03,
            last_consumption_quantity=4.0,
            last_labor_hours=80.0,
            last_tax_paid=2.0,
            last_lump_sum=1.0,
        )

    def test_only_memory_block_changes(self):
        base = build_base_decision_prompt(self.state(), UtilityConfig())
        left = compose_decision_prompt(base, "episode A")
        right = compose_decision_prompt(base, "episode B")
        self.assertEqual(left.base_prompt_hash, right.base_prompt_hash)
        self.assertNotEqual(left.memory_hash, right.memory_hash)
        self.assertEqual(left.full_prompt.count(MEMORY_START), 1)
        self.assertEqual(left.full_prompt.count(MEMORY_END), 1)
        self.assertEqual(
            left.full_prompt.split(MEMORY_START)[0],
            right.full_prompt.split(MEMORY_START)[0],
        )
        self.assertEqual(
            left.full_prompt.split(MEMORY_END)[1],
            right.full_prompt.split(MEMORY_END)[1],
        )

    def test_contract_explains_direct_hours_and_utility(self):
        base = build_base_decision_prompt(self.state(), UtilityConfig())
        self.assertIn("fraction of maximum hours", base)
        self.assertIn("realized flow utility", base)
        self.assertIn("wealth and macro variables are diagnostics", base)

    def test_base_prompt_has_no_memory_delimiter(self):
        base = build_base_decision_prompt(self.state(), UtilityConfig())
        self.assertNotIn(MEMORY_START, base)
        self.assertNotIn(MEMORY_END, base)

    def test_bootstrap_prompt_marks_prior_period_unavailable(self):
        values = self.state().to_dict()
        values.update(
            decision_t=0,
            previous_period_available=False,
            last_consumption_quantity=0.0,
            last_labor_hours=0.0,
            last_tax_paid=0.0,
            last_lump_sum=0.0,
        )
        base = build_base_decision_prompt(
            DecisionPromptState(**values),
            UtilityConfig(),
            causal_context_summary="prior_low_labor_rate=unavailable",
        )
        self.assertIn("No completed prior month is available", base)
        self.assertIn("prior_low_labor_rate=unavailable", base)
        self.assertNotIn("last completed month you worked 0", base)

    def test_reserved_delimiters_rejected(self):
        with self.assertRaises(ValueError):
            compose_decision_prompt("choose carefully", MEMORY_START)

    def test_invalid_state_rejected(self):
        values = self.state().to_dict()
        values["price"] = 0
        with self.assertRaises(ValueError):
            DecisionPromptState(**values)


if __name__ == "__main__":
    unittest.main()
