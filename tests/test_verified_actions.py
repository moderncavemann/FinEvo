import unittest

from verified_memory.actions import (
    ActionParseError,
    action_contract_prompt,
    parse_direct_action,
)
from verified_memory.m0_utility import UtilityConfig


class VerifiedActionTest(unittest.TestCase):
    def test_direct_hours_mapping_has_multiple_levels(self):
        result = parse_direct_action(
            '{"reflection":"balance", "work":0.5,"consumption":0.33}',
            labor_step=8,
        )
        self.assertEqual(result.labor_action_index, 11)
        self.assertEqual(result.executed_labor_hours, 88.0)
        self.assertEqual(result.consumption_action_index, 17)
        self.assertAlmostEqual(result.executed_consumption_rate, 0.34)
        self.assertEqual(result.environment_action(), [11, 17])

    def test_code_fence_repair_and_clipping_are_audited(self):
        result = parse_direct_action(
            'text```json\n{"work":1.2,"consumption":-0.1}\n```'
        )
        self.assertEqual(result.repair_attempts, 1)
        self.assertTrue(result.clipped)
        self.assertEqual(result.executed_labor_hours, 168.0)
        self.assertEqual(result.executed_consumption_rate, 0.0)

    def test_invalid_or_missing_schema_fails_closed(self):
        for content in ("not json", '{"work":0.5}', '{"work":"many","consumption":0.2}'):
            with self.subTest(content=content), self.assertRaises(ActionParseError):
                parse_direct_action(content)

    def test_objective_prompt_names_tradeoff_and_schema(self):
        prompt = action_contract_prompt(
            UtilityConfig(rho=1.0, labor_weight=2.0, inverse_frisch=1.5)
        )
        self.assertIn("flow utility", prompt)
        self.assertIn("labor effort", prompt)
        self.assertIn("wealth and macro variables are diagnostics", prompt)
        self.assertIn("Return ONLY JSON", prompt)


if __name__ == "__main__":
    unittest.main()
