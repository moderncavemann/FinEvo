import unittest
from pathlib import Path

from export_emnlp_outputs import _simulator_unemployment_pct, _trajectory_rows


class TrajectoryExportTest(unittest.TestCase):
    def test_sparse_annual_values_are_carried_forward(self):
        dense_log = {
            "world": [
                {"Price": 100.0},
                {
                    "Price": 101.0,
                    "Interest Rate": 0.04,
                    "Price Inflation": 0.06,
                    "Real GDP": 200.0,
                    "Real GDP Growth": 0.02,
                },
                {"Price": 102.0},
            ],
            "states": [
                {"0": {"inventory": {"Coin": 1.0}}},
                {"0": {"inventory": {"Coin": 2.0}}},
                {"0": {"inventory": {"Coin": 3.0}}},
            ],
            "actions": [
                {"0": {"SimpleLabor": 1}},
                {"0": {"SimpleLabor": 0}},
            ],
        }
        summary = {"final_metrics": {"sentiment_history": [0.0, 0.1, 0.2]}}

        rows = _trajectory_rows(dense_log, summary, "13", "m", "s", "v")

        self.assertEqual(rows[2]["inflation_pct"], 6.0)
        self.assertEqual(rows[2]["interest_rate"], 0.04)
        self.assertEqual(rows[2]["gdp"], 200.0)
        self.assertEqual(rows[2]["gdp_growth_pct"], 2.0)

    def test_unemployment_matches_simulator_year_to_date_definition(self):
        actions = [
            {"0": {"SimpleLabor": 1}, "1": {"SimpleLabor": 0}},
            {"0": {}, "1": {"SimpleLabor": 0}},
        ]

        value = _simulator_unemployment_pct(actions, month=2, num_agents=2)

        self.assertAlmostEqual(value, 12.5)  # 3 / (12 months * 2 agents)

    def test_sparse_months_are_not_gdp_growth_observations(self):
        from export_emnlp_outputs import _metrics_row

        dense_log = {
            "world": [
                {"Price": 100.0},
                {"Price": 101.0, "Real GDP Growth": 0.02, "Real GDP": 200.0},
                {"Price": 102.0},
                {"Price": 103.0, "Real GDP Growth": 0.04, "Real GDP": 208.0},
            ],
            "states": [
                {"0": {"inventory": {"Coin": float(month + 1)}}}
                for month in range(4)
            ],
            "actions": [{"0": {"SimpleLabor": 1}} for _ in range(3)],
        }
        summary = {
            "num_agents": 1,
            "episode_length": 3,
            "final_metrics": {
                "avg_wealth": 4.0,
                "median_wealth": 4.0,
                "gini": 0.0,
                "avg_unemployment": 0.0,
                "avg_inflation": 0.0,
            },
        }

        row = _metrics_row(
            Path("source"), dense_log, {}, summary, "E", "1", "m", "s", "v"
        )

        self.assertAlmostEqual(row["gdp_growth_mean_pct"], 3.0)
        self.assertAlmostEqual(row["gdp_volatility_pct"], 2 ** -0.5 * 2.0)


if __name__ == "__main__":
    unittest.main()
