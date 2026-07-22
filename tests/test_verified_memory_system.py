import unittest

from verified_memory.m1_context import CausalContextRouter
from verified_memory.system import VerifiedDualTrackMemory


class VerifiedMemorySystemTest(unittest.TestCase):
    def make_system(self, mode="retrieval-only", semantic=True):
        router = CausalContextRouter(
            base_feature_names=("price", "inflation", "wealth"),
            window_size=3,
            mode=mode,
        )
        return VerifiedDualTrackMemory(
            run_id="smoke",
            seed=7,
            agent_id=0,
            context_router=router,
            context_mode=mode,
            enable_semantic=semantic,
        )

    def finish_month(self, system, t, *, utility=1.0):
        state = {"price": 100.0 + t, "inflation": 0.04, "wealth": 1000.0 + t}
        bundle = system.prepare_decision(
            decision_t=t,
            context_observation={"timestamp": t, **state},
            retrieval_state=state,
        )
        system.begin_episode(
            decision_t=t,
            pre_state=state,
            proposed_action={"work_propensity": 0.5, "consumption_fraction": 0.4},
            executed_action={
                "work_propensity": 0.5,
                "labor_hours": 80.0,
                "consumption_fraction": 0.4,
            },
        )
        record = system.finalize_episode(
            decision_t=t,
            next_state={**state, "wealth": state["wealth"] + 10.0},
            outcome={"wealth_change": 10.0},
            reward=0.1,
            flow_utility=utility,
        )
        return bundle, record

    def test_prepare_begin_finalize_contract(self):
        system = self.make_system()
        with self.assertRaises(ValueError):
            system.begin_episode(
                decision_t=0,
                pre_state={},
                proposed_action={},
                executed_action={},
            )
        bundle, record = self.finish_month(system, 0)
        self.assertEqual(record.outcome_t, 1)
        self.assertEqual(bundle.context_route.mode, "retrieval-only")
        self.assertFalse(bundle.context_route.to_prompt)
        system.validate()

    def test_prompt_only_never_routes_context_to_retrieval(self):
        system = self.make_system(mode="prompt-only", semantic=False)
        first, _ = self.finish_month(system, 0)
        second = system.prepare_decision(
            decision_t=1,
            context_observation={
                "timestamp": 1,
                "price": 101.0,
                "inflation": 0.04,
                "wealth": 1001.0,
            },
            retrieval_state={"price": 101.0, "inflation": 0.04, "wealth": 1001.0},
        )
        self.assertTrue(second.context_route.to_prompt)
        self.assertFalse(second.context_route.to_retrieval)
        self.assertIn("Causal context summary", second.prompt)
        self.assertEqual(
            second.episodic_hits[0].components["context_similarity"], 0.0
        )
        self.assertIn("Causal context summary", first.prompt)

    def test_active_rule_not_exposed_before_later_evidence(self):
        system = self.make_system()
        records = [self.finish_month(system, t, utility=1.0 + t)[1] for t in range(2)]
        raw = {
            "condition": {"field": "inflation", "operator": ">", "value": 0.03, "tolerance": 0.0},
            "action_guidance": {"target": "labor_hours", "direction": "increase", "threshold": 70.0, "tolerance": 0.0},
            "outcome_criterion": {"metric": "flow_utility", "operator": ">", "value": 0.0, "tolerance": 0.0},
            "rationale": "Both high-inflation episodes support higher labor.",
            "supporting_episode_ids": [record.episode_id for record in records],
        }
        import json

        rule = system.submit_rule_proposal(
            json.dumps(raw), current_t=2, generator_id="stub"
        )
        self.assertEqual(rule.status, "provisional")
        bundle, _ = self.finish_month(system, 2, utility=4.0)
        self.assertEqual(bundle.selected_rule_ids, ())
        active = system.semantic.get(rule.rule_id)
        self.assertEqual(active.status, "active")
        next_bundle = system.prepare_decision(
            decision_t=3,
            context_observation={"timestamp": 3, "price": 103.0, "inflation": 0.04, "wealth": 1003.0},
            retrieval_state={"price": 103.0, "inflation": 0.04, "wealth": 1003.0},
        )
        self.assertEqual(next_bundle.selected_rule_ids, (rule.rule_id,))
        self.assertIn("Verified active rules", next_bundle.prompt)

    def test_round_trip_after_finalized_month(self):
        system = self.make_system()
        self.finish_month(system, 0)
        restored = VerifiedDualTrackMemory.from_dict(system.to_dict())
        self.assertEqual(restored.to_dict(), system.to_dict())
        restored.validate()


if __name__ == "__main__":
    unittest.main()
