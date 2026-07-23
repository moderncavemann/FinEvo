import copy
import hashlib
import json
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

    def make_system_with_active_selection(self):
        system = self.make_system()
        records = [self.finish_month(system, t, utility=1.0 + t)[1] for t in range(2)]
        raw = {
            "context_scope": {"scope_id": "global", "predicates": []},
            "condition": {
                "field": "inflation",
                "operator": ">",
                "value": 0.03,
                "tolerance": 0.0,
            },
            "action_guidance": {
                "target": "labor_hours",
                "direction": "at_least",
                "threshold": 70.0,
                "tolerance": 0.0,
            },
            "rationale": "The finalized transitions support higher labor.",
            "supporting_episode_ids": [record.episode_id for record in records],
        }
        rule = system.submit_rule_proposal(
            json.dumps(raw), current_t=2, generator_id="stub"
        )
        self.finish_month(system, 2, utility=4.0)
        selected, _ = self.finish_month(system, 3, utility=5.0)
        self.assertEqual(selected.selected_rule_ids, (rule.rule_id,))
        return system, rule

    @staticmethod
    def rehash_episode(row):
        core = dict(row)
        core.pop("record_hash")
        encoded = json.dumps(
            core, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        row["record_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

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

    def test_constructor_rejects_implicit_native_type_coercions(self):
        router = CausalContextRouter(
            base_feature_names=("price",), window_size=2
        )
        valid = {
            "run_id": "typed",
            "seed": 7,
            "agent_id": 0,
            "context_router": router,
            "context_mode": "retrieval-only",
            "episodic_capacity": 4,
            "enable_episodic_retrieval": True,
            "enable_semantic": False,
        }
        invalid_values = {
            "run_id": 7,
            "seed": "7",
            "agent_id": "0",
            "context_mode": 1,
            "episodic_capacity": 4.0,
            "enable_episodic_retrieval": 1,
            "enable_semantic": 0,
            "semantic_config": [],
        }
        for field, invalid in invalid_values.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    VerifiedDualTrackMemory(**{**valid, field: invalid})

    def test_failed_prepare_begin_and_finalize_are_retry_safe(self):
        system = self.make_system(semantic=False)
        state = {"price": 100.0, "inflation": 0.04, "wealth": 1000.0}

        # This router declares no event features, so the event is rejected after
        # observation validation. The failed attempt must not consume month zero.
        with self.assertRaises(ValueError):
            system.prepare_decision(
                decision_t=0,
                context_observation={"timestamp": 0, **state},
                retrieval_state=state,
                event={"timestamp": 0},
            )
        self.assertEqual(system.history, ())
        system.prepare_decision(
            decision_t=0,
            context_observation={"timestamp": 0, **state},
            retrieval_state=state,
        )

        with self.assertRaises(ValueError):
            system.begin_episode(
                decision_t=0,
                pre_state=state,
                proposed_action={},
                executed_action={},
                rng_draw=2.0,
            )
        system.begin_episode(
            decision_t=0,
            pre_state=state,
            proposed_action={},
            executed_action={},
        )

        with self.assertRaises(ValueError):
            system.finalize_episode(
                decision_t=0,
                next_state={"wealth": 1001.0},
                outcome={},
                reward=float("nan"),
                flow_utility=0.0,
            )
        self.assertEqual(system.episodic.pending_count, 1)
        record = system.finalize_episode(
            decision_t=0,
            next_state={"wealth": 1001.0},
            outcome={},
            reward=0.0,
            flow_utility=0.0,
        )
        self.assertEqual(record.outcome_t, 1)
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
        self.assertTrue(second.protected_context_prompt)
        self.assertNotIn("Causal context summary", second.memory_prompt)
        self.assertEqual(
            second.episodic_hits[0].components["context_similarity"], 0.0
        )
        self.assertIn("Causal context summary", first.prompt)

    def test_active_rule_not_exposed_before_later_evidence(self):
        system = self.make_system()
        records = [self.finish_month(system, t, utility=1.0 + t)[1] for t in range(2)]
        raw = {
            "context_scope": {"scope_id": "global", "predicates": []},
            "condition": {"field": "inflation", "operator": ">", "value": 0.03, "tolerance": 0.0},
            "action_guidance": {"target": "labor_hours", "direction": "at_least", "threshold": 70.0, "tolerance": 0.0},
            "rationale": "Both high-inflation episodes support higher labor.",
            "supporting_episode_ids": [record.episode_id for record in records],
        }
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
        self.assertEqual(restored.decision_events, (None,))
        self.assertEqual(
            restored.decision_retrievals,
            (
                {
                    "retrieval_state": {
                        "inflation": 0.04,
                        "price": 100.0,
                        "wealth": 1000.0,
                    },
                    "retrieval_k": 5,
                    "rule_budget": 3,
                    "semantic_event_cursor": 0,
                },
            ),
        )
        restored.validate()

    def test_current_system_snapshot_requires_history_and_decision_arrays(self):
        system = self.make_system(semantic=False)
        self.finish_month(system, 0)
        for missing in (
            "history",
            "decision_events",
            "decision_retrievals",
            "decision_ids",
        ):
            payload = system.to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, "keys are not exact"):
                    VerifiedDualTrackMemory.from_dict(payload)

    def test_restore_replays_exact_m2_top_k_from_decision_query(self):
        system = self.make_system(semantic=False)
        self.finish_month(system, 0)
        self.finish_month(system, 1)
        payload = system.to_dict()
        payload["decision_retrievals"][1]["retrieval_k"] = 0

        with self.assertRaisesRegex(ValueError, "exact M2 top-k query"):
            VerifiedDualTrackMemory.from_dict(payload)

    def test_restore_replays_m2_with_exact_historical_retrieval_state(self):
        system = self.make_system(mode="no-context", semantic=False)
        old_state = {"price": 0.0, "inflation": 0.0, "wealth": 0.0}
        new_state = {
            "price": 10_000.0,
            "inflation": 1.0,
            "wealth": 100_000_000.0,
        }
        for t, state in enumerate((old_state, new_state)):
            system.prepare_decision(
                decision_t=t,
                context_observation={"timestamp": t, **state},
                retrieval_state=state,
                retrieval_k=1,
            )
            system.begin_episode(
                decision_t=t,
                pre_state=state,
                proposed_action={},
                executed_action={},
            )
            system.finalize_episode(
                decision_t=t,
                next_state=state,
                outcome={},
                reward=0.0,
                flow_utility=0.0,
            )
        selected = system.prepare_decision(
            decision_t=2,
            context_observation={"timestamp": 2, **old_state},
            retrieval_state=old_state,
            retrieval_k=1,
        )
        self.assertEqual(
            selected.retrieved_episode_ids,
            (system.episodic.finalized_episodes[0].episode_id,),
        )
        system.begin_episode(
            decision_t=2,
            pre_state=old_state,
            proposed_action={},
            executed_action={},
        )
        payload = system.to_dict()
        payload["decision_retrievals"][2]["retrieval_state"] = new_state

        with self.assertRaisesRegex(ValueError, "exact M2 top-k query"):
            VerifiedDualTrackMemory.from_dict(payload)

    def test_restore_rebuilds_m1_and_rejects_rehashed_context_vector_tamper(self):
        system = self.make_system(semantic=False)
        self.finish_month(system, 0, utility=0.0)
        self.finish_month(system, 1, utility=0.0)
        payload = system.to_dict()
        first, second = payload["episodic"]["episodes"]
        first["context_vector"], second["context_vector"] = (
            second["context_vector"],
            first["context_vector"],
        )
        self.rehash_episode(first)
        self.rehash_episode(second)

        with self.assertRaisesRegex(ValueError, "reconstructed causal M1 packet"):
            VerifiedDualTrackMemory.from_dict(payload)

    def test_event_enabled_round_trip_and_causal_event_binding(self):
        router = CausalContextRouter(
            base_feature_names=("price",),
            event_feature_names=("shock",),
            window_size=2,
            mode="full",
        )
        system = VerifiedDualTrackMemory(
            run_id="event-smoke",
            seed=7,
            agent_id=0,
            context_router=router,
            context_mode="full",
            enable_semantic=False,
        )
        system.prepare_decision(
            decision_t=0,
            context_observation={"timestamp": 0, "price": 100.0},
            retrieval_state={"price": 100.0},
            event={"timestamp": 0, "shock": 0.5},
        )
        system.begin_episode(
            decision_t=0,
            pre_state={"price": 100.0},
            proposed_action={},
            executed_action={},
        )
        system.finalize_episode(
            decision_t=0,
            next_state={"price": 101.0},
            outcome={},
            reward=0.0,
            flow_utility=0.0,
        )

        payload = system.to_dict()
        self.assertEqual(
            payload["decision_events"],
            [{"timestamp": 0, "shock": 0.5}],
        )
        restored = VerifiedDualTrackMemory.from_dict(copy.deepcopy(payload))
        self.assertEqual(restored.to_dict(), payload)

        payload["decision_events"][0]["shock"] = -0.5
        with self.assertRaisesRegex(ValueError, "reconstructed causal M1 packet"):
            VerifiedDualTrackMemory.from_dict(payload)

    def test_restore_rejects_history_that_no_longer_covers_episode_times(self):
        system = self.make_system(semantic=False)
        self.finish_month(system, 0)
        payload = system.to_dict()
        payload["history"] = []

        with self.assertRaisesRegex(
            ValueError, "decision_events|history timestamps"
        ):
            VerifiedDualTrackMemory.from_dict(payload)

    def test_pending_round_trip_requires_exact_facade_to_m2_mapping(self):
        system = self.make_system(semantic=False)
        state = {"price": 100.0, "inflation": 0.04, "wealth": 1000.0}
        system.prepare_decision(
            decision_t=0,
            context_observation={"timestamp": 0, **state},
            retrieval_state=state,
        )
        system.begin_episode(
            decision_t=0,
            pre_state=state,
            proposed_action={},
            executed_action={},
        )
        payload = system.to_dict()
        restored = VerifiedDualTrackMemory.from_dict(payload)
        restored.finalize_episode(
            decision_t=0,
            next_state={"wealth": 1001.0},
            outcome={},
            reward=0.0,
            flow_utility=0.0,
        )
        restored.validate()

        payload["decision_ids"] = {}
        with self.assertRaisesRegex(ValueError, "do not match M2 pending"):
            VerifiedDualTrackMemory.from_dict(payload)

    def test_round_trip_binds_selected_rule_to_same_time_active_retrieval(self):
        system, _ = self.make_system_with_active_selection()

        restored = VerifiedDualTrackMemory.from_dict(system.to_dict())

        self.assertEqual(restored.to_dict(), system.to_dict())

    def test_restore_rejects_cursor_that_includes_future_semantic_event(self):
        system = self.make_system()
        self.finish_month(system, 0)
        with self.assertRaises(ValueError):
            system.submit_rule_proposal(
                "not valid candidate JSON",
                current_t=1,
                generator_id="invalid-fixture",
            )
        payload = system.to_dict()
        self.assertGreater(
            len(payload["semantic"]["events"]),
            payload["decision_retrievals"][0]["semantic_event_cursor"],
        )
        payload["decision_retrievals"][0]["semantic_event_cursor"] = len(
            payload["semantic"]["events"]
        )

        with self.assertRaisesRegex(ValueError, "event from after decision_t"):
            VerifiedDualTrackMemory.from_dict(payload)

    def test_restore_replays_exact_m3_budget_and_historical_ranking(self):
        system, _ = self.make_system_with_active_selection()
        original = system.to_dict()
        selected_index = next(
            index
            for index, episode in enumerate(original["episodic"]["episodes"])
            if episode["selected_rule_ids"]
        )
        budget_tamper = copy.deepcopy(original)
        budget_tamper["decision_retrievals"][selected_index]["rule_budget"] = 0

        with self.assertRaisesRegex(ValueError, "exact M3 eligibility/ranking"):
            VerifiedDualTrackMemory.from_dict(budget_tamper)

        state_tamper = copy.deepcopy(original)
        state_tamper["decision_retrievals"][selected_index]["retrieval_state"][
            "inflation"
        ] = 0.01

        with self.assertRaisesRegex(ValueError, "exact M3 eligibility/ranking"):
            VerifiedDualTrackMemory.from_dict(state_tamper)

    def test_restore_uses_retrieval_state_not_later_episode_pre_state_for_m3(self):
        system, rule = self.make_system_with_active_selection()
        retrieval_state = {
            "price": 104.0,
            "inflation": 0.04,
            "wealth": 1004.0,
        }
        selected = system.prepare_decision(
            decision_t=4,
            context_observation={"timestamp": 4, **retrieval_state},
            retrieval_state=retrieval_state,
        )
        self.assertEqual(selected.selected_rule_ids, (rule.rule_id,))
        system.begin_episode(
            decision_t=4,
            pre_state={**retrieval_state, "inflation": 0.01},
            proposed_action={},
            executed_action={},
        )

        restored = VerifiedDualTrackMemory.from_dict(system.to_dict())

        self.assertEqual(restored.to_dict(), system.to_dict())

    def test_restore_rejects_future_rule_forged_into_earlier_episode(self):
        system, rule = self.make_system_with_active_selection()
        payload = system.to_dict()
        earlier_episode = next(
            row for row in payload["episodic"]["episodes"] if row["decision_t"] == 0
        )
        earlier_episode["selected_rule_ids"] = [rule.rule_id]
        self.rehash_episode(earlier_episode)

        with self.assertRaisesRegex(ValueError, "same-time active-rule retrieval"):
            VerifiedDualTrackMemory.from_dict(payload)

if __name__ == "__main__":
    unittest.main()
