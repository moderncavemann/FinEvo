import copy
import unittest

from verified_memory.m2_episodic import EvidenceLinkedEpisodicTrack


def finalize(
    track: EvidenceLinkedEpisodicTrack,
    t: int,
    *,
    context=(1.0, 0.0),
    state=None,
    utility=1.0,
):
    state = state or {"price": 100.0, "wealth": 1000.0 + t}
    decision_id = track.begin_episode(
        decision_t=t,
        pre_state=state,
        context_id=f"c{t}",
        context_vector=context,
        retrieved_episode_ids=(),
        selected_rule_ids=(),
        proposed_action={"work": 0.5, "consumption": 0.4},
        executed_action={"labor_hours": 80.0, "consumption_fraction": 0.4},
    )
    return track.finalize_episode(
        decision_id,
        outcome_t=t + 1,
        next_state={**state, "wealth": float(state.get("wealth", 0.0)) + 10.0},
        outcome={"wealth_change": 10.0},
        reward=0.25,
        flow_utility=utility,
    )


class EpisodicTrackTest(unittest.TestCase):
    def make_track(self, capacity=3):
        return EvidenceLinkedEpisodicTrack(
            run_id="smoke", seed=13, agent_id=2, prompt_capacity=capacity
        )

    def test_pending_transition_is_not_retrievable(self):
        track = self.make_track()
        decision_id = track.begin_episode(
            decision_t=0,
            pre_state={"price": 100.0},
            context_id="c0",
            context_vector=(1.0,),
            retrieved_episode_ids=(),
            selected_rule_ids=(),
            proposed_action={"work": 0.5},
            executed_action={"labor_hours": 84.0},
        )
        self.assertEqual(track.pending_count, 1)
        self.assertEqual(
            track.retrieve(
                current_t=1,
                current_state={"price": 100.0},
                context_vector=(1.0,),
                use_context=True,
            ),
            (),
        )
        record = track.finalize_episode(
            decision_id,
            outcome_t=1,
            next_state={"price": 101.0},
            outcome={"wealth_change": 2.0},
            reward=0.1,
            flow_utility=0.5,
        )
        self.assertEqual(record.decision_t, 0)
        self.assertEqual(record.outcome_t, 1)
        self.assertEqual(track.finalized_count, 1)

    def test_finalized_future_outcome_is_not_retrievable(self):
        track = self.make_track()
        future = finalize(track, 5)

        self.assertEqual(
            track.retrieve(
                current_t=0,
                current_state={"price": 100.0},
                context_vector=None,
                use_context=False,
            ),
            (),
        )
        visible = track.retrieve(
            current_t=future.outcome_t,
            current_state={"price": 100.0},
            context_vector=None,
            use_context=False,
        )
        self.assertEqual(visible[0].episode.episode_id, future.episode_id)

    def test_temporal_alignment_and_duplicate_guard(self):
        track = self.make_track()
        decision_id = track.begin_episode(
            decision_t=3,
            pre_state={},
            context_id="c",
            context_vector=(),
            retrieved_episode_ids=(),
            selected_rule_ids=(),
            proposed_action={},
            executed_action={},
        )
        with self.assertRaises(ValueError):
            track.finalize_episode(
                decision_id,
                outcome_t=3,
                next_state={},
                outcome={},
                reward=0.0,
                flow_utility=0.0,
            )
        self.assertEqual(track.pending_count, 1)
        track.finalize_episode(
            decision_id,
            outcome_t=4,
            next_state={},
            outcome={},
            reward=0.0,
            flow_utility=0.0,
        )
        # Begin a fresh track for the duplicate-ID check.
        track = self.make_track()
        finalize(track, 3)
        with self.assertRaises(ValueError):
            track.begin_episode(
                decision_t=3,
                pre_state={},
                context_id="c",
                context_vector=(),
                retrieved_episode_ids=(),
                selected_rule_ids=(),
                proposed_action={},
                executed_action={},
            )

    def test_malformed_finalization_is_retry_safe(self):
        track = self.make_track()
        decision_id = track.begin_episode(
            decision_t=0,
            pre_state={"price": 100.0},
            context_id="c0",
            context_vector=(),
            retrieved_episode_ids=(),
            selected_rule_ids=(),
            proposed_action={},
            executed_action={},
        )

        with self.assertRaises(ValueError):
            track.finalize_episode(
                decision_id,
                outcome_t=1,
                next_state={"price": float("nan")},
                outcome={},
                reward=0.0,
                flow_utility=0.0,
            )
        self.assertEqual(track.pending_count, 1)
        record = track.finalize_episode(
            decision_id,
            outcome_t=1,
            next_state={"price": 101.0},
            outcome={},
            reward=0.0,
            flow_utility=0.0,
        )
        self.assertEqual(record.outcome_t, 1)

    def test_ledger_survives_prompt_eviction(self):
        track = self.make_track(capacity=2)
        for t in range(3):
            finalize(track, t)
        self.assertEqual(track.finalized_count, 3)
        hits = track.retrieve(
            current_t=4,
            current_state={"price": 100.0},
            context_vector=None,
            use_context=False,
            k=5,
        )
        self.assertEqual(len(hits), 2)
        self.assertIsNotNone(track.get("smoke:s13:a2:t0"))

    def test_context_route_changes_ranking_only_when_enabled(self):
        track = self.make_track()
        first = finalize(track, 0, context=(1.0, 0.0))
        second = finalize(track, 1, context=(0.0, 1.0))
        kwargs = dict(
            current_t=2,
            current_state={"price": 100.0, "wealth": 1002.0},
            context_vector=(1.0, 0.0),
            k=2,
            recency_weight=0.0,
            state_weight=0.0,
            importance_weight=0.0,
            context_weight=1.0,
        )
        without_context = track.retrieve(use_context=False, **kwargs)
        with_context = track.retrieve(use_context=True, **kwargs)
        self.assertEqual(without_context[0].episode.episode_id, second.episode_id)
        self.assertEqual(with_context[0].episode.episode_id, first.episode_id)
        self.assertTrue(all(hit.components["context_similarity"] == 0 for hit in without_context))

    def test_round_trip_and_input_immutability(self):
        track = self.make_track()
        source = {"price": 100.0, "nested": {"value": 1}}
        source_before = copy.deepcopy(source)
        finalize(track, 0, state=source)
        source["nested"]["value"] = 99
        self.assertEqual(track.finalized_episodes[0].pre_state["nested"]["value"], 1)
        self.assertNotEqual(source, source_before)

        restored = EvidenceLinkedEpisodicTrack.from_dict(track.to_dict())
        self.assertEqual(restored.to_dict(), track.to_dict())
        self.assertEqual(
            restored.finalized_episodes[0].record_hash,
            track.finalized_episodes[0].record_hash,
        )

    def test_restore_rejects_tampered_record_with_stale_hash(self):
        track = self.make_track()
        finalize(track, 0)
        payload = track.to_dict()
        payload["episodes"][0]["pre_state"]["price"] = 999.0

        with self.assertRaisesRegex(ValueError, "record hash mismatch"):
            EvidenceLinkedEpisodicTrack.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
