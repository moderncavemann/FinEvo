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

    def test_constructor_and_restore_share_identity_contract(self):
        for invalid_run_id in ("", "   ", None, 7):
            with self.subTest(run_id=invalid_run_id):
                with self.assertRaisesRegex(ValueError, "run_id"):
                    EvidenceLinkedEpisodicTrack(
                        run_id=invalid_run_id, seed=13, agent_id=2
                    )
        with self.assertRaisesRegex(ValueError, "agent_id"):
            EvidenceLinkedEpisodicTrack(run_id="valid", seed=13, agent_id=-1)

        track = self.make_track()
        finalize(track, 0)
        self.assertEqual(
            EvidenceLinkedEpisodicTrack.from_dict(track.to_dict()).to_dict(),
            track.to_dict(),
        )

    def test_restore_requires_canonical_prompt_buffer_suffix(self):
        track = self.make_track(capacity=2)
        records = [finalize(track, t) for t in range(3)]
        canonical = track.to_dict()
        self.assertEqual(
            canonical["prompt_ids"],
            [records[1].episode_id, records[2].episode_id],
        )

        for prompt_ids in (
            [],
            [records[0].episode_id],
            [records[2].episode_id, records[0].episode_id],
            [records[2].episode_id, records[1].episode_id],
        ):
            with self.subTest(prompt_ids=prompt_ids):
                payload = copy.deepcopy(canonical)
                payload["prompt_ids"] = prompt_ids
                with self.assertRaisesRegex(ValueError, "canonical.*suffix"):
                    EvidenceLinkedEpisodicTrack.from_dict(payload)

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

    def test_out_of_order_backfill_cannot_change_utility_advantage_history(self):
        track = self.make_track()
        future = finalize(track, 5, utility=100.0)

        self.assertEqual(future.utility_advantage, 100.0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            finalize(track, 0, utility=1.0)
        self.assertEqual(track.finalized_count, 1)

        chronological = self.make_track()
        earlier = finalize(chronological, 0, utility=100.0)
        later = finalize(chronological, 5, utility=10.0)
        self.assertEqual(later.utility_advantage, -90.0)
        payload = chronological.to_dict()
        payload["episodes"] = list(reversed(payload["episodes"]))
        with self.assertRaisesRegex(
            ValueError, "strictly increasing|causal ledger prefix"
        ):
            EvidenceLinkedEpisodicTrack.from_dict(payload)
        self.assertEqual(earlier.utility_advantage, 100.0)

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

    def test_prompt_capacity_bounds_results_not_long_horizon_candidates(self):
        track = self.make_track(capacity=2)
        early = finalize(track, 0, utility=100.0)
        for t in range(1, 120):
            finalize(track, t, utility=100.0)

        self.assertEqual(track.finalized_count, 120)
        self.assertNotIn(early.episode_id, track.to_dict()["prompt_ids"])
        restored = EvidenceLinkedEpisodicTrack.from_dict(track.to_dict())

        for current_t in (60, 120):
            for candidate_track in (track, restored):
                hits = candidate_track.retrieve(
                    current_t=current_t,
                    current_state={"price": 100.0},
                    context_vector=None,
                    use_context=False,
                    k=5,
                    recency_weight=0.0,
                    state_weight=0.0,
                    importance_weight=1.0,
                    context_weight=0.0,
                )
                self.assertEqual(len(hits), candidate_track.prompt_capacity)
                self.assertEqual(hits[0].episode.episode_id, early.episode_id)
                self.assertTrue(
                    all(hit.episode.outcome_t <= current_t for hit in hits),
                    "retrieval must exclude finalized evidence from the caller's future",
                )
                self.assertTrue(
                    all(
                        hit.episode.run_id == candidate_track.run_id
                        and hit.episode.seed == candidate_track.seed
                        and hit.episode.agent_id == candidate_track.agent_id
                        for hit in hits
                    ),
                    "retrieval must preserve track identity",
                )

        decision_id = track.begin_episode(
            decision_t=120,
            pre_state={"price": 100.0},
            context_id="pending",
            context_vector=(),
            retrieved_episode_ids=(),
            selected_rule_ids=(),
            proposed_action={},
            executed_action={},
        )
        pending_episode_id = track.make_episode_id(
            track.run_id, track.seed, track.agent_id, 120
        )
        hits = track.retrieve(
            current_t=121,
            current_state={"price": 100.0},
            context_vector=None,
            use_context=False,
            k=5,
            recency_weight=0.0,
            state_weight=0.0,
            importance_weight=1.0,
            context_weight=0.0,
        )
        self.assertNotIn(pending_episode_id, {hit.episode.episode_id for hit in hits})
        self.assertEqual(track.pending_episodes[0].decision_id, decision_id)

    def test_public_records_are_deep_copies_of_internal_ledger_state(self):
        track = self.make_track()
        returned = finalize(
            track,
            0,
            state={"price": 100.0, "wealth": 1000.0},
        )
        returned.pre_state["price"] = 999.0
        returned.executed_action["labor_hours"] = 0.0
        exposed = track.finalized_episodes[0]
        exposed.pre_state["price"] = 888.0
        hit = track.retrieve(
            current_t=1,
            current_state={"price": 100.0},
            context_vector=None,
            use_context=False,
        )[0]
        hit.episode.pre_state["price"] = 777.0

        internal_view = track.get(returned.episode_id)
        self.assertEqual(internal_view.pre_state["price"], 100.0)
        self.assertEqual(internal_view.executed_action["labor_hours"], 80.0)
        track.validate_references()

    def test_public_pending_record_cannot_mutate_finalized_action(self):
        track = self.make_track()
        decision_id = track.begin_episode(
            decision_t=0,
            pre_state={"price": 100.0},
            context_id="c0",
            context_vector=(),
            retrieved_episode_ids=(),
            selected_rule_ids=(),
            proposed_action={"work": 0.5},
            executed_action={"labor_hours": 84.0},
        )
        track.pending_episodes[0].executed_action["labor_hours"] = 0.0
        record = track.finalize_episode(
            decision_id,
            outcome_t=1,
            next_state={"price": 101.0},
            outcome={},
            reward=0.0,
            flow_utility=0.0,
        )
        self.assertEqual(record.executed_action["labor_hours"], 84.0)

    def test_current_snapshot_requires_exact_state_arrays(self):
        track = self.make_track()
        track.begin_episode(
            decision_t=0,
            pre_state={},
            context_id="",
            context_vector=(),
            retrieved_episode_ids=(),
            selected_rule_ids=(),
            proposed_action={},
            executed_action={},
        )
        for missing in ("episodes", "prompt_ids", "pending"):
            payload = track.to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, "keys are not exact"):
                    EvidenceLinkedEpisodicTrack.from_dict(payload)

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

    def test_pending_restore_rejects_invalid_structural_fields(self):
        track = self.make_track()
        track.begin_episode(
            decision_t=0,
            pre_state={"price": 100.0},
            context_id="ctx-0",
            context_vector=(1.0, 0.0),
            retrieved_episode_ids=(),
            selected_rule_ids=(),
            proposed_action={"work": 0.5},
            executed_action={"labor_hours": 80.0},
            rng_draw=0.5,
        )
        canonical = track.to_dict()

        cases = (
            ("negative decision", "decision_t", -1, "decision_t"),
            ("non-finite context", "context_vector", [float("nan")], "context_vector"),
            ("out-of-range rng", "rng_draw", 1.01, "rng_draw"),
            ("non-finite rng", "rng_draw", float("nan"), "rng_draw"),
            ("non-string context ID", "context_id", 7, "context_id"),
            ("non-string rule ID", "selected_rule_ids", [None], "selected_rule_ids"),
        )
        for label, field, value, message in cases:
            with self.subTest(label=label):
                payload = copy.deepcopy(canonical)
                payload["pending"][0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    EvidenceLinkedEpisodicTrack.from_dict(payload)

        for label, field, nested_field, value in (
            ("non-finite state", "pre_state", "price", float("inf")),
            ("non-finite proposed action", "proposed_action", "work", float("nan")),
            ("non-JSON executed action", "executed_action", "bad", object()),
        ):
            with self.subTest(label=label):
                payload = copy.deepcopy(canonical)
                payload["pending"][0][field][nested_field] = value
                with self.assertRaisesRegex(ValueError, field):
                    EvidenceLinkedEpisodicTrack.from_dict(payload)

    def test_pending_restore_preserves_empty_fields_and_rng_boundaries(self):
        for rng_draw in (None, 0.0, 1.0):
            with self.subTest(rng_draw=rng_draw):
                track = self.make_track()
                track.begin_episode(
                    decision_t=0,
                    pre_state={},
                    context_id="",
                    context_vector=(),
                    retrieved_episode_ids=(),
                    selected_rule_ids=(),
                    proposed_action={},
                    executed_action={},
                    rng_draw=rng_draw,
                    reflection="",
                )
                restored = EvidenceLinkedEpisodicTrack.from_dict(track.to_dict())
                self.assertEqual(restored.to_dict(), track.to_dict())

    def test_episode_restore_rejects_invalid_time_and_rng_before_hash(self):
        track = self.make_track()
        finalize(track, 0)
        canonical = track.to_dict()

        negative_time = copy.deepcopy(canonical)
        negative_time["episodes"][0]["decision_t"] = -1
        negative_time["episodes"][0]["outcome_t"] = 0
        with self.assertRaisesRegex(ValueError, "decision_t"):
            EvidenceLinkedEpisodicTrack.from_dict(negative_time)

        invalid_rng = copy.deepcopy(canonical)
        invalid_rng["episodes"][0]["rng_draw"] = 1.5
        with self.assertRaisesRegex(ValueError, "rng_draw"):
            EvidenceLinkedEpisodicTrack.from_dict(invalid_rng)

    def test_finalize_revalidates_pending_structure_without_consuming_it(self):
        for field, value, outcome_t, message in (
            ("decision_t", -1, 0, "decision_t"),
            ("rng_draw", 1.5, 1, "rng_draw"),
        ):
            with self.subTest(field=field):
                track = self.make_track()
                decision_id = track.begin_episode(
                    decision_t=0,
                    pre_state={},
                    context_id="",
                    context_vector=(),
                    retrieved_episode_ids=(),
                    selected_rule_ids=(),
                    proposed_action={},
                    executed_action={},
                )
                # Frozen records defend normal callers; this deliberately
                # emulates private-state corruption to exercise the seal gate.
                object.__setattr__(next(iter(track._pending.values())), field, value)
                with self.assertRaisesRegex(ValueError, message):
                    track.finalize_episode(
                        decision_id,
                        outcome_t=outcome_t,
                        next_state={},
                        outcome={},
                        reward=0.0,
                        flow_utility=0.0,
                    )
                self.assertEqual(track.pending_count, 1)
                self.assertEqual(track.finalized_count, 0)

    def test_restore_rejects_tampered_record_with_stale_hash(self):
        track = self.make_track()
        finalize(track, 0)
        payload = track.to_dict()
        payload["episodes"][0]["pre_state"]["price"] = 999.0

        with self.assertRaisesRegex(ValueError, "record hash mismatch"):
            EvidenceLinkedEpisodicTrack.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
