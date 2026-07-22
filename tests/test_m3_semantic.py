import json
import unittest

from verified_memory.m2_episodic import EvidenceLinkedEpisodicTrack
from verified_memory.m3_semantic import (
    ActionGuidance,
    CandidateParseError,
    ConditionPredicate,
    OutcomeCriterion,
    VerifiedSemanticRuleTrack,
)


def add_episode(
    track: EvidenceLinkedEpisodicTrack,
    t: int,
    *,
    inflation: float = 0.05,
    consumption: float = 0.20,
    labor_hours: float = 120.0,
    utility: float = 1.0,
    wealth_change: float = 10.0,
):
    pre_state = {
        "inflation": inflation,
        "price": 100.0 + t,
        "wealth": 1000.0 + t,
    }
    decision_id = track.begin_episode(
        decision_t=t,
        pre_state=pre_state,
        context_id=f"ctx-{t}",
        context_vector=(inflation, float(t)),
        retrieved_episode_ids=(),
        selected_rule_ids=(),
        proposed_action={
            "consumption_fraction": consumption,
            "labor_hours": labor_hours,
        },
        executed_action={
            "consumption_fraction": consumption,
            "labor_hours": labor_hours,
        },
    )
    return track.finalize_episode(
        decision_id,
        outcome_t=t + 1,
        next_state={**pre_state, "wealth": pre_state["wealth"] + wealth_change},
        outcome={"wealth_change": wealth_change},
        reward=utility,
        flow_utility=utility,
    )


def candidate_json(
    support_ids,
    *,
    condition_field="inflation",
    condition_operator=">",
    condition_value=0.03,
    action_target="consumption_fraction",
    action_direction="decrease",
    action_threshold=0.30,
    outcome_metric="flow_utility",
    outcome_operator=">",
    outcome_value=0.0,
    extra=None,
):
    value = {
        "condition": {
            "field": condition_field,
            "operator": condition_operator,
            "value": condition_value,
            "tolerance": 1e-9,
        },
        "action_guidance": {
            "target": action_target,
            "direction": action_direction,
            "threshold": action_threshold,
            "tolerance": 0.01,
        },
        "outcome_criterion": {
            "metric": outcome_metric,
            "operator": outcome_operator,
            "value": outcome_value,
            "tolerance": 1e-9,
        },
        "rationale": "The cited finalized transitions support this response.",
        "supporting_episode_ids": list(support_ids),
    }
    if extra:
        value.update(extra)
    return json.dumps(value)


class SemanticRuleTest(unittest.TestCase):
    def make_tracks(self, **semantic_kwargs):
        episodes = EvidenceLinkedEpisodicTrack(
            run_id="m3-smoke", seed=13, agent_id=2, prompt_capacity=20
        )
        semantic = VerifiedSemanticRuleTrack(episodes, **semantic_kwargs)
        return episodes, semantic

    def seed_valid_support(self, episodes):
        first = add_episode(episodes, 0)
        second = add_episode(episodes, 1)
        return first, second

    def activate_rule(self, episodes, semantic):
        first, second = self.seed_valid_support(episodes)
        provisional = semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=2
        )
        third = add_episode(episodes, 2)
        active = semantic.observe_episode(
            provisional.rule_id, third.episode_id, current_t=3
        )
        self.assertEqual(active.status, "active")
        return active

    def test_proposal_prompt_lists_schema_allowed_fields_and_episode_ids(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)

        prompt = semantic.build_proposal_prompt()

        self.assertIn("Allowed JSON schema", prompt)
        self.assertIn("supporting_episode_ids", prompt)
        self.assertIn("consumption_fraction", prompt)
        self.assertIn(first.episode_id, prompt)
        self.assertIn(second.episode_id, prompt)
        self.assertIn("Do not invent or duplicate episode IDs", prompt)

    def test_parser_accepts_fenced_json_but_rejects_malformed_and_extra_fields(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        raw = candidate_json([first.episode_id, second.episode_id])

        parsed = semantic.parse_candidate(f"```json\n{raw}\n```")
        self.assertEqual(parsed.supporting_episode_ids, (first.episode_id, second.episode_id))

        with self.assertRaises(CandidateParseError):
            semantic.propose("not-json", current_t=2)
        self.assertEqual(semantic.events[-1].event_type, "candidate_parse_rejected")

        with self.assertRaisesRegex(CandidateParseError, "extra"):
            semantic.parse_candidate(
                candidate_json(
                    [first.episode_id, second.episode_id], extra={"confidence": 1.0}
                )
            )

    def test_unsupported_fields_and_missing_episode_ids_are_rejected(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        with self.assertRaisesRegex(CandidateParseError, "unsupported condition"):
            semantic.parse_candidate(
                candidate_json(
                    [first.episode_id, second.episode_id],
                    condition_field="future_inflation",
                )
            )

        candidate = semantic.parse_candidate(
            candidate_json([first.episode_id, "invented:s0:a0:t999"])
        )
        rule = semantic.submit_candidate(candidate, current_t=2)
        self.assertEqual(rule.status, "rejected")
        self.assertTrue(
            any("does not exist" in reason for reason in rule.verification_reasons)
        )
        semantic.validate_referential_integrity()

    def test_candidate_and_observation_cannot_use_future_finalized_evidence(self):
        episodes, semantic = self.make_tracks()
        first = add_episode(episodes, 0)
        future = add_episode(episodes, 5)
        candidate = semantic.parse_candidate(
            candidate_json([first.episode_id, future.episode_id])
        )

        rejected = semantic.submit_candidate(candidate, current_t=2)
        self.assertEqual(rejected.status, "rejected")
        self.assertTrue(
            any("not observable" in reason for reason in rejected.verification_reasons)
        )
        with self.assertRaisesRegex(ValueError, "not finalized"):
            semantic.build_proposal_prompt(
                [future.episode_id], observed_through=2
            )

    def test_duplicate_claimed_support_is_rejected(self):
        episodes, semantic = self.make_tracks()
        first = add_episode(episodes, 0)
        candidate = semantic.parse_candidate(
            candidate_json([first.episode_id, first.episode_id])
        )

        rule = semantic.submit_candidate(candidate, current_t=1)

        self.assertEqual(rule.status, "rejected")
        self.assertEqual(rule.support_score, 1)
        self.assertTrue(
            any("duplicate" in reason for reason in rule.verification_reasons)
        )

    def test_condition_action_and_outcome_must_all_match_claimed_support(self):
        cases = [
            ({"inflation": 0.01, "consumption": 0.20, "utility": 1.0}, "condition"),
            ({"inflation": 0.05, "consumption": 0.80, "utility": 1.0}, "action"),
            ({"inflation": 0.05, "consumption": 0.20, "utility": -1.0}, "outcome"),
        ]
        for invalid, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                episodes, semantic = self.make_tracks()
                first = add_episode(episodes, 0)
                second = add_episode(episodes, 1, **invalid)
                rule = semantic.propose(
                    candidate_json([first.episode_id, second.episode_id]),
                    current_t=2,
                )
                self.assertEqual(rule.status, "rejected")
                self.assertTrue(
                    any(
                        expected_reason in reason
                        for reason in rule.verification_reasons
                    )
                )

    def test_verifier_searches_unlisted_condition_matched_counterexamples(self):
        episodes, semantic = self.make_tracks(proposal_confidence_floor=0.50)
        first, second = self.seed_valid_support(episodes)
        counterexample = add_episode(
            episodes, 2, consumption=0.80, utility=1.0
        )

        rule = semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=3
        )

        self.assertEqual(rule.status, "provisional")
        self.assertEqual(rule.support_score, 2)
        self.assertEqual(rule.contradiction_score, 1)
        self.assertEqual(rule.margin, 1)
        self.assertAlmostEqual(rule.confidence, 0.60)
        self.assertIn(counterexample.episode_id, rule.contradicting_episode_ids)

    def test_candidate_is_never_directly_active_and_distinct_evidence_activates(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        rule = semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=2
        )

        self.assertEqual(rule.status, "provisional")
        self.assertEqual(
            semantic.retrieve({"inflation": 0.05}, current_t=2), ()
        )

        third = add_episode(episodes, 2)
        active = semantic.observe_episode(rule.rule_id, third.episode_id, current_t=3)
        self.assertEqual(active.status, "active")
        self.assertEqual(active.post_proposal_evidence_count, 1)
        self.assertEqual(semantic.events[-1].event_type, "rule_activated")

    def test_preproposal_unlisted_evidence_cannot_satisfy_activation_delay(self):
        episodes, semantic = self.make_tracks()
        first = add_episode(episodes, 0)
        second = add_episode(episodes, 1)
        already_observed = add_episode(episodes, 2)
        rule = semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=3
        )

        with self.assertRaisesRegex(ValueError, "not post-proposal evidence"):
            semantic.observe_episode(
                rule.rule_id, already_observed.episode_id, current_t=3
            )
        unchanged = semantic.get(rule.rule_id)
        self.assertEqual(unchanged.status, "provisional")
        self.assertEqual(unchanged.post_proposal_evidence_count, 0)

    def test_duplicate_later_evidence_is_idempotent(self):
        episodes, semantic = self.make_tracks()
        active = self.activate_rule(episodes, semantic)
        before = active.to_dict()
        before_event_count = len(semantic.events)

        duplicate = semantic.observe_episode(
            active.rule_id, active.supporting_episode_ids[-1], current_t=4
        )

        self.assertEqual(duplicate.to_dict(), before)
        self.assertEqual(len(semantic.events), before_event_count + 1)
        self.assertEqual(semantic.events[-1].event_type, "duplicate_evidence_ignored")

    def test_consecutive_failures_retire_active_rule(self):
        episodes, semantic = self.make_tracks(
            retirement_patience=2, retirement_confidence_threshold=0.0
        )
        active = self.activate_rule(episodes, semantic)
        failure_one = add_episode(episodes, 3, consumption=0.20, utility=-1.0)
        after_one = semantic.observe_episode(
            active.rule_id, failure_one.episode_id, current_t=4
        )
        self.assertEqual(after_one.status, "active")
        self.assertEqual(after_one.consecutive_failures, 1)

        failure_two = add_episode(episodes, 4, consumption=0.80, utility=1.0)
        retired = semantic.observe_episode(
            active.rule_id, failure_two.episode_id, current_t=5
        )
        self.assertEqual(retired.status, "retired")
        self.assertEqual(retired.consecutive_failures, 2)
        self.assertEqual(semantic.events[-1].event_type, "rule_retired")
        self.assertEqual(
            semantic.retrieve({"inflation": 0.05}, current_t=5), ()
        )

    def test_confidence_floor_can_retire_before_patience(self):
        episodes, semantic = self.make_tracks(
            retirement_patience=5, retirement_confidence_threshold=0.70
        )
        active = self.activate_rule(episodes, semantic)
        failure = add_episode(episodes, 3, utility=-1.0)

        retired = semantic.observe_episode(
            active.rule_id, failure.episode_id, current_t=4
        )

        self.assertEqual(retired.status, "retired")
        self.assertLess(retired.confidence, 0.70)

    def test_experimental_bad_rule_injection_has_provenance_and_retires(self):
        episodes, semantic = self.make_tracks(
            retirement_patience=2, retirement_confidence_threshold=0.0
        )
        injected = semantic.inject_active_rule(
            condition=ConditionPredicate("inflation", ">", 0.03),
            action_guidance=ActionGuidance(
                "consumption_fraction", "increase", 0.90, 0.01
            ),
            outcome_criterion=OutcomeCriterion("flow_utility", ">", 0.0),
            rationale="Deliberately wrong rule for error-injection evaluation.",
            current_t=0,
            injection_id="bad-consumption-v1",
            provenance={"experiment_id": "M3-INJECT", "treatment": "false-rule"},
        )
        self.assertEqual(injected.status, "active")
        self.assertTrue(injected.injected)
        self.assertEqual(
            semantic.events[-1].event_type, "experimental_rule_injected_active"
        )

        first_failure = add_episode(episodes, 0, consumption=0.20)
        still_active = semantic.observe_episode(
            injected.rule_id, first_failure.episode_id, current_t=1
        )
        self.assertEqual(still_active.status, "active")
        second_failure = add_episode(episodes, 1, consumption=0.20)
        retired = semantic.observe_episode(
            injected.rule_id, second_failure.episode_id, current_t=2
        )
        self.assertEqual(retired.status, "retired")
        semantic.validate_referential_integrity()

    def test_retrieval_exposes_only_active_relevant_rules(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        provisional = semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=2
        )
        injected = semantic.inject_active_rule(
            condition=ConditionPredicate("inflation", ">", 0.03),
            action_guidance=ActionGuidance(
                "consumption_fraction", "maintain", 0.20, 0.01
            ),
            outcome_criterion=OutcomeCriterion("flow_utility", ">", 0.0),
            rationale="Controlled active rule.",
            current_t=2,
            injection_id="retrieval-control",
            provenance={"experiment_id": "M3-RETRIEVE"},
        )

        selected = semantic.retrieve({"inflation": 0.05}, current_t=2)
        self.assertEqual([rule.rule_id for rule in selected], [injected.rule_id])
        self.assertNotIn(provisional.rule_id, [rule.rule_id for rule in selected])
        self.assertEqual(
            semantic.retrieve({"inflation": 0.01}, current_t=2), ()
        )
        self.assertEqual(semantic.events[-1].event_type, "active_rule_retrieved")

    def test_full_round_trip_and_referential_integrity(self):
        episodes, semantic = self.make_tracks()
        active = self.activate_rule(episodes, semantic)
        selected = semantic.retrieve({"inflation": 0.05}, current_t=4)
        self.assertEqual(selected[0].rule_id, active.rule_id)
        semantic.validate_referential_integrity()

        serialized = semantic.to_dict()
        restored = VerifiedSemanticRuleTrack.from_dict(
            serialized, episodic_track=episodes
        )

        self.assertEqual(restored.to_dict(), serialized)
        restored.validate_referential_integrity()
        self.assertEqual(
            restored.retrieve(
                {"inflation": 0.05}, current_t=5, log_selection=False
            )[0].rule_id,
            active.rule_id,
        )

    def test_restore_rejects_falsified_postproposal_timing(self):
        episodes, semantic = self.make_tracks()
        active = self.activate_rule(episodes, semantic)
        serialized = semantic.to_dict()
        serialized_rule = next(
            item for item in serialized["rules"] if item["rule_id"] == active.rule_id
        )
        # The activation evidence has outcome_t=3. Moving creation to t=3 makes
        # it contemporaneous, not later, while leaving the claimed count at one.
        serialized_rule["created_at"] = 3

        with self.assertRaisesRegex(ValueError, "post-proposal evidence count"):
            VerifiedSemanticRuleTrack.from_dict(
                serialized, episodic_track=episodes
            )


if __name__ == "__main__":
    unittest.main()
