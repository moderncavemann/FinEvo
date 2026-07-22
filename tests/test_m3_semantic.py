import copy
import hashlib
import json
import unittest

from verified_memory.m2_episodic import EvidenceLinkedEpisodicTrack
from verified_memory.m3_semantic import (
    ActionGuidance,
    CandidateParseError,
    ConditionPredicate,
    ContextScope,
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
    utility: float | None = None,
    wealth_change: float = 10.0,
):
    utility = float(1.0 + t if utility is None else utility)
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
    action_direction="at_most",
    action_threshold=0.30,
    include_context_scope=True,
    legacy_outcome_criterion=None,
    extra=None,
):
    value = {
        **(
            {"context_scope": {"scope_id": "global", "predicates": []}}
            if include_context_scope
            else {}
        ),
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
        "rationale": "The cited finalized transitions support this response.",
        "supporting_episode_ids": list(support_ids),
    }
    if legacy_outcome_criterion is not None:
        value["outcome_criterion"] = legacy_outcome_criterion
    if extra:
        value.update(extra)
    return json.dumps(value)


def rehash_rule_event(payload, event_index):
    event = payload["events"][event_index]
    core = {
        key: value
        for key, value in event.items()
        if key not in {"schema_version", "event_id"}
    }
    core["index"] = event_index
    digest = hashlib.sha256(
        json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    event["event_id"] = f"rle-{event_index:06d}-{digest[:12]}"


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
        self.assertIn(
            "Every claimed support episode must independently satisfy", prompt
        )
        self.assertIn("at_least means executed value >= threshold", prompt)
        self.assertIn("Do not output or choose an outcome criterion", prompt)
        self.assertIn('"metric": "utility_advantage"', prompt)
        self.assertIn("consumption_fraction", prompt)
        self.assertIn(first.episode_id, prompt)
        self.assertIn(second.episode_id, prompt)
        self.assertIn("Do not invent or duplicate episode IDs", prompt)

    def test_parser_accepts_fenced_json_but_rejects_malformed_and_extra_fields(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        raw = candidate_json([first.episode_id, second.episode_id])

        parsed = semantic.parse_candidate(f"```json\n{raw}\n```")
        self.assertEqual(
            parsed.supporting_episode_ids, (first.episode_id, second.episode_id)
        )

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
            semantic.build_proposal_prompt([future.episode_id], observed_through=2)

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
        counterexample = add_episode(episodes, 2, consumption=0.80, utility=4.0)

        rule = semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=3
        )

        self.assertEqual(rule.status, "provisional")
        self.assertEqual(rule.support_score, 2)
        self.assertEqual(rule.contradiction_score, 0.5)
        self.assertEqual(rule.margin, 1.5)
        self.assertAlmostEqual(rule.confidence, 2.0 / 3.0)
        self.assertIn(counterexample.episode_id, rule.contradicting_episode_ids)
        self.assertIn(counterexample.episode_id, rule.alternative_success_episode_ids)

    def test_candidate_is_never_directly_active_and_distinct_evidence_activates(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        rule = semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=2
        )

        self.assertEqual(rule.status, "provisional")
        self.assertEqual(semantic.retrieve({"inflation": 0.05}, current_t=2), ())

        third = add_episode(episodes, 2)
        active = semantic.observe_episode(rule.rule_id, third.episode_id, current_t=3)
        self.assertEqual(active.status, "active")
        self.assertEqual(active.post_proposal_evidence_count, 1)
        self.assertEqual(active.post_proposal_support_count, 1)
        self.assertEqual(active.activation_episode_id, third.episode_id)
        self.assertEqual(semantic.events[-1].event_type, "rule_activated")

    def test_first_postproposal_contradiction_can_never_activate_rule(self):
        episodes, semantic = self.make_tracks(
            activation_min_support=3,
            retirement_patience=5,
            retirement_confidence_threshold=0.0,
        )
        historical = [
            add_episode(episodes, 0, utility=1.0),
            add_episode(episodes, 1, utility=3.0),
            add_episode(episodes, 2, utility=5.0),
        ]
        provisional = semantic.propose(
            candidate_json([episode.episode_id for episode in historical]),
            current_t=3,
        )
        later_harm = add_episode(episodes, 3, consumption=0.20, utility=-10.0)

        unchanged = semantic.observe_episode(
            provisional.rule_id, later_harm.episode_id, current_t=4
        )

        self.assertEqual(unchanged.status, "provisional")
        self.assertEqual(unchanged.post_proposal_support_count, 0)
        self.assertEqual(unchanged.post_proposal_contradiction_count, 1)
        self.assertEqual(unchanged.activation_episode_id, None)
        self.assertEqual(
            semantic.events[-1].event_type,
            "harmful_compliance_evidence_added",
        )

    def test_five_way_evidence_taxonomy_has_auditable_weights(self):
        episodes, semantic = self.make_tracks(proposal_confidence_floor=0.50)
        support_one = add_episode(episodes, 0, utility=1.0)
        support_two = add_episode(episodes, 1, utility=3.0)
        harmful = add_episode(episodes, 2, consumption=0.20, utility=-10.0)
        alternative_success = add_episode(episodes, 3, consumption=0.80, utility=10.0)
        alternative_failure = add_episode(episodes, 4, consumption=0.80, utility=-20.0)
        irrelevant = add_episode(
            episodes, 5, inflation=0.01, consumption=0.20, utility=30.0
        )

        rule = semantic.propose(
            candidate_json([support_one.episode_id, support_two.episode_id]),
            current_t=6,
        )

        self.assertEqual(rule.status, "provisional")
        self.assertEqual(
            rule.supporting_episode_ids,
            (support_one.episode_id, support_two.episode_id),
        )
        self.assertEqual(rule.harmful_compliance_episode_ids, (harmful.episode_id,))
        self.assertEqual(
            rule.alternative_success_episode_ids,
            (alternative_success.episode_id,),
        )
        self.assertEqual(
            rule.alternative_failure_episode_ids,
            (alternative_failure.episode_id,),
        )
        self.assertEqual(rule.irrelevant_episode_ids, (irrelevant.episode_id,))
        self.assertIn(irrelevant.episode_id, semantic._all_evidence_ids(rule))
        self.assertEqual(rule.support_score, 2.0)
        self.assertEqual(rule.contradiction_score, 1.5)
        self.assertEqual(rule.margin, 0.5)
        metrics = semantic.events[-1].metrics
        self.assertEqual(metrics["evidence_weights"]["harmful_compliance"], 1.0)
        self.assertEqual(metrics["evidence_weights"]["alternative_success"], 0.5)
        self.assertEqual(metrics["evidence_weights"]["alternative_failure"], 0.0)
        self.assertEqual(metrics["evidence_weights"]["irrelevant"], 0.0)
        self.assertEqual(metrics["evidence_type_counts"]["irrelevant"], 1)

    def test_postproposal_irrelevant_observation_is_counted_without_weight(self):
        episodes, semantic = self.make_tracks()
        active = self.activate_rule(episodes, semantic)
        irrelevant = add_episode(episodes, 3, inflation=0.01, utility=100.0)

        updated = semantic.observe_episode(
            active.rule_id, irrelevant.episode_id, current_t=4
        )

        self.assertEqual(updated.status, "active")
        self.assertEqual(updated.irrelevant_episode_ids, (irrelevant.episode_id,))
        self.assertEqual(updated.post_proposal_irrelevant_count, 1)
        self.assertEqual(updated.post_proposal_evidence_count, 2)
        self.assertEqual(updated.support_score, active.support_score)
        self.assertEqual(updated.confidence, active.confidence)
        self.assertEqual(
            semantic.events[-1].event_type, "irrelevant_evidence_added"
        )
        self.assertEqual(
            semantic.events[-1].metrics["evidence_type_counts"]["irrelevant"],
            1,
        )
        semantic.validate_referential_integrity()

    def test_restore_replays_exhaustive_creation_irrelevant_search(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        irrelevant = add_episode(episodes, 2, inflation=0.01)
        semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=3
        )
        serialized = semantic.to_dict()
        serialized["rules"][0]["irrelevant_episode_ids"] = []
        serialized["events"][0]["metrics"]["evidence_type_counts"][
            "irrelevant"
        ] = 0
        rehash_rule_event(serialized, 0)

        with self.assertRaisesRegex(ValueError, "creation evidence search"):
            VerifiedSemanticRuleTrack.from_dict(
                serialized, episodic_track=episodes
            )
        self.assertEqual(irrelevant.outcome_t, 3)

    def test_duplicate_scope_predicates_are_rejected_before_family_assignment(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        predicate = {
            "field": "wealth",
            "operator": ">=",
            "value": 0.0,
            "tolerance": 0.0,
        }
        raw = candidate_json(
            [first.episode_id, second.episode_id],
            extra={
                "context_scope": {
                    "scope_id": "duplicated-shape",
                    "predicates": [predicate, predicate],
                }
            },
        )

        with self.assertRaisesRegex(CandidateParseError, "duplicates"):
            semantic.parse_candidate(raw)

    def test_restore_recomputes_historical_event_metrics(self):
        episodes, semantic = self.make_tracks()
        active = self.activate_rule(episodes, semantic)
        semantic.retrieve({"inflation": 0.05}, current_t=4)
        serialized = semantic.to_dict()
        event_index = next(
            index
            for index, event in enumerate(serialized["events"])
            if event["event_type"] == "active_rule_retrieved"
        )
        serialized["events"][event_index]["metrics"]["confidence"] = 0.123
        rehash_rule_event(serialized, event_index)

        with self.assertRaisesRegex(ValueError, "event metrics do not reproduce"):
            VerifiedSemanticRuleTrack.from_dict(
                serialized, episodic_track=episodes
            )

    def test_legacy_direction_and_criterion_migration_is_explicit(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        registered = semantic.registered_outcome_criterion.to_dict()
        raw = candidate_json(
            [first.episode_id, second.episode_id],
            action_direction="decrease",
            include_context_scope=False,
            legacy_outcome_criterion=registered,
        )

        parsed = semantic.parse_candidate(raw)

        self.assertEqual(parsed.action_guidance.direction, "at_most")
        self.assertEqual(len(parsed.migration_notes), 3)
        rule = semantic.submit_candidate(parsed, current_t=2)
        self.assertEqual(
            semantic.events[-1].provenance["migration_notes"],
            list(parsed.migration_notes),
        )
        with self.assertRaisesRegex(ValueError, "legacy action direction"):
            ActionGuidance.from_dict(
                {
                    "target": "labor_hours",
                    "direction": "increase",
                    "threshold": 80.0,
                    "tolerance": 0.0,
                }
            )
        self.assertEqual(rule.action_guidance.direction, "at_most")

    def test_candidate_cannot_choose_outcome_threshold_and_config_is_sealed(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        serialized = semantic.to_dict()
        self.assertEqual(
            serialized["config"]["registered_outcome_criterion"],
            {
                "metric": "utility_advantage",
                "operator": ">",
                "value": 0.0,
                "tolerance": 0.0,
            },
        )
        with self.assertRaisesRegex(CandidateParseError, "not the verifier-registered"):
            semantic.parse_candidate(
                candidate_json(
                    [first.episode_id, second.episode_id],
                    legacy_outcome_criterion={
                        "metric": "utility_advantage",
                        "operator": ">=",
                        "value": -1000.0,
                        "tolerance": 0.0,
                    },
                )
            )

        custom = VerifiedSemanticRuleTrack(
            episodes,
            registered_outcome_criterion={
                "metric": "flow_utility",
                "operator": ">=",
                "value": 1.0,
                "tolerance": 0.0,
            },
        )
        restored = VerifiedSemanticRuleTrack.from_dict(
            custom.to_dict(), episodic_track=episodes
        )
        self.assertEqual(
            restored.registered_outcome_criterion.to_dict(),
            custom.registered_outcome_criterion.to_dict(),
        )
        default_candidate = semantic.parse_candidate(
            candidate_json([first.episode_id, second.episode_id])
        )
        with self.assertRaisesRegex(ValueError, "verifier-registered criterion"):
            custom.submit_candidate(default_candidate, current_t=2)
        self.assertEqual(custom.to_dict()["candidates"], [])
        custom.validate_referential_integrity()

    def test_candidate_provenance_is_append_only_across_equivalent_raw_responses(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        exact_raw = candidate_json([first.episode_id, second.episode_id])
        fenced_raw = f"```json\n{exact_raw}\n```"
        exact = semantic.parse_candidate(exact_raw)
        fenced = semantic.parse_candidate(fenced_raw)

        self.assertNotEqual(exact.raw_response_hash, fenced.raw_response_hash)
        self.assertNotEqual(exact.candidate_id, fenced.candidate_id)
        rule = semantic.submit_candidate(exact, current_t=2)
        duplicate = semantic.submit_candidate(fenced, current_t=2)

        self.assertEqual(duplicate.rule_id, rule.rule_id)
        serialized_candidates = {
            row["candidate_id"]: row["raw_response_hash"]
            for row in semantic.to_dict()["candidates"]
        }
        self.assertEqual(
            serialized_candidates,
            {
                exact.candidate_id: exact.raw_response_hash,
                fenced.candidate_id: fenced.raw_response_hash,
            },
        )
        self.assertEqual(rule.candidate_ids, (exact.candidate_id,))
        semantic.validate_referential_integrity()

        tampered = semantic.to_dict()
        tampered["rules"][0]["candidate_ids"] = [fenced.candidate_id]
        with self.assertRaisesRegex(ValueError, "creation candidate/raw provenance"):
            VerifiedSemanticRuleTrack.from_dict(tampered, episodic_track=episodes)

    def test_restore_requires_exact_current_nested_schema_and_config(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=2
        )
        serialized = semantic.to_dict()

        cases = (
            ("rule schema", ("rules", 0, "schema_version")),
            ("event schema", ("events", 0, "schema_version")),
            ("verifier config", ("config", "evidence_weights")),
        )
        for label, path in cases:
            with self.subTest(label=label):
                payload = copy.deepcopy(serialized)
                if len(path) == 3:
                    del payload[path[0]][path[1]][path[2]]
                else:
                    del payload[path[0]][path[1]]
                with self.assertRaisesRegex(ValueError, "missing"):
                    VerifiedSemanticRuleTrack.from_dict(
                        payload, episodic_track=episodes
                    )

    def test_rejected_family_can_form_new_version_only_with_new_support(self):
        episodes, semantic = self.make_tracks(proposal_confidence_floor=0.50)
        first = add_episode(episodes, 0, utility=1.0)
        harmful = add_episode(episodes, 1, consumption=0.20, utility=-5.0)
        rejected = semantic.propose(
            candidate_json([first.episode_id, harmful.episode_id]), current_t=2
        )
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(rejected.rule_version, 1)

        no_new_support = semantic.propose(
            candidate_json([first.episode_id, harmful.episode_id]), current_t=2
        )
        self.assertEqual(no_new_support.rule_id, rejected.rule_id)
        self.assertEqual(len(semantic.rules), 1)

        new_support = add_episode(episodes, 2, utility=10.0)
        revised = semantic.propose(
            candidate_json([first.episode_id, new_support.episode_id]), current_t=3
        )

        self.assertEqual(revised.status, "provisional")
        self.assertEqual(revised.rule_family_id, rejected.rule_family_id)
        self.assertEqual(revised.rule_version, 2)
        self.assertEqual(revised.supersedes_rule_id, rejected.rule_id)
        self.assertEqual(revised.derived_from_rule_ids, (rejected.rule_id,))
        self.assertNotEqual(revised.rule_id, rejected.rule_id)
        semantic.validate_referential_integrity()

    def test_uncounted_preterminal_support_cannot_create_new_family_version(self):
        episodes, semantic = self.make_tracks(proposal_confidence_floor=0.50)
        first = add_episode(episodes, 0, utility=1.0)
        harmful = add_episode(episodes, 1, consumption=0.20, utility=-5.0)
        uncounted_old_support = add_episode(episodes, 2, utility=10.0)

        rejected = semantic.propose(
            candidate_json([first.episode_id, harmful.episode_id]), current_t=3
        )
        self.assertEqual(rejected.status, "rejected")
        self.assertNotIn(
            uncounted_old_support.episode_id,
            semantic._all_evidence_ids(rejected),
        )
        self.assertEqual(rejected.updated_at, uncounted_old_support.outcome_t)

        old_support_only = semantic.propose(
            candidate_json([first.episode_id, uncounted_old_support.episode_id]),
            current_t=3,
        )

        self.assertEqual(old_support_only.rule_id, rejected.rule_id)
        self.assertEqual(len(semantic.rules), 1)
        self.assertEqual(
            semantic.events[-1].event_type,
            "terminal_family_candidate_without_new_support_ignored",
        )
        self.assertEqual(
            semantic.events[-1].provenance["latest_terminal_updated_at"],
            rejected.updated_at,
        )

        later_support = add_episode(episodes, 3, utility=11.0)
        revised = semantic.propose(
            candidate_json([first.episode_id, later_support.episode_id]),
            current_t=4,
        )

        self.assertEqual(revised.status, "provisional")
        self.assertEqual(revised.rule_version, 2)
        self.assertEqual(revised.supersedes_rule_id, rejected.rule_id)
        self.assertGreater(later_support.outcome_t, rejected.updated_at)
        semantic.validate_referential_integrity()

    def test_restore_rechecks_terminal_parent_time_and_new_creation_support(self):
        episodes, semantic = self.make_tracks(proposal_confidence_floor=0.50)
        first = add_episode(episodes, 0, utility=1.0)
        harmful = add_episode(episodes, 1, consumption=0.20, utility=-5.0)
        rejected = semantic.propose(
            candidate_json([first.episode_id, harmful.episode_id]), current_t=2
        )
        later_support = add_episode(episodes, 2, utility=10.0)
        revised = semantic.propose(
            candidate_json([first.episode_id, later_support.episode_id]),
            current_t=3,
        )
        self.assertEqual(revised.rule_version, 2)

        serialized = semantic.to_dict()
        parent = next(
            row for row in serialized["rules"] if row["rule_id"] == rejected.rule_id
        )
        parent["updated_at"] = revised.created_at
        with self.assertRaisesRegex(ValueError, "terminal parent update|lifecycle"):
            VerifiedSemanticRuleTrack.from_dict(serialized, episodic_track=episodes)

    def test_opposite_predicate_and_guidance_shapes_are_distinct_families(self):
        episodes, semantic = self.make_tracks()
        high_one = add_episode(
            episodes, 0, inflation=0.06, consumption=0.20, utility=1.0
        )
        high_two = add_episode(
            episodes, 1, inflation=0.07, consumption=0.20, utility=3.0
        )
        high_rule = semantic.propose(
            candidate_json(
                [high_one.episode_id, high_two.episode_id],
                condition_operator=">",
                condition_value=0.05,
                action_direction="at_most",
                action_threshold=0.30,
            ),
            current_t=2,
        )
        low_one = add_episode(
            episodes, 2, inflation=0.01, consumption=0.80, utility=10.0
        )
        low_two = add_episode(
            episodes, 3, inflation=0.02, consumption=0.80, utility=12.0
        )
        low_rule = semantic.propose(
            candidate_json(
                [low_one.episode_id, low_two.episode_id],
                condition_operator="<",
                condition_value=0.03,
                action_direction="at_least",
                action_threshold=0.70,
            ),
            current_t=4,
        )

        self.assertNotEqual(high_rule.rule_family_id, low_rule.rule_family_id)
        self.assertNotEqual(high_rule.rule_id, low_rule.rule_id)
        self.assertEqual(len(semantic.rules), 2)
        self.assertEqual(semantic.events[-1].event_type, "candidate_verified")
        scope_high = ContextScope(
            "inflation-regime", (ConditionPredicate("inflation", ">=", 0.04),)
        )
        scope_high_revision = ContextScope(
            "inflation-regime", (ConditionPredicate("inflation", ">=", 0.05),)
        )
        scope_high_other_label = ContextScope(
            "candidate-authored-display-label",
            (ConditionPredicate("inflation", ">=", 0.04),),
        )
        scope_low = ContextScope(
            "inflation-regime", (ConditionPredicate("inflation", "<=", 0.02),)
        )
        family_args = (
            ConditionPredicate("wealth", ">", 0.0),
            ActionGuidance("consumption_fraction", "at_most", 0.30),
            semantic.registered_outcome_criterion,
        )
        self.assertEqual(
            semantic._rule_family_id(scope_high, *family_args),
            semantic._rule_family_id(scope_high_revision, *family_args),
        )
        self.assertEqual(
            semantic._rule_family_id(scope_high, *family_args),
            semantic._rule_family_id(scope_high_other_label, *family_args),
        )
        self.assertNotEqual(
            semantic._rule_family_id(scope_high, *family_args),
            semantic._rule_family_id(scope_low, *family_args),
        )
        self.assertNotEqual(
            semantic._rule_family_id(ContextScope.global_scope(), *family_args),
            semantic._rule_family_id(scope_high, *family_args),
        )
        semantic.validate_referential_integrity()

    def test_scope_display_label_cannot_fragment_live_family_deduplication(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        support_ids = [first.episode_id, second.episode_id]
        scope_predicates = [
            {
                "field": "inflation",
                "operator": ">=",
                "value": 0.04,
                "tolerance": 1e-9,
            }
        ]
        first_rule = semantic.propose(
            candidate_json(
                support_ids,
                extra={
                    "context_scope": {
                        "scope_id": "high-inflation",
                        "predicates": scope_predicates,
                    }
                },
            ),
            current_t=2,
        )
        duplicate = semantic.propose(
            candidate_json(
                support_ids,
                extra={
                    "context_scope": {
                        "scope_id": "renamed-by-candidate",
                        "predicates": scope_predicates,
                    }
                },
            ),
            current_t=2,
        )

        self.assertEqual(duplicate.rule_id, first_rule.rule_id)
        self.assertEqual(len(semantic.rules), 1)
        self.assertEqual(
            semantic.events[-1].event_type,
            "duplicate_semantic_candidate_ignored",
        )
        restored = VerifiedSemanticRuleTrack.from_dict(
            semantic.to_dict(), episodic_track=episodes
        )
        self.assertEqual(restored.to_dict(), semantic.to_dict())

    def test_context_scope_is_enforced_during_retrieval(self):
        episodes, semantic = self.make_tracks()
        scoped = semantic.inject_active_rule(
            context_scope=ContextScope(
                "high-inflation",
                (ConditionPredicate("inflation", ">=", 0.04),),
            ),
            condition=ConditionPredicate("wealth", ">", 0.0),
            action_guidance=ActionGuidance(
                "consumption_fraction", "at_most", 0.30, 0.0
            ),
            outcome_criterion=OutcomeCriterion("flow_utility", ">", 0.0),
            rationale="Scoped experimental rule.",
            current_t=0,
            injection_id="context-scope",
            provenance={"experiment_id": "M3-SCOPE"},
        )

        self.assertEqual(
            semantic.retrieve(
                {"wealth": 100.0, "inflation": 0.05},
                current_t=0,
                log_selection=False,
            )[0].rule_id,
            scoped.rule_id,
        )
        self.assertEqual(
            semantic.retrieve(
                {"wealth": 100.0, "inflation": 0.02},
                current_t=0,
                log_selection=False,
            ),
            (),
        )
        restored = VerifiedSemanticRuleTrack.from_dict(
            semantic.to_dict(), episodic_track=episodes
        )
        restored.validate_referential_integrity()

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

        failure_two = add_episode(episodes, 4, consumption=0.20, utility=-2.0)
        retired = semantic.observe_episode(
            active.rule_id, failure_two.episode_id, current_t=5
        )
        self.assertEqual(retired.status, "retired")
        self.assertEqual(retired.consecutive_failures, 2)
        self.assertEqual(semantic.events[-1].event_type, "rule_retired")
        self.assertEqual(semantic.retrieve({"inflation": 0.05}, current_t=5), ())

    def test_restore_recomputes_harmful_compliance_streak_from_events(self):
        episodes, semantic = self.make_tracks(
            retirement_patience=5, retirement_confidence_threshold=0.0
        )
        active = self.activate_rule(episodes, semantic)
        failure = add_episode(episodes, 3, consumption=0.20, utility=-1.0)
        updated = semantic.observe_episode(
            active.rule_id, failure.episode_id, current_t=4
        )
        self.assertEqual(updated.consecutive_failures, 1)

        serialized = semantic.to_dict()
        serialized_rule = next(
            row for row in serialized["rules"] if row["rule_id"] == active.rule_id
        )
        serialized_rule["consecutive_failures"] = 0

        with self.assertRaisesRegex(ValueError, "consecutive_failures"):
            VerifiedSemanticRuleTrack.from_dict(
                serialized, episodic_track=episodes
            )

    def test_restore_replays_retirement_schedule_under_serialized_config(self):
        episodes, semantic = self.make_tracks(
            retirement_patience=5, retirement_confidence_threshold=0.0
        )
        active = self.activate_rule(episodes, semantic)
        failure = add_episode(episodes, 3, consumption=0.20, utility=-1.0)
        semantic.observe_episode(active.rule_id, failure.episode_id, current_t=4)

        serialized = semantic.to_dict()
        serialized["config"]["retirement_patience"] = 1

        with self.assertRaisesRegex(ValueError, "activation/retirement schedule"):
            VerifiedSemanticRuleTrack.from_dict(
                serialized, episodic_track=episodes
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
                "consumption_fraction", "at_least", 0.90, 0.01
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

        first_failure = add_episode(episodes, 0, consumption=0.95, utility=-1.0)
        still_active = semantic.observe_episode(
            injected.rule_id, first_failure.episode_id, current_t=1
        )
        self.assertEqual(still_active.status, "active")
        second_failure = add_episode(episodes, 1, consumption=0.95, utility=-2.0)
        retired = semantic.observe_episode(
            injected.rule_id, second_failure.episode_id, current_t=2
        )
        self.assertEqual(retired.status, "retired")
        semantic.validate_referential_integrity()

    def test_restore_recomputes_injected_rule_family_identity(self):
        episodes, semantic = self.make_tracks()
        semantic.inject_active_rule(
            condition=ConditionPredicate("inflation", ">", 0.03),
            action_guidance=ActionGuidance(
                "consumption_fraction", "at_most", 0.30, 0.01
            ),
            outcome_criterion=OutcomeCriterion("flow_utility", ">", 0.0),
            rationale="Controlled injected rule.",
            current_t=0,
            injection_id="family-integrity",
            provenance={"experiment_id": "M3-FAMILY"},
        )
        serialized = semantic.to_dict()
        rule = serialized["rules"][0]
        forged_family = f"family-{'0' * 20}:injected:deadbeef0000"
        forged_rule_id = f"{forged_family}:v1"
        rule["rule_family_id"] = forged_family
        rule["rule_id"] = forged_rule_id
        event = serialized["events"][0]
        event["rule_id"] = forged_rule_id
        event["metrics"]["rule_family_id"] = forged_family
        rehash_rule_event(serialized, 0)

        with self.assertRaisesRegex(ValueError, "inconsistent rule_family_id"):
            VerifiedSemanticRuleTrack.from_dict(
                serialized, episodic_track=episodes
            )

    def test_public_rules_events_and_serialization_are_deep_copies(self):
        episodes, semantic = self.make_tracks()
        returned = semantic.inject_active_rule(
            condition=ConditionPredicate("inflation", ">", 0.03),
            action_guidance=ActionGuidance(
                "consumption_fraction", "at_most", 0.30, 0.01
            ),
            outcome_criterion=semantic.registered_outcome_criterion,
            rationale="Aliasing regression fixture.",
            current_t=0,
            injection_id="no-alias",
            provenance={"experiment": {"name": "alias-test"}},
        )
        canonical = semantic.to_dict()

        returned.injection_provenance["experiment"]["name"] = "mutated-return"
        semantic.rules[0].injection_provenance["experiment"]["name"] = (
            "mutated-rules"
        )
        semantic.get(returned.rule_id).injection_provenance["experiment"][
            "name"
        ] = "mutated-get"
        semantic.events[0].metrics["evidence_type_counts"]["support"] = 999
        semantic.events[0].provenance["experiment"]["name"] = "mutated-events"
        snapshot = semantic.to_dict()
        snapshot["rules"][0]["injection_provenance"]["experiment"]["name"] = (
            "mutated-snapshot"
        )
        snapshot["events"][0]["metrics"]["evidence_type_counts"]["support"] = 999

        self.assertEqual(semantic.to_dict(), canonical)

    def test_retrieval_exposes_only_active_relevant_rules(self):
        episodes, semantic = self.make_tracks()
        first, second = self.seed_valid_support(episodes)
        provisional = semantic.propose(
            candidate_json([first.episode_id, second.episode_id]), current_t=2
        )
        injected = semantic.inject_active_rule(
            condition=ConditionPredicate("inflation", ">", 0.03),
            action_guidance=ActionGuidance(
                "consumption_fraction", "approximately", 0.20, 0.01
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
        self.assertEqual(semantic.retrieve({"inflation": 0.01}, current_t=2), ())
        self.assertEqual(semantic.events[-1].event_type, "active_rule_retrieved")

    def test_empty_retrieval_marker_is_opt_in_and_strictly_validated(self):
        episodes, semantic = self.make_tracks()
        state = {"inflation": 0.01}

        self.assertEqual(
            semantic.retrieve(
                state,
                current_t=0,
                limit=2,
                log_empty_selection=True,
            ),
            (),
        )
        marker = semantic.events[-1]
        self.assertEqual(marker.event_type, "active_rule_retrieval_empty")
        self.assertIsNone(marker.rule_id)
        self.assertEqual(marker.provenance["limit"], 2)
        self.assertRegex(marker.provenance["state_hash"], r"^[0-9a-f]{64}$")
        semantic.validate_referential_integrity()

        serialized = semantic.to_dict()
        serialized["events"][-1]["provenance"]["state_hash"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "empty retrieval marker"):
            VerifiedSemanticRuleTrack.from_dict(
                serialized, episodic_track=episodes
            )

    def test_retrieval_cannot_expose_final_active_state_at_an_earlier_time(self):
        episodes, semantic = self.make_tracks()
        active = self.activate_rule(episodes, semantic)
        before_events = semantic.events

        with self.assertRaisesRegex(ValueError, "must not move backward"):
            semantic.retrieve(
                {"inflation": 0.05}, current_t=active.created_at, log_selection=False
            )

        self.assertEqual(semantic.events, before_events)
        semantic.validate_referential_integrity()

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
            restored.retrieve({"inflation": 0.05}, current_t=5, log_selection=False)[
                0
            ].rule_id,
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
            VerifiedSemanticRuleTrack.from_dict(serialized, episodic_track=episodes)

    def test_legacy_snapshot_is_explicitly_fail_closed(self):
        episodes, semantic = self.make_tracks()
        serialized = semantic.to_dict()
        serialized["schema_version"] = "m3-verified-semantic-v1"

        with self.assertRaisesRegex(ValueError, "v1 snapshots are fail-closed"):
            VerifiedSemanticRuleTrack.from_dict(serialized, episodic_track=episodes)


if __name__ == "__main__":
    unittest.main()
