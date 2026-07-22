# P0 methodology completion audit

Status: **locally validated and code-ready for a small matched pilot;
scientific evidence pending**. Remote status belongs to the pushed commit/PR
and is not asserted by this file.

This is a completion ledger for the current implementation and its local
validation record, not a scientific-results report. It maps the author-provided
P0-1 through P0-8 requirements, the protected prompt/replay contract, the
semantic parse contract, and legacy-evidence isolation to repository code and
named tests. The final local run accepted 264 repository tests together with
the compilation, diff-hygiene, manifest-integrity, deterministic no-network
G0, and release-scope credential-pattern gates described below. No scientific
experiment, commit hash, or completed remote check is asserted here.

## Frozen research contract

**Method name:** **Evidence-Grounded Rule Memory under Endogenous Multi-Agent Feedback**.

**Research question:** How should evidence-grounded rules be created,
activated, contradicted, versioned, and retired when memory-guided actions
endogenously change the multi-agent environment in which those rules will
later be reused?

**Setting:** A controlled, stylized multi-agent economic simulator used as a
dynamic testbed for memory mechanisms. At decision time `t`, an agent may use
only causally available state/context, finalized episodes, and active rules.
The agent emits bounded labor/consumption actions; the environment produces
the next state and realized flow-utility ledger; only then may the transition
be finalized as evidence.

**Primary evidence targets:**

- evidence-linked rule reliability: false-rule activation, harmful-rule
  survival, retirement latency, rule calibration, and evidence taxonomy;
- dynamic adaptation: discounted cumulative flow utility, adaptation regret,
  and recovery time after preregistered shocks;
- retrieval mechanism: long-horizon recall, retrieval quality/overlap, and
  the independent effects of retrieval-routed versus prompt-routed context;
- intervention behavior: matched action deltas and, only with a compatible
  checkpoint rollout, immediate utility, next-state, downstream utility, and
  population-aggregate deltas.

**Metric boundary:** Wealth, Gini, inflation deviation, GDP volatility, and
simulator-defined low-labor measures are **secondary diagnostics**. They are
not the primary evidence for the memory-method claim.

**Meaning of `verified`:** In code and artifact names, `verified` means only
machine-checked evidence consistency and provenance: valid schema, resolvable
references, causal time ordering, condition/action consistency, a
verifier-registered outcome criterion, rule lifecycle checks, hashes, and
lineage. It does **not** mean that a rule is true, economically correct,
optimal, hallucination-free, robust in deployment, or causally identified.

**Explicit non-claims:**

- This is not a model of real financial markets and does not establish claims
  about portfolio choice, borrowing, credit, default, asset pricing, firms,
  policy forecasting, or real-world deployment.
- Provider-neutral execution does not establish backbone independence.
- A hash-bound hosted-model replay is controlled prompt-level sensitivity,
  not strict causal identification; hosted seeds may remain best-effort.
- Implementation validity does not establish method superiority.
- Legacy deterministic-template FinEvo results do not validate the current
  method.

## Completion semantics

- **Implementation present** means that the named source path and test case
  exist in the current worktree after static inspection.
- **Locally validated** means the named repository tests and local
  CI-equivalent gates passed for this worktree. It does not certify hosted
  provider behavior or scientific effectiveness.
- **Remote pending/external** means GitHub status belongs to the exact pushed
  commit and must be read from its PR/checks rather than predicted here.
- **Code-ready** means only
  eligible to enter a small matched pilot. It does not mean scientifically
  validated, paper-ready, or ready for a 5-seed/full-scale run.
- Experiments A-D, utility calibration/sensitivity, a second model, any
  proposer-by-actor cross, a method-matched 5-seed result, and a full run
  have not been run for the current method.
- No current-method hosted-provider/API smoke was run in this change set. A
  bounded, preregistered API preflight is the first gate before any hosted arm
  of the pilot.

## Requirement-to-evidence matrix

| ID | Requirement | Implementation evidence | Test evidence present | Current status | Remaining scientific evidence |
| --- | --- | --- | --- | --- | --- |
| P0-1 | Retrieve from the full finalized M2 ledger; use `prompt_capacity` only to bound returned Top-K; exclude pending/future outcomes and preserve behavior across restore. | `verified_memory/m2_episodic.py`: `EvidenceLinkedEpisodicTrack._ledger`, `retrieve`, `from_dict`, `validate_references`; `verified_memory/system.py`: exact decision query ledger and restore replay. | `tests/test_m2_episodic.py`: long-horizon capacity, pending/future exclusion, canonical prompt-suffix, native identity, and defensive-copy tests; `tests/test_verified_memory_system.py`: exact historical retrieval-state and Top-K restore tests. | **Locally validated.** | Measure recall/retrieval quality at long horizons and test whether access to old evidence improves utility/adaptation in Experiments A-B. A synthetic retrieval test is not long-horizon scientific evidence. |
| P0-2 | A provisional rule may activate only after at least one qualifying post-proposal **support**; a later contradiction must never satisfy the delay gate. | `verified_memory/m3_semantic.py`: `VerifiedRule.post_proposal_support_count`, `post_proposal_contradiction_count`, `activation_episode_id`; `VerifiedSemanticRuleTrack.observe_episode`. | `tests/test_m3_semantic.py`: `test_candidate_is_never_directly_active_and_distinct_evidence_activates`, `test_first_postproposal_contradiction_can_never_activate_rule`, `test_preproposal_unlisted_evidence_cannot_satisfy_activation_delay`, `test_restore_rejects_falsified_postproposal_timing`. | **Locally validated.** | Experiment C must estimate false-rule activation, harmful-rule survival, retirement latency, and utility loss for natural and injected rules over paired seeds. |
| P0-3 | Separate evidence into `support`, `harmful_compliance`, `alternative_success`, `alternative_failure`, and `irrelevant`; do not treat all off-policy outcomes as equally contradictory; expose weights and counts for audit. | `verified_memory/m3_semantic.py`: `EVIDENCE_TYPES`, `DEFAULT_EVIDENCE_WEIGHTS`, `_classification`, `_scores_from_categories`, per-category episode IDs, `RuleEvent.metrics`. | `tests/test_m3_semantic.py`: five-way weights, post-proposal irrelevant counting, exhaustive creation search, historical event-metric replay, and idempotence tests; `tests/test_runner_artifacts.py`: sealed-stream lifecycle/irrelevant replay tests. | **Locally validated.** | Preregister and sensitivity-test taxonomy weights; compare categories with human- or protocol-labeled rule outcomes and report calibration rather than assuming the default weights are correct. |
| P0-4 | Use accurate absolute action semantics (`at_least`, `at_most`, `approximately`) instead of describing threshold checks as relative `increase/decrease`; make legacy migration explicit. | `verified_memory/m3_semantic.py`: `ActionGuidance`, `from_candidate_dict`, `build_proposal_prompt`; canonical directions and migration notes. | `tests/test_m3_semantic.py`: `test_legacy_direction_and_criterion_migration_is_explicit`, `test_condition_action_and_outcome_must_all_match_claimed_support`, `test_proposal_prompt_lists_schema_allowed_fields_and_episode_ids`. | **Locally validated.** | Inspect action distributions and rule-following rates in pilots. Paper language must describe absolute thresholds unless a separate relative-action mechanism is implemented and tested. |
| P0-5 | The verifier, not the LLM candidate, preregisters the outcome metric/operator/threshold; candidate-generation evidence and post-proposal activation evidence remain time-separated. | `verified_memory/m3_semantic.py`: `VerifiedSemanticRuleTrack.registered_outcome_criterion`, `parse_candidate`, `submit_candidate`, serialized verifier config. A legacy criterion is accepted only when identical to the registered criterion and is recorded as migration. | `tests/test_m3_semantic.py`: `test_candidate_cannot_choose_outcome_threshold_and_config_is_sealed`, `test_candidate_and_observation_cannot_use_future_finalized_evidence`, `test_preproposal_unlisted_evidence_cannot_satisfy_activation_delay`. | **Locally validated.** | Calibrate/sensitivity-test the preregistered utility-advantage criterion and compare full verifier versus verifier-disabled arms in Experiment C. Self-consistency alone is insufficient. |
| P0-6 | Represent rule family, version, context scope, and lineage; allow a terminal rejected/retired family to form a new version only with new evidence; enforce scope during retrieval. | `verified_memory/m3_semantic.py`: `ContextScope`, `rule_family_id`, `rule_version`, `supersedes_rule_id`, `derived_from_rule_ids`, `_family_rules`, family/version integrity checks, scope-gated `retrieve`; artifact/system restore paths replay historical eligibility and ranking. | `tests/test_m3_semantic.py`: versioning, duplicate-scope, injected-family, context-scope, round-trip, and fail-closed legacy tests; `tests/test_verified_memory_system.py` and `tests/test_runner_artifacts.py`: exact same-time M3 selection/ranking and forged-selection rejection. | **Locally validated.** | Report natural version creation, scope applicability, supersession, and failure cases in Experiment C; do not infer useful context specialization from schema support alone. |
| P0-7 | Do not initialize month 0 as artificial 100% or 0% low labor. Encode prior-period unavailability explicitly and identically across all four M1 routes, state similarity, and prompts. | `verified_memory/runner.py`: `previous_low_labor_rate=None`, `_context_observation`, `_m2_state`, `_prompt_state`; `verified_memory/prompts.py`: `DecisionPromptState.previous_period_available`; `verified_memory/m1_context.py`: availability-mask narration/hash behavior; `verified_memory/foundation_adapter.py`: reset-safe state mapping. | `tests/test_verified_runner.py`: `test_bootstrap_missingness_contract_is_shared_by_all_context_routes`; `tests/test_m1_context.py`: `test_missing_value_mask_is_hashed_but_placeholder_is_not_narrated`; `tests/test_verified_prompts.py`: `test_bootstrap_prompt_marks_prior_period_unavailable`; `tests/test_foundation_adapter.py`: `test_snapshot_capture_is_t0_safe_and_uses_explicit_current_fields`; artifact reconstruction rejects t0 substitution. | **Locally validated.** | Run utility calibration and action-distribution sensitivity before interpreting labor choices; confirm the first-period fix does not merely move saturation to another initialization choice. |
| P0-8 | Add reproducible CI for tests, compilation, diff hygiene, manifest integrity, secret scanning, and deterministic no-network G0 execution. | `.github/workflows/verified-memory-ci.yml`: commit-pinned actions, exact Python and direct dependency versions, Linux+macOS test matrix, compile check, diff check, high-confidence secret scan, trusted hashes plus manifest rehash, and two-run G0 comparison with an explicit scripted M3 contract; `.github/network_guard/sitecustomize.py`: fail-closed Python outbound-socket guard for G0. Transitive packages are resolved afresh and are not hash-locked. | Local parity accepted: 264 tests, compilation/diff checks, six trusted-manifest anchors plus rehash/load checks, release-scope credential-pattern scan, and byte-identical twin outbound-socket-denied G0 runs with the scripted M3 contract. | **Local parity validated; remote status external.** | CI is reproducibility infrastructure, not scientific evidence. Its anchored rehash of older sealed manifests preserves those historical bytes but does not upgrade them into evidence for the current P0 implementation. |
| C-1 prompt/replay | Keep prompt-routed M1 context in the protected base prompt, vary only the replaceable memory block, bind replay to the source full-prompt hash and prompt schema, and fail closed on mismatch. | `verified_memory/system.py`: `protected_context_prompt` versus `memory_prompt`; `verified_memory/prompts.py`: `PROMPT_SCHEMA_VERSION`, `build_base_decision_prompt`; `verified_memory/runner.py`: protected context/hash in snapshots; `verified_memory/replay.py`: `DecisionSnapshot.source_full_prompt_hash`, `verify_treatment_integrity`; `verified_memory/replay_experiment.py`: `build_paired_snapshot`. | `tests/test_verified_prompts.py`: `test_only_memory_block_changes`; `tests/test_counterfactual_replay.py`: prompt-difference, tampering, source-hash, parser/provider fail-closed tests; `tests/test_replay_experiment.py`: sealed-run pairing, exact legacy-v1 reconstruction, and four-route hash binding. | **Locally validated.** | Experiment A must run all four context routes. Experiment D must continue the same checkpoint for 3-6 steps to measure utility/next-state/population effects. Until then replay supports prompt-level action sensitivity only. |
| C-2 parse contract | Predeclare semantic parse handling: `record-and-skip` retains malformed candidates in the scientific denominator and prevents activation; `fail-run` emits a structured failure; provider failures always abort; record parse mode, hashes, and counts. | `simulate_verified.py`: `--semantic-parse-failure-policy`; `verified_memory/runner.py`: `SEMANTIC_PARSE_FAILURE_POLICIES`, `RunnerFailure`, `_semantic_parse_mode`, proposal/error streams, summary counts; `verified_memory/runner_artifacts.py`: optional semantic streams for zero-accepted-rule runs. | `tests/test_verified_runner.py`: `test_all_semantic_parse_failures_remain_in_denominator_and_run_seals`, `test_semantic_parse_fail_run_has_structured_failure_details`, `test_semantic_provider_failure_always_aborts_with_structured_receipt_data`; artifact tests reconcile proposal/event/summary denominators. | **Locally validated.** | Freeze the policy before paid runs and report attempts, exact/fenced/substring recovery, failures, provider errors, missing runs, and seed denominators by treatment/model. Do not drop a model or seed because its parse rate is poor. |
| C-3 legacy isolation | Preserve pre-`3a3f30c` deterministic-template material and later pre-P0 v1 artifacts as distinct historical strata, keep stable redirects, and prevent old paper/results from being cited as current-method evidence. | `paper/reviewer_HoMs_rebuttal_draft.md` redirect; `paper/legacy/reviewer_HoMs_rebuttal_legacy.md`; `paper/legacy/README.md`; a visible compiled warning in `paper/main.tex`; explicit opt-in fail-closed guards in `paper/experiments.tex` and every `paper/generated_tex/*.tex`; historical banners on prior audit/smoke reports. | `tests/test_legacy_evidence_guards.py` inventories all guarded TeX and checks that the historical rebuttal path remains a redirect. | **Locally validated.** | All current-method tables and claims require method-matched reruns. Legacy aggregate values may appear only in explicitly labeled historical/reproduction sections, never as Evidence-Grounded Rule Memory results. |

## Artifact and evidence boundary

Older files under `artifacts/verified_runs/`, `artifacts/verified_replays/`, and
`artifacts/verified_memory_smoke_report.md` predate the current P0 completion
changes. They may remain useful as historical regression fixtures or integrity
targets, but they are **not** scientific evidence for the current method and
must not be cited as a current-method smoke, replay, or performance result.
Rehashing an old manifest proves only that the old bytes still match it.

No current-method quantitative result is accepted by this audit. The current
worktree has passed its local implementation gates and is code-ready only for
a small matched pilot; scientific evidence remains pending.

## No-go conditions

### No-go before a paid small matched pilot

Stop and fix the implementation if any of the following holds:

1. The local/remote gates for the exact pilot commit are not green, or a run
   cannot be tied to a clean code/config provenance record.
2. Full-ledger retrieval loses old finalized evidence after serialization,
   admits pending/future evidence, or silently reverts to the bounded buffer.
3. A post-proposal contradiction can activate a rule, evidence categories or
   lineage fail integrity checks, or an inactive rule reaches the actor.
4. Prompt-routed context appears inside the replaceable memory block, the
   matched source prompt cannot be reconstructed byte-for-byte, or a treatment
   changes protected model/state/parser/decoding fields.
5. Parse/provider outcomes are absent from the denominator, the parse policy
   is selected after looking at results, or failure receipts omit the raw
   output/prompt hashes needed for audit.
6. Month-0 missingness is again encoded as a substantive low-labor value.
7. Legacy artifacts or tables are relabeled as current-method evidence.

### No-go before paper claims or a full-scale run

Do not advance to a full run or rewrite the paper around effectiveness claims
unless:

1. Utility calibration/sensitivity yields a defensible region and the action
   distribution is not explained only by 152/168-hour ceiling saturation.
2. Experiment A separates prompt hinting from retrieval conditioning under
   matched routes and reports retrieval-specific diagnostics.
3. Experiment C shows that the verifier reduces false-rule activation or
   harmful-rule persistence relative to a preregistered unverified control;
   otherwise narrow or reject the rule-reliability claim.
4. Experiment D includes checkpoint-backed utility and next-state effects;
   otherwise retain only the prompt-level action-sensitivity claim.
5. A second model completes at least a small pilot with parse/provider failures
   reported. Provider-neutral code alone is not sufficient.
6. Method-matched stochastic comparisons use the preregistered paired seed set
   and uncertainty. No 5-seed or full-run evidence exists yet.

## Next small experiments

The next authorized scientific stage is a small matched pilot, not a
100-agent by 240-month run. The intended common pilot is 10 agents by 24
months with 5 paired seeds, but that design is still planned and unrun.

### Stage 0: utility/action calibration

Before interpreting method effects, preregister a small grid over labor
disutility weight, inverse Frisch elasticity, and consumption scale. Report
labor/consumption distributions, ceiling mass, budget residuals, and utility
components. Choose a stable region by an ex-ante rule, not the best method
outcome.

### Experiment A: M1 route decomposition

Compare `no-context`, `prompt-only`, `retrieval-only`, and `full` with all
other inputs, budgets, actor, shocks, and seeds matched. Primary evidence is
discounted utility, adaptation regret, recovery time, retrieval
quality/overlap, and action distribution; macro quantities remain secondary.

### Experiment B: memory architecture

Compare no memory, episodic-only, semantic-only decision input, unverified
dual memory, and full Evidence-Grounded Rule Memory. The semantic-only arm may
retain M2 writes for provenance but must not expose episodes to the decision
actor.

### Experiment C: rule reliability

Cross natural candidates with injected false rules, shuffled evidence IDs,
wrong-context rules, verifier-disabled, and full-verifier treatments. Report
candidate parse/acceptance, false-rule activation, harmful-rule survival,
retirement latency, cumulative utility loss, and calibration.

### Experiment D: checkpoint-backed intervention

From the same environment checkpoint, apply matched, no-memory, shuffled,
context-mismatched, and erroneous-rule treatments, then continue for at least
3-6 environment steps. Report action, immediate utility, next state,
downstream cumulative utility, and population-aggregate deltas. Hosted-model
results must retain the residual nondeterminism caveat.

### After P0

Only after A-D and calibration should work proceed to a second model, a
proposer-by-actor cross, narrative content/order/opposite-event interventions,
and matched EconAgent/EconAI-style baselines. A 48/60-month extension or
100-agent by 240-month run is gated on those smaller falsification tests, not
on aggregate wealth looking favorable.

## Final completion checklist

- [x] Final local validation accepted (264 tests plus local CI-equivalent gates).
- [ ] Remote CI/status checks accepted.
- [ ] Clean code/config provenance frozen for the pilot.
- [ ] Parse-failure policy and seed/denominator rules preregistered.
- [ ] Utility calibration/sensitivity completed.
- [ ] Experiment A completed.
- [ ] Experiment B completed.
- [ ] Experiment C completed.
- [ ] Experiment D completed with downstream checkpoint continuation.
- [ ] Second-model pilot completed.
- [x] Current and legacy results are kept in separate tables and prose.

Until the scientific boxes are supported by method-matched artifacts, the
correct status is: **implementation locally validated and code-ready for a
small matched pilot; scientific evidence pending**.
