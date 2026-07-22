# Evidence-Grounded Rule Memory under Endogenous Multi-Agent Feedback

Status: **locally validated implementation specification; scientific evidence
pending**. Remote status is attached to the relevant pushed commit/PR rather
than asserted by this file.

The paper's primary question is no longer whether a memory-augmented agent
maximizes wealth in one simulator. It is whether an LLM agent can maintain an
auditable, evidence-linked, and intervention-testable memory under changing
conditions. The economic simulator is the controlled testbed; wealth and
aggregate macro variables are secondary diagnostics.

The research-method name is **Evidence-Grounded Rule Memory under Endogenous
Multi-Agent Feedback**. Code retains the historical `verified_memory` and
`VerifiedDualTrackMemory` identifiers for compatibility. Here, `verified`
means evidence-consistency and provenance checks only; it does not mean that a
rule is true, economically correct, optimal, hallucination-free, or causally
identified. The old hand-crafted sentiment cue, legacy deterministic semantic
templates, and the name `FinEvo` remain only for reproducing prior baselines.
The current completion ledger and scientific claim boundary are in
[`p0_methodology_completion_audit.md`](p0_methodology_completion_audit.md).

## Method modules

### M1: causal context router

M1 maps information observed no later than decision month `t` into a continuous
`ContextPacket`. Inputs may include a rolling macro/private-state window and a
structured event representation. Future state and outcome fields are forbidden.

M1 owns two independent routes:

1. `to_retrieval`: the context vector contributes to episodic and rule
   retrieval scores;
2. `to_prompt`: a deterministic causal summary is shown to the decision LLM.

The four required configurations are therefore:

| Variant | Retrieval route | Prompt route |
|---|---:|---:|
| no context | off | off |
| prompt-only | off | on |
| retrieval-only (primary) | on | off |
| full | on | on |

Every packet records a stable context ID, `observed_through`, feature schema,
encoder version, transformed vector, source hash, and routing flags. A frozen
predictive projection may be fitted on separate training seeds and loaded at
evaluation time. The unfitted fallback is an explicit causal rolling-feature
encoder, not a learned model.

For a value feature paired with `<feature>_available`, the fixed `last`,
`mean`, and `slope` slots use only rows whose mask is positive. `last` is the
latest available value, `mean` averages available values, and `slope` is the
first-to-last available-value change divided by their timestamp difference;
zero or one available observation gives slope `0`. An all-missing window maps
all three value slots to finite neutral zeros. Availability-mask features keep
ordinary rolling statistics, and the prompt still renders the current value as
`unavailable` whenever its current mask is off.

### M2: evidence-linked episodic track

M2 stores finalized transitions, not prompt-time impressions. Its write path is
split deliberately:

```text
begin(decision_t, state_t, context_t, retrieved memory, proposed/executed action)
environment.step(action_t)
finalize(state_t+1, realized consumption/labor, utility_t, reward_t, outcome_t)
```

Only finalized episodes are retrievable. Episode IDs include run, seed, agent,
and decision month. The compact evidence ledger outlives the bounded prompt
buffer so semantic provenance never points to an evicted object.

Episodic retrieval combines recency, observable-state similarity, importance,
and (only when enabled by M1) context similarity. Every component and candidate
rank is logged. The episode stores proposed action, executed action, any random
draw, realized labor hours, nominal/real consumption, wealth transition,
utility, reward, and the IDs of memory that influenced the decision.

### M3 v2: evidence-grounded semantic-rule track

M3 accepts structured LLM candidates but never treats candidate generation as
verification. Each persisted rule records:

- a stable `rule_family_id`, monotonic `rule_version`, `rule_id`,
  `supersedes_rule_id`, and `derived_from_rule_ids` lineage;
- one machine-checkable state condition plus a machine-checkable
  `ContextScope`; schema support for a scope is not evidence that the scope is
  scientifically useful;
- absolute action guidance using `at_least`, `at_most`, or `approximately`.
  Legacy `increase`, `decrease`, and `maintain` labels are accepted only through
  an explicitly logged migration and must not be described as relative actions;
- a rationale, unique evidence IDs by category, lifecycle timestamps,
  confidence, margins, transition reasons, and `provisional`, `active`,
  `retired`, or `rejected` status.

The verifier preregisters the outcome criterion; the candidate cannot choose
its own success metric or threshold. The default is
`utility_advantage > 0` with zero tolerance. Candidate evidence must resolve to
unique finalized M2 episodes, satisfy the condition and scope, and contain
executed actions consistent with the absolute guidance. The verifier also
searches the eligible ledger for unlisted condition-matched evidence.

Evidence is classified into five distinct categories:

- `support`: compliant action and successful registered outcome;
- `harmful_compliance`: compliant action and failed outcome;
- `alternative_success`: non-compliant action and successful outcome;
- `alternative_failure`: non-compliant action and failed outcome;
- `irrelevant`: condition or context scope does not apply.

The default weights are respectively `1.0`, `1.0`, `0.5`, `0.0`, and `0.0`.
These weights are configurable, logged protocol assumptions, not discovered
truth; they require preregistration, calibration, and sensitivity analysis.
Duplicate evidence is idempotent.

A valid candidate first enters the provisional buffer. It cannot activate from
candidate-generation evidence alone. Activation requires a currently observed
post-proposal episode classified as `support`, at least one post-proposal
support in total, the configured minimum total support, a positive configured
margin, and the confidence threshold. Only active, condition-matched,
scope-matched rules can enter the decision prompt. Harmful compliance and
alternative success contribute different negative weights; repeated failures
or confidence decay can retire a rule. A terminal rejected or retired family
may form a later version only with new support, while preserving lineage.
Rejected and retired versions remain in the audit ledger.

## M0 evaluation contract (not a memory track)

The current simulator reward is a marginal change in a legacy state metric and
is not presented to the LLM. It must not be mislabeled as realized monthly
utility. The redesign adds a side-effect-free environment ledger and explicit
flow-utility diagnostic:

```text
u_i,t = log(1 + real_consumption_i,t)
        - psi * (labor_hours_i,t / H)^(1 + nu) / (1 + nu)
```

`psi`, `nu`, `H`, and the discount factor are configuration values and must be
reported. The current Foundation environment has no borrowing/default state, so
the ledger records debt/default as not applicable rather than inventing a
bankruptcy penalty. The ledger separately records pre-wealth, income, taxes,
transfer, saving return/residual, consumption, and post-wealth. This diagnostic
does not silently change legacy environment dynamics. New-method prompts state
the consumption/labor/saving objective explicitly, and the direct-hours action
mode removes the old Bernoulli `0/168`-hour artifact for all matched systems.

Primary individual metrics are discounted cumulative flow utility, adaptation
regret after shocks, recovery time, action variance, and intervention effects.
Wealth, Gini, inflation deviation, GDP volatility, and simulator-defined
low-labor rates remain secondary diagnostics.

## Counterfactual evaluation layer

Each new run stores a decision snapshot containing the base prompt without
memory, state/context IDs, matched memory bundle, and action parser settings.
The replay harness holds state, model, prompt, decoding, and seed fixed while
changing only memory:

- matched memory;
- no memory;
- shuffled memory;
- context-mismatched memory;
- injected erroneous semantic rule.

The prompt-level replay is designed to report paired action changes. Immediate
utility and downstream transitions require a compatible environment checkpoint
and are not established by the current implementation-completion audit. A trace
alone is descriptive; an integrity-matched replay tests action sensitivity, but
residual provider nondeterminism must be bounded before attributing the change
strictly to memory. Older single-snapshot replays are historical regression
fixtures, not evidence for the current method.

The replay contract sends the protected decoding seed to the provider, records
the requested alias and served model, and rejects conflicting non-null system
fingerprints. OpenAI documents the seed as best-effort, not guaranteed
determinism, and the fingerprint may be absent; the result is therefore
controlled prompt-level sensitivity rather than strict causal identification.
It changes only episodic-entry order in the shuffled arm and keeps the
active-rule section byte-identical. The current P0 implementation moves
prompt-routed context into a protected base-prompt field and binds replay to
the source full-prompt hash. This path has passed the local validation gates.
Prompt-level replay still does not establish
downstream utility, next-state effects, or strict causal identification.

## Compatibility and provenance

- Historical `memory_module.DualTrackMemory`, `EpisodicMemory`, and
  `SemanticMemory` paths remain loadable for old pickle artifacts.
- The new implementation lives in `verified_memory/` and uses the separate
  `simulate_verified.py` entrypoint. The legacy `simulate.py` path is unchanged.
- Legacy flags map to documented compatibility behavior. In the verified path,
  episodic writes remain on for provenance, episodic retrieval has its own
  switch, and semantic proposal/verification/retrieval currently share one
  enable switch. Finer semantic-stage ablations remain future work.
- Existing `semantic_rules.jsonl` remains a compatibility snapshot. New runs
  additionally write `context_trace.jsonl`, `episodes.jsonl`,
  `semantic_rule_events.jsonl`, and `decision_snapshots.jsonl`.
- Current system-schema-v2 snapshots require the exact M1 decision-event and
  M2/M3 decision-retrieval arrays used for restore replay. Earlier
  in-development v2 snapshots without those arrays fail closed; no implicit
  migration is claimed.
- Every formal run records the exact code commit, dirty-worktree status,
  effective Foundation configuration and hash, source-config hash, requested
  and served model IDs, request seed, system fingerprint, usage/cache details,
  and provider request ID when the provider exposes them. Decision snapshots
  record the prompt schema version; the current artifact schema does not yet
  store a separate API-date field.
- Each budget reservation permits exactly one provider attempt. Post-setup
  execution failures seal their error and completed-call budget ledger;
  preflight failures remain stderr-only, and partial in-memory simulation
  streams are explicitly not checkpointed by the current writer.

## Validation gates before full experiments

Full 5-seed or 10-agent by 48--60-month experiments remain disabled until all
of the following pass:

1. M1 channel, causality, deterministic-ID, and frozen-projection tests;
2. M2 `action_t -> state_t+1` alignment and finalized-only retrieval tests;
3. M3 lifecycle, duplicate-evidence, counterevidence, rejection, activation,
   and retirement tests;
4. referential integrity for every memory and rule ID;
5. deterministic two-agent smoke through M1 -> M2 -> M3;
6. before any hosted-provider arm, a limited API preflight smoke with complete
   provider/parser ledgers and a preregistered parse-failure policy, with every
   failure accounted for;
7. forced false-rule injection is rejected, retired, or shown not to alter the
   paired action under an integrity-matched treatment;
8. 2x2 context configurations differ only in their intended route;
9. paired replay demonstrates that the memory manipulation—not an unrelated
   prompt/state change—is the only treatment;
10. legacy exporter/audit tests still pass.

The local gates have passed for the current worktree. Hosted-provider gate 6
has not been run for the current method and must be the first bounded stage of
any API-backed pilot. Passing the applicable gates establishes implementation
validity, not method superiority. A small matched experiment must then test
whether Evidence-Grounded Rule Memory improves utility, rule reliability,
retrieval quality, and post-shock adaptation.

## Current completion status (2026-07-22)

The current P0 implementation contains source paths and named tests for the
requirements in the completion audit and has passed the local validation
record documented there. Remote status belongs to the pushed commit/PR. No
scientific result or commit hash is asserted by this specification.
Older sealed runs, replays, and the historical smoke report predate the current
P0 changes; they are regression fixtures, not scientific evidence for this
method.

Here, `code-ready` means only that a small matched pilot may begin, with a
bounded API preflight required before any hosted-provider arm. Experiments A-D,
utility calibration/sensitivity, a second model, a method-matched 5-seed
result, and a full run have not been completed. Full
multi-seed or large-scale experiments remain gated by
[`p0_methodology_completion_audit.md`](p0_methodology_completion_audit.md).
