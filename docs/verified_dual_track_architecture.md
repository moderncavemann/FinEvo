# Verified dual-track memory redesign

Status: implementation specification for the post-EMNLP redesign.

The paper's primary question is no longer whether a memory-augmented agent
maximizes wealth in one simulator. It is whether an LLM agent can maintain an
auditable, evidence-linked, and intervention-testable memory under changing
conditions. The economic simulator is the controlled testbed; wealth and
aggregate macro variables are secondary diagnostics.

The working method name in code is **verified dual-track memory**. This is a
descriptive label, not a final paper brand. The old hand-crafted sentiment cue,
legacy deterministic semantic templates, and the name `FinEvo` remain only for
reproducing prior baselines.

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

### M3: verified semantic-rule track

M3 accepts structured LLM candidates but never trusts them directly. A rule has:

- `rule_id` and version;
- one machine-checkable state predicate in schema v1 (a conjunction is a future
  schema extension, not a current claim);
- structured labor/consumption guidance;
- rationale;
- unique supporting and contradicting episode IDs;
- context scope;
- `provisional`, `active`, `retired`, or `rejected` status;
- support, contradiction, confidence, and transition reason.

The verifier checks that cited episodes exist, are unique, satisfy the
condition, contain actions consistent with the guidance, and meet an
outcome/utility criterion. It also searches the ledger for condition-matched
counterevidence. Duplicate evidence is idempotent.

A valid candidate first enters the provisional buffer. Admission requires at
least two independent supporting episodes. It becomes active only after a
positive support-minus-contradiction margin and at least one qualifying episode
whose outcome is strictly later than rule creation (the default activation
threshold is three total supports). Only active
rules can enter the decision prompt. Repeated negative outcomes or new
counterevidence lower confidence and retire the rule. Rejected and retired
rules remain in the audit ledger.

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

The implemented prompt-level replay reports paired action changes. Immediate
utility and downstream transitions require a compatible environment checkpoint
and are deliberately not claimed by the current smoke. A trace alone is
descriptive; an integrity-matched replay is evidence for memory-induced action
change, not for improved economic outcomes.

The replay contract sends the protected decoding seed to the provider, records
the requested alias and served model, and rejects conflicting non-null system
fingerprints. OpenAI documents the seed as best-effort, not guaranteed
determinism, and the fingerprint may be absent; the result is therefore
controlled prompt-level sensitivity rather than strict causal identification.
It changes only episodic-entry order in the shuffled arm and keeps the
active-rule section byte-identical. Prompt-only/full M1 runs
currently fail closed at snapshot construction because prompt-routed context is
still co-located with memory text; support for those routes requires moving
that context into the protected base prompt first.

## Compatibility and provenance

- Historical `memory_module.DualTrackMemory`, `EpisodicMemory`, and
  `SemanticMemory` paths remain loadable for old pickle artifacts.
- The new implementation lives in `verified_memory/` and uses the separate
  `simulate_verified.py` entrypoint. The legacy `simulate.py` path is unchanged.
- Legacy flags map to documented compatibility behavior. New experiments use
  separate switches for episodic writes/retrieval and semantic
  generation/verification/retrieval.
- Existing `semantic_rules.jsonl` remains a compatibility snapshot. New runs
  additionally write `context_trace.jsonl`, `episodes.jsonl`,
  `semantic_rule_events.jsonl`, and `decision_snapshots.jsonl`.
- Every formal run records the exact code commit, dirty-worktree status,
  effective Foundation configuration and hash, source-config hash, requested
  and served model IDs, request seed, system fingerprint, usage/cache details,
  API date, and prompt version when the provider exposes them.
- Each budget reservation permits exactly one provider attempt. A failed CLI
  run seals its error and completed-call budget ledger; partial in-memory
  simulation streams are explicitly not checkpointed by the current writer.

## Validation gates before full experiments

Full 5-seed or 10-agent by 48--60-month experiments remain disabled until all
of the following pass:

1. M1 channel, causality, deterministic-ID, and frozen-projection tests;
2. M2 `action_t -> state_t+1` alignment and finalized-only retrieval tests;
3. M3 lifecycle, duplicate-evidence, counterevidence, rejection, activation,
   and retirement tests;
4. referential integrity for every memory and rule ID;
5. deterministic two-agent smoke through M1 -> M2 -> M3;
6. limited API smoke with zero provider/parser failures and complete ledgers;
7. forced false-rule injection is rejected, retired, or shown not to alter the
   paired action under an integrity-matched treatment;
8. 2x2 context configurations differ only in their intended route;
9. paired replay demonstrates that the memory manipulation—not an unrelated
   prompt/state change—is the only treatment;
10. legacy exporter/audit tests still pass.

Passing these gates establishes implementation validity, not method
superiority. Only after the gates pass should a small matched experiment test
whether verified dual memory improves utility, rule reliability, retrieval
quality, and post-shock adaptation.

## Current gate status (2026-07-22)

- Gates 1--6 and 10 pass in the bounded implementation suite.
- Gate 7 passes only at prompt-level for one GPT-5.2 snapshot: the strengthened
  injected rule did not alter its action. Retirement remains unit-tested, but a
  downstream false-rule rollout is still pending.
- Gate 8 passes at the routing/unit level; the four API context variants have
  not been run.
- Gate 9's code-level integrity checks pass, but the stored v2 API replay
  predates enforced seed forwarding and correct section-preserving shuffling.
  A replacement sealed replay is required before claiming paired action
  sensitivity. Downstream checkpoint replay remains pending.

Therefore full multi-seed experiments remain disabled. See
`artifacts/verified_memory_smoke_report.md` for the bounded evidence and claim
limits.
