> [!WARNING]
> **HISTORICAL / LEGACY EVIDENCE ONLY**
>
> This document preserves rebuttal evidence produced by the legacy
> deterministic-template implementation before commit `3a3f30c`. It does not
> validate, and must not be cited as evidence for, the current
> Evidence-Grounded Rule Memory method or its conclusions. See the
> [legacy evidence index](README.md) and the
> [current architecture specification](../../docs/verified_dual_track_architecture.md).

# Draft response to Reviewer HoMs

We thank the reviewer for identifying places where our aggregate results were
not accompanied by sufficiently direct mechanism evidence. We added targeted
audits and a small paired control, and we narrowed claims that were stronger
than the evaluated implementation supports.

## Reflection and semantic-rule reliability

The review correctly exposed ambiguity in our description of reflection and
semantic memory. In the evaluated implementation, quarterly LLM reflection is
retained in the agent's decision context, but the persistent semantic-rule
state is not free-form LLM output. It is produced by deterministic
consolidation of logged episodes into two pre-specified rule templates. We will
revise the method to make this separation explicit and will not claim that the
system rejects hallucinated LLM-authored rules.

We nevertheless performed a rule-level provenance and confidence audit. Across
the completed GPT-4o and Gemini-3-Flash long-horizon runs, we recovered 9,400
rule create/update events from checkpoints. Every event reproduced the current
code's trigger, strategy, and confidence computation from its cited episodes
(9,400/9,400; zero code-level contradictions). The audit also revealed a real
limitation: 49.7% of source references reused episodes that had already
contributed to the same agent-rule trajectory, so confidence should not be
interpreted as independent accumulated evidence. Among 2,424 updates with
negative evidence, confidence decreased in 2,386; high-confidence rules did
not have consistently better three-month outcomes across the two backbones.

The five GPT-5.2 10-agent x 24-month full runs produced zero semantic-rule
activations because neither implemented trigger fired. We therefore remove any
claim that those short-horizon runs establish semantic-rule reliability. The
remaining hallucination risk lies in LLM rationales and reflection context and
is now stated as a limitation.

## Closed-loop mechanism trace

We added a deterministic, log-grounded trace selected by the lowest sentiment
state among eligible runs. At month 69 in the GPT-4o seed-13 run, the retrieval
query contained 6.018% inflation, 1.5% unemployment, and sentiment -0.120. The
decision prompt then contained the updated sentiment cue -0.148. The agent
retrieved episodes E68-E64 and a rule stating that inflation above 3% should
reduce consumption to 20% (confidence 0.667); its parsed action used 168 labor
hours and a 20% consumption fraction. Focal wealth rose by 11,595.91 in the
next transition and by 10,877.43 over six transitions, while mean wealth rose
by 55,737.37 over the same window.

This is an observational mechanism trace, not a causal estimate: aggregate
state changes combine all agents and environment dynamics. We also corrected
the earlier figure, which had called this state a crash and had filled sparse
annual macro fields with zero. The source `crash_flag` is 0.

## Dependence on the LLM backbone

We agree that FinEvo is not backbone-independent. The Llama-4-Maverick row is a
diagnostic counterexample to a universal improvement claim: it improves some
outcomes while degrading wealth and Gini relative to its text-only counterpart.
We therefore make the matched GPT-5.2 experiment the scope of the primary
performance claim and describe cross-model rows as route- and
backbone-dependent diagnostics. We do not tune Llama-4-Maverick post hoc to
rescue the row.

## Explicit cue control

To separate explicit prompt hinting from the remaining internal sentiment
state, we ran a same-day five-seed GPT-5.2 pair (10 agents x 24 months). Both
conditions keep endogenous sentiment and its numeric contribution to episodic
state similarity; only the explicit sentiment sentence is hidden.

Visible versus hidden cue results were: final wealth 183.1k +/- 13.3k versus
178.0k +/- 6.6k (hidden-visible difference -5.14k, 95% paired t CI
[-19.31k, 9.04k], p=.371); Gini .318 +/- .044 versus .363 +/- .014
(difference .045, CI [-.018, .109], p=.119); and full-horizon low-labor rate
.750% +/- .456% versus 1.417% +/- 1.337% (difference .667 percentage points,
CI [-.824, 2.157], p=.282). All 10 runs had zero invalid actions and zero API
errors.

The point estimates favor showing the cue in four of five seeds for wealth,
Gini, and low labor, but all three paired intervals include zero. Thus the
result supports neither a statistically detectable cue effect nor an
equivalence/non-inferiority claim. It does show that hiding the cue does not
produce an obvious point-estimate collapse: the hidden condition retains 97.2%
of visible-control mean wealth. We also clarify that the current
`regime_match` retrieval component is fixed at zero; this control isolates cue
text, not a fully implemented regime-matching retriever.

## Labor metric sensitivity

We audited matched GPT-5.2 baseline/FinEvo logs at thresholds of 1, 20, and 40
monthly hours. Executed labor is binary (0 or 168 hours), so all three executed
thresholds are identical by construction: 2.500% +/- .722% for text-only and
1.389% +/- .481% for FinEvo over three matched seeds. The continuous proposed
labor actions have zero mass below all three thresholds; their ranges are
120.96-168 hours and 144.48-168 hours, respectively. This rules out the
specific artifact in which FinEvo merely moves agents from 0 hours to 2-5
hours, but it also exposes ceiling saturation and a stylized binary labor
interface. We now state both limitations and do not present the three executed
thresholds as independent robustness checks.

## Textual events and related work

We agree that the reported event channel does not establish rich narrative or
real-news understanding. It is a synthetic aligned/shuffled/random textual
fixture and will be named accordingly; real-news evaluation remains future
work.

We also expand the comparison with EconAgent and EconAI. EconAgent uses a
perception-reflection-action loop with rolling dialogue memory and quarterly
LLM reflection. EconAI adds long- and short-term vector event memories,
fine-tuned event summarization, dynamic persona storage, an Economic Sentiment
Index, households and firms, and a textual COVID intervention. Our evaluated
implementation instead focuses on bounded outcome-indexed episodic retrieval,
deterministic inspectable rule consolidation, controlled cue/memory ablations,
and per-decision provenance. We will present these as overlapping but distinct
design choices rather than imply that memory plus sentiment is unique to
FinEvo.
