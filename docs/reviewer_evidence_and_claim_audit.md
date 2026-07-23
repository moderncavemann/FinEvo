> [!WARNING]
> **HISTORICAL / LEGACY EVIDENCE ONLY**
>
> This audit describes the deterministic-template implementation and evidence
> produced before commit `3a3f30c`. References below to "current code" are
> historical to that audit and must not be read as claims about the current
> Evidence-Grounded Rule Memory method. See
> [`paper/legacy/README.md`](../paper/legacy/README.md).

# Reviewer evidence and claim audit

This note records what the current code and logs can support after the HoMs
review. It is an internal authoring guardrail, not reviewer-facing prose.

## Bottom line

The requested lightweight evidence is now available: a rule-provenance audit,
an observational closed-loop trace, a labor-threshold audit, and a same-day
five-seed cue-visibility control. These results support a narrower paper about
auditable episodic retrieval, deterministic semantic consolidation, and an
explicit prompt cue whose removal does not cause an obvious point-estimate
collapse.

They do **not** support the stronger draft description of free-form LLM-created
semantic rules, rejection of hallucinated candidate rules, or an active
regime-match retrieval term.

## Evidence matrix

| Reviewer concern | Evidence now available | Defensible conclusion | Claim that remains unsupported |
|---|---|---|---|
| Reflection hallucination / error reinforcement | 12,400 rule snapshots and 9,400 create/update events audited against recovered episodes | All 9,400 updates reproduce the current deterministic threshold, strategy, and confidence code; 49.7% of source references reuse earlier evidence | A rate of hallucinated, rejected, or over-generalized LLM-authored rules; the current implementation has no such candidate/reject path |
| Closed loop | One deterministic GPT-4o trace with raw/checkpoint provenance and corrected macro state | State, retrieval, rule, and action are traceable; follow-up outcomes can be reported observationally | Causal attribution of the macro change to one agent or one rule |
| Explicit regime cue as a “cheat code” | Five same-day paired GPT-5.2 seeds, 10 agents x 24 months | Hiding cue text retains 97.2% of visible-control mean wealth; wealth/Gini/low-labor paired CIs all cross zero | Equivalence, non-inferiority, or proof that the cue has no effect |
| Near-zero unemployment artifact | Executed and proposed labor audited for matched GPT-5.2 legacy seeds | The result is not produced by moving proposed labor from 0 to 2-5 hours | Realistic labor calibration; executed labor is binary 0/168 and proposed labor is ceiling-saturated |
| Backbone dependence | Existing cross-model diagnostics plus rule-audit availability check | FinEvo gains must be described as backbone- and route-dependent; the matched GPT-5.2 suite is the primary evidence | A universal model-family or architecture claim |
| Textual events | Existing numeric/text/shuffled/random fixture controls | The channel tests sensitivity to synthetic text alignment | Real-news understanding or long-form narrative reasoning |

## Material claim-implementation gaps

1. `memory_module.py` creates semantic memory through two deterministic
   templates (`high_inflation_strategy` and `high_unemployment_strategy`). It
   does not parse quarterly LLM reflection into semantic rules.
2. There is no semantic candidate/reject branch. Logged `validity_note` values
   are labels written by the logger, not independent validation outcomes.
3. The retrieval trace contains a `regime_match` field, but its value is fixed
   at `0.0`. Sentiment is currently one numeric feature in cosine state
   similarity.
4. All five GPT-5.2 10-agent x 24-month full runs contain zero semantic-rule
   activations because neither implemented trigger fires. The semantic-memory
   ablation at this horizon therefore cannot be interpreted as an active-rule
   mechanism ablation.
5. Episodic record `E_t` stores the wealth change observed on entry to month
   `t` beside `decision_t`; the effect of `decision_t` must be checked in the
   transition to `t+1`, not read from the same record.

## Recommended revision path

For the current rebuttal, revise the method and claims to match the evaluated
implementation:

- separate quarterly LLM reflection context from deterministic semantic-rule
  consolidation;
- remove claims that unsupported LLM-authored rules are rejected;
- describe sentiment-weighted state similarity, not active regime-match
  retrieval;
- treat the five-seed GPT-5.2 semantic activation count of zero as a limitation;
- call the trace observational and correct the one-step outcome alignment;
- rename the event result a synthetic textual-event control;
- make matched GPT-5.2 results primary and cross-model rows diagnostic.

If the paper instead retains the intended LLM-generated,
regime-conditioned semantic-rule mechanism, that mechanism must first be
implemented with episode IDs, candidate/reject logging, regime tags, and
condition-gated confidence updates. At minimum, the main matched comparison
and semantic/reflection ablations would then need to be rerun. Results from the
current deterministic implementation cannot be relabeled as evidence for that
new mechanism.

## Reproduction

```bash
python artifacts/rule_audit/audit_semantic_rules.py
python artifacts/case_study/extract_case_study.py --verify-only
python artifacts/labor_threshold_sensitivity/analyze_labor_thresholds.py
python artifacts/no_visible_cue/validate_no_visible_cue.py
python artifacts/no_visible_cue/analyze_paired_results.py
```
