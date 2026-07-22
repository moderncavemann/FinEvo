> [!WARNING]
> **HISTORICAL PRE-P0 V1 EVIDENCE ONLY**
>
> This audit covers the legacy deterministic-template implementation and must
> not be cited as evidence for current Evidence-Grounded Rule Memory v2.

# FinEvo semantic-rule audit

This artifact audits the semantic-rule hallucination and error-reinforcement
question against real local run logs. It does not fabricate candidate rules,
rejections, manual labels, or counterfactual outcomes.

## Scope

The default cohort contains:

- GPT-4o E1 FinEvo, seed 13, 100 agents × 240 months;
- Gemini-3-Flash E1 FinEvo, seed 13, 100 agents × 240 months;
- GPT-5.2 E2 full, seeds 13, 21, 42, 87, and 2026, 10 agents × 24 months.

The two E1 runs are the only visible runs with semantic-rule activity. The five
GPT-5.2 E2 files contain real zero-row `semantic_rules.jsonl` logs and zero
final semantic rules because neither implemented trigger threshold activates.
They are retained in the audit as observed zeroes.

## Reproduce

Run from the repository root:

```bash
python artifacts/rule_audit/audit_semantic_rules.py
```

The command reads trusted local pickle checkpoints. Never point it at an
untrusted pickle file.

Optional parameters:

```bash
python artifacts/rule_audit/audit_semantic_rules.py \
  --horizon 3 \
  --high-confidence 0.75 \
  --output-dir artifacts/rule_audit/results \
  --run data/openai-gpt-4o-E1_finevo-seed13-100agents-240months
```

`--run` is repeatable. When omitted, the seven-run cohort above is used.

## Output meanings

- `audit_report.html`: primary technical report with an embedded live chart
  and same-data static fallback.
- `run_inventory.csv`: source presence, recovered episode coverage, trigger
  counts, and observed-zero versus active-rule status.
- `rule_snapshot_audit.csv`: every logged rule snapshot, including carried
  snapshots that contain no new evidence.
- `rule_update_audit.csv`: inferred create/merge events with episode-level
  condition, strategy, confidence, and provenance replay checks.
- `rule_use_outcomes.csv`: rule-selected decisions and descriptive next-H
  wealth/employment outcomes.
- `summary_by_run.csv`: compact run-level audit table.
- `metric_availability.csv`: explicit observed/computed/not-computable status
  for each requested analysis item.
- `negative_evidence_examples.jsonl`: log-grounded updates whose source
  evidence is negative; these are not causal failure claims.
- `source_manifest.csv`: source sizes and SHA-256 hashes.
- `audit_manifest.json`: parameters and output row counts.
- `source_notes.md`: report contract, chart map, and interpretation limits.

## Definitions and boundaries

The audited legacy implementation has two deterministic rule templates:
`high_inflation_strategy` and `high_unemployment_strategy`. It does not use LLM
reflection text to propose semantic rules. Therefore:

- accepted create/merge events can be reconstructed;
- mechanical episode support can be checked;
- confidence transitions and repeated episode references can be measured;
- candidate proposal and rejection counts cannot be recovered;
- an LLM hallucinated-rule or over-generalization rate is not applicable to
  the observed rule objects and cannot be computed from these logs;
- next-H outcomes are descriptive associations, not treatment effects.

`verified_supported` means only that a logged update exactly matches the
checked-in threshold, source timing, templated strategy, and confidence
equation. It is not a human judgment that the rule is economically sound.

## Tests

```bash
python -m unittest artifacts.rule_audit.test_audit_semantic_rules
```
