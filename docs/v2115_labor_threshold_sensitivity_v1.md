# V2.11.5 labor-threshold sensitivity publication contract

This publication is a zero-provider, retrospective diagnostic over the sealed
V2.11.5 evidence package. It exists to inspect whether conclusions about
executed labor actions are sensitive to three fixed thresholds. It does not
create or repair scientific results.

## Source and denominator

- Source: the annotated `pilot-v2.11.5-diagnostic-evidence-v1` package, resolved
  at commit `34134f2624833e45f0e1f559332b0d11ea1942d6`.
- Scope: all 45 registered Experiment A/C cells.
- Actor scope: 40 actor runs, each with 4 agents × 12 months = 48 registered
  action opportunities.
- Structural scope: five `verified-error-candidate` candidate-admission cells
  remain in the cell inventory but are not applicable to an action metric.
- Observed: 37 complete actor runs, or 1,776 actions.
- Missing: three failed Experiment A runs, or 144 actions. They remain null and
  are never retried, replaced, dropped, or imputed.

## Metric

The metric name is **below-threshold executed labor action rate**:

`count(executed labor_hours < h) / 48`

It is computed within each complete seed/run. The inference unit is the
seed/run, and summaries report the raw seed values plus unweighted mean,
median, and range. The frozen thresholds are:

- `h < 1`: grid value 0;
- `h < 20`: grid values 0, 8, and 16;
- `h < 40`: grid values 0, 8, 16, 24, and 32.

This metric is not unemployment. It does not identify employment state,
involuntary joblessness, labor-force participation, or a causal effect.

## Claim boundary

The output is `retrospective=true`, `descriptive=true`,
`diagnostic_only=true`, and `scientific_evidence=false`. It cannot restore or
reverse the authoritative Experiment A retrieval-effect no-go or Experiment C
rule-reliability no-go.

## Reproduction

After committing the implementation in a clean worktree, build once:

```bash
python scripts/build_v2115_labor_threshold_sensitivity.py
```

Validate the resulting package without modifying it:

```bash
python scripts/build_v2115_labor_threshold_sensitivity.py --validate-only
```

Both operations are local file and Git integrity checks only. The implementation
does not import provider clients, read `.env`, inspect API-key variables, or
make network requests.
