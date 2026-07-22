> [!WARNING]
> **HISTORICAL PRE-P0 V1 EVIDENCE ONLY**
>
> This trace comes from the legacy GPT-4o E1 deterministic-template system. It
> is not a current-method trace and must not be used as evidence for Evidence-
> Grounded Rule Memory v2.

# Case Study Extraction and Validation Report

## Verdict

The selected trace is derived from real run artifacts, not a hand-filled template. Native action, retrieval, semantic-rule, event, and API-error logs are checked byte-for-byte against the raw run directory. The trace is publishable only as an **observational closed-loop mechanism trace**, not as a single-agent causal estimate.

## Selected Trace

- Trace ID: `GPT-4o_seed13_m69_a61`
- Source run: `runs/E1/GPT-4o/finevo/default/seed_13`
- Raw source: `data/openai-gpt-4o-E1_finevo-seed13-100agents-240months`
- Month / agent: `69` / `61`
- Selection: `lowest_logged_sentiment`; this month is **not** a crash (`crash_flag=0`).
- Retrieval query: inflation `6.017598%`, unemployment `1.500000%`, sentiment `-0.120467`.
- Rule: `inflation is high (>3%)` → `reduce consumption to 20%` at confidence `0.666667`.
- Parsed action: labor-if-realized `168.0` hours, consumption `0.2`.
- Rule precondition matched: `True`; action target matched: `True`.
- Immediate focal wealth change (state m→m+1): `11595.905645`.
- Six-transition focal / mean wealth changes: `10877.434826` / `55737.373542`.

## Corrections Relative to the Previous Artifact

- The figure now says “lowest-sentiment decision state,” not “crash state.”
- The normalized trajectory exporter now carries the most recent annual macro values into non-boundary months; month 69 agrees with the raw checkpoint on inflation, unemployment, interest, and price.
- The raw checkpoint remains necessary to distinguish the pre-update sentiment used for memory retrieval from the post-update sentiment cue shown in the decision prompt.
- The action at month `t` is evaluated against the post-action state at `t+1`; no same-record episodic outcome is treated as its effect.
- `valid_json=true` is reported as parser acceptance. Because one code-fence repair was required, the route is not described as strict JSON without repair.
- The next-state and six-month values are labeled observational joint-system follow-up. They are not attributed causally to one agent.

## Provenance and Validation

- Schema valid: `True`
- Source values verified: `True`
- Native logs match raw source: `True`
- Retrieval score components sum to logged totals: `True`
- Immediate wealth delta matches the next checkpoint episode record: `True`
- Publishable as observational trace: `True`

Every source file is recorded with a SHA-256 digest and byte size under `case_study_trace.json → provenance.files`. Row and line pointers are zero-based and are stored under `provenance.pointers`.

## Missing Fields

- `text_event.value`: numeric-only or empty textual-event channel

## Excluded Higher-Priority Candidates

- `runs/E2/GPT-5.2/finevo/default/seed_13`: empty semantic_rules.jsonl
- `runs/E2/GPT-5.2/finevo/default/seed_2026`: empty semantic_rules.jsonl
- `runs/E2/GPT-5.2/finevo/default/seed_21`: empty semantic_rules.jsonl
- `runs/E2/GPT-5.2/finevo/default/seed_42`: empty semantic_rules.jsonl
- `runs/E2/GPT-5.2/finevo/default/seed_87`: empty semantic_rules.jsonl
- `runs/E5/GPT-5.2/finevo/default-prompt/seed_13`: empty semantic_rules.jsonl
- `runs/E5/GPT-5.2/finevo/default-prompt/seed_2026`: empty semantic_rules.jsonl
- `runs/E5/GPT-5.2/finevo/default-prompt/seed_21`: empty semantic_rules.jsonl
- `runs/E5/GPT-5.2/finevo/default-prompt/seed_42`: empty semantic_rules.jsonl
- `runs/E5/GPT-5.2/finevo/default-prompt/seed_87`: empty semantic_rules.jsonl

## Known Source Limitation

The simulator appends an episodic record before applying that month’s action. Its stored `wealth_change` is therefore the change entering the month, while the stored decision is the action chosen for the next transition. This artifact exposes that timing explicitly and uses state `t+1` for the observed follow-up. It does not silently relabel the same-record outcome.

## Reproduce

```bash
python artifacts/case_study/extract_case_study.py
python artifacts/case_study/extract_case_study.py --verify-only
```
