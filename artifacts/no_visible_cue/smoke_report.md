> [!WARNING]
> **HISTORICAL PRE-P0 V1 EVIDENCE ONLY**
>
> This code-path smoke used legacy `simulate.py`; it is not current-method
> scientific evidence or current M1 route validation.

# No-visible-cue paired smoke report

Date: 2026-07-10

Two real GPT-5.2 runs were completed with seed 314159, 10 agents, one month,
temperature 0.2, and top-p 1.0. The runs differed only in
`show_regime_cue`:

| Variant | Cue flag | Valid actions | API errors | Cost (USD) | Summary SHA-256 |
|---|---:|---:|---:|---:|---|
| visible-cue-control | true | 10/10 | 0 | 0.015279 | `ed42cf8ea0f8e4b7bad2806d1eabb7ed4553924a8d853bd8ee55511619a59ea6` |
| no-visible-cue | false | 10/10 | 0 | 0.014733 | `1687e6b51c3e8d04184a84978de8da433507cc8c8e1a168b900d79e97ecaabac` |

Prompt audit across the 10 matched agents:

- all 10 visible-control prompts contain the neutral cue text;
- none of the 10 hidden prompts contains it;
- after removing that cue sentence from the visible prompt, every matched
  prompt is byte-equivalent after whitespace normalization;
- both runs completed with zero parser errors and zero provider errors.
- normalized smoke configs preserve `show_regime_cue: true/false` under the
  corresponding `runs/E9-smoke/...` folders.

Local ignored source folders:

- `data/openai-gpt-5.2-E9_smoke_visible_cue_control-seed314159-10agents-1months`
- `data/openai-gpt-5.2-E9_smoke_no_visible_cue-seed314159-10agents-1months`

This smoke test validates wiring and prompt isolation. Its one-month outcome
difference is not evidence for or against the paper's performance claim; the
planned same-day five-seed 10-agent/24-month pair is required for that.
