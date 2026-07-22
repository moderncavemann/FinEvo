# FinEvo EMNLP Experiment Toolkit

Experiment runner and output normalizer for the FinEvo EMNLP revision.
Current results are preliminary until regenerated from the ignored `runs/`
artifact tree with the documented seed set.

## Verified dual-track redesign

The post-review method is implemented as a separate, auditable path. The legacy
`simulate.py` and `memory_module.py` remain available for reproduction; new
method work uses:

- `verified_memory/m1_context.py`: causal context encoding with independent
  retrieval/prompt routes;
- `verified_memory/m2_episodic.py`: staged `action_t` and finalized
  `outcome_t+1` evidence;
- `verified_memory/m3_semantic.py`: LLM candidates with support,
  counterevidence, provisional activation, and retirement;
- `verified_memory/m0_utility.py`: evaluation-only cash-flow and flow-utility
  ledger;
- `simulate_verified.py`: direct-hours, hard-budget, bounded runner;
- `replay_verified.py`: hash-bound five-treatment paired replay.

Budgeted calls use one HTTP attempt per reserved call; post-setup execution
failures write a content-addressed error/config/budget receipt. Preflight
failures remain stderr-only. Partial in-memory simulation streams are not yet
checkpointed on failure. Effective Foundation parameters,
request seed, served model, system fingerprint, cache usage, and provider
request ID are retained when available.

Run the no-network integration diagnostic (synthetic fixture, not a result):

```bash
python simulate_verified.py \
  --provider diagnostic \
  --run-id local-g0 \
  --num-agents 2 \
  --episode-length 6
```

Run a limited API smoke only after local tests pass:

```bash
python simulate_verified.py \
  --provider openai \
  --model gpt-5.2-2025-12-11 \
  --run-id api-g4 \
  --num-agents 2 \
  --episode-length 6 \
  --max-calls 24 \
  --max-cost-usd 0.25
```

The runner blocks more than four agents or twelve months unless
`--allow-larger-run` is explicit. Passing a smoke establishes implementation
validity, not superiority. Current evidence and remaining gates are recorded in
`artifacts/verified_memory_smoke_report.md`.

The replay command currently accepts only runs with `context_to_prompt=false`;
it fails closed for prompt-only/full M1 routes until prompt-routed context can
be retained as a protected non-memory field.

## Features

- **Multi-model support**: OpenAI, OpenRouter-compatible APIs, Google Gemini, and local OpenAI-compatible servers
- **FinEvo components**: market sentiment, episodic memory, semantic rule memory, and reflection
- **Revision experiments**: E1-E5 matrices for main results, component ablations, sentiment robustness, text-event controls, and prompt/decoding robustness
- **Structured artifacts**: action logs, memory retrieval logs, semantic rule logs, event logs, summaries, trajectories, and seed-level configs

## Installation

```bash
pip install -r requirements.txt
```

Set API keys:
```bash
export OPENAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

## Usage

### OpenAI Models

```bash
# GPT-5.2 FinEvo run
python simulate.py --model=gpt-5.2 --provider=openai --num_agents=100 --episode_length=240 --gap_fixes=True

# GPT-4o text-only baseline
python simulate.py --model=gpt-4o --provider=openai --gap_fixes=False
```

### OpenRouter-Compatible Models

```bash
python simulate.py \
  --provider=thirdparty \
  --api_base="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}" \
  --model=google/gemini-3-flash-preview \
  --workers=3
```

### Local Models (MLX)

```bash
# Start MLX server
mlx_lm.server --model mlx-community/Llama-3.3-70B-Instruct-4bit --port 8002

# Run simulation
python simulate.py --model=mlx-community/Llama-3.3-70B-Instruct-4bit --provider=local --local_url=http://localhost:8002/v1
```

### EMNLP Revision Runs

Print the supported E2-E5 experiment matrix without spending API budget:

```bash
python run_emnlp_experiments.py --exps E2,E3,E4,E5 --seeds 13,21,42,87,2026
```

Execute a focused subset and export it to the required `runs/<exp_id>/<model>/<setting>/<variant>/seed_<seed>/` layout:

```bash
python run_emnlp_experiments.py --run --exps E2 --variants default --seeds 13
```

Existing legacy `data/<run>/summary.json` folders can be normalized directly:

```bash
python export_emnlp_outputs.py "data/<run-folder>" --runs-root runs --exp-id E2 --model GPT-5.2 --setting finevo --variant default --seed 13
```

Generate paper figures locally when result folders are present:

```bash
python paper/generate_emnlp_figures.py
```

Generated `data/`, `runs/`, `figs/`, logs, pickles, and zip artifacts are intentionally ignored by Git.

## Project Structure

```
eccv26_EconAgent/
├── simulate.py          # Main simulation script
├── simulate_verified.py # Bounded verified-memory runner
├── replay_verified.py   # Paired memory-intervention runner
├── llm_providers.py     # Multi-model LLM interface
├── memory_module.py     # Episodic and semantic memory
├── verified_memory/     # M0/M1/M2/M3, replay, budgets, provenance
├── market_sentiment.py  # Market sentiment dynamics
├── run_emnlp_experiments.py # EMNLP E1-E8 run matrix
├── export_emnlp_outputs.py  # Normalizes outputs to required schema
├── utils.py             # Utility functions
├── config.yaml          # Environment configuration
├── requirements.txt     # Dependencies
└── README.md
```

## License

MIT
