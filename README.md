# FinEvo EMNLP Experiment Toolkit

Experiment runner and output normalizer for the FinEvo EMNLP revision.
Current results are preliminary until regenerated from the ignored `runs/`
artifact tree with the documented seed set.

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
├── llm_providers.py     # Multi-model LLM interface
├── memory_module.py     # Episodic and semantic memory
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
