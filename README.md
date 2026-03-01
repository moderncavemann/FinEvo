# EconAgent - ECCV 2026

Multi-LLM Economic Agent Simulation with GAP Fixes

## Features

- **Multi-Model Support**: OpenAI (GPT-4o, GPT-5.2), Google Gemini, Local MLX/vLLM
- **GAP Fixes**: Memory module, market sentiment, strategy evolution
- **Scalable**: 10-100+ agents, 24-240+ months simulation

## Installation

```bash
pip install -r requirements.txt
```

Set API keys:
```bash
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

## Usage

### OpenAI Models

```bash
# GPT-5.2 with GAP fixes
python simulate.py --model=gpt-5.2 --provider=openai --num_agents=100 --episode_length=240 --gap_fixes=True

# GPT-4o baseline (no GAP fixes)
python simulate.py --model=gpt-4o --provider=openai --gap_fixes=False
```

### Google Gemini

```bash
python simulate.py --model=gemini-3-pro-preview --provider=gemini --workers=1
```

### Local Models (MLX)

```bash
# Start MLX server
mlx_lm.server --model mlx-community/Llama-3.3-70B-Instruct-4bit --port 8002

# Run simulation
python simulate.py --model=mlx-community/Llama-3.3-70B-Instruct-4bit --provider=local --local_url=http://localhost:8002/v1
```

## Project Structure

```
eccv26_EconAgent/
├── simulate.py          # Main simulation script
├── llm_providers.py     # Multi-model LLM interface
├── memory_module.py     # Dual-track memory (GAP 3)
├── market_sentiment.py  # Market sentiment (GAP 1 & 2)
├── utils.py             # Utility functions
├── config.yaml          # Environment configuration
├── requirements.txt     # Dependencies
└── README.md
```

## GAP Fixes

1. **GAP 1 & 2**: Market sentiment module with endogenous sentiment index
2. **GAP 3**: Dual-track memory (episodic + semantic)
3. **GAP 4**: Strategy evolution with quarterly reflection

## Results

| Model | Avg Wealth | Unemployment | Gini | Cost |
|-------|-----------|--------------|------|------|
| GPT-5.2 + GAP | $2,614,208 | 0.16% | 0.39 | $143 |
| Llama-3.3-70B | $2,297,338 | 1.29% | 0.51 | $0 |
| Llama-4-Maverick | $1,401,208 | 1.98% | 0.76 | $0 |
| GPT-5.2 Baseline | $428,417 | 15.61% | 0.40 | $94 |

## License

MIT
