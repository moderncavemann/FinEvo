#!/usr/bin/env bash
set -euo pipefail

# Run GPT-5.2 large-scale text-only baseline rows required for the matched
# significance table. This script intentionally runs seeds sequentially because
# each simulation already performs agent-level parallel API calls.

cd "$(dirname "$0")"

set -a
source .env
set +a

mkdir -p logs

seeds="${SEEDS:-13 21 42 87 2026}"
workers="${WORKERS:-8}"
api_access_date="$(date -u +%Y-%m-%d)"
code_commit="$(git rev-parse HEAD)"

for seed in $seeds; do
  src="data/openai-gpt-5.2-LS_baseline-seed${seed}-100agents-240months"
  echo "[$(date)] START GPT-5.2 text-only baseline seed=${seed} workers=${workers}"

  PYTHONUNBUFFERED=1 python simulate.py \
    --provider openai \
    --model gpt-5.2 \
    --num_agents 100 \
    --episode_length 240 \
    --gap_fixes False \
    --workers "$workers" \
    --seed "$seed" \
    --tag LS_baseline \
    --exp_id E1 \
    --setting text-only \
    --variant baseline \
    --temperature 0.2 \
    --top_p 1.0

  if [[ ! -f "$src/summary.json" ]]; then
    echo "MISSING summary.json: $src" >&2
    exit 2
  fi

  python export_emnlp_outputs.py "$src" \
    --runs-root runs \
    --exp-id E1 \
    --model GPT-5.2 \
    --setting text-only \
    --variant baseline \
    --seed "$seed" \
    --temperature 0.2 \
    --top-p 1.0 \
    --max-tokens 800 \
    --decision-max-tokens 800 \
    --reflection-max-tokens 200 \
    --api-access-date "$api_access_date" \
    --code-commit "$code_commit" \
    --prompt-version finevo_emnlp_v1 \
    --reflection-prompt-version finevo_reflection_v1

  echo "[$(date)] DONE GPT-5.2 text-only baseline seed=${seed}"
done
