#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a

run_one() {
  local setting="$1"
  local variant="$2"
  local gap="$3"
  local tag="$4"

  local src="data/openai-gpt-4o-${tag}-seed13-100agents-240months"

  echo "[$(date)] START GPT-4o ${setting}/${variant} seed13"

  PYTHONUNBUFFERED=1 python simulate.py \
    --provider openai \
    --model gpt-4o \
    --num_agents 100 \
    --episode_length 240 \
    --gap_fixes "$gap" \
    --workers 4 \
    --seed 13 \
    --tag "$tag" \
    --exp_id E1 \
    --setting "$setting" \
    --variant "$variant" \
    --temperature 0.2 \
    --top_p 1.0

  if [ ! -f "$src/summary.json" ]; then
    echo "MISSING summary.json: $src"
    exit 2
  fi

  python export_emnlp_outputs.py "$src" \
    --runs-root runs \
    --exp-id E1 \
    --model "GPT-4o" \
    --setting "$setting" \
    --variant "$variant" \
    --seed 13 \
    --temperature 0.2 \
    --top-p 1.0 \
    --max-tokens 800 \
    --api-access-date "$(date -u +%Y-%m-%d)" \
    --code-commit "$(git rev-parse HEAD)" \
    --prompt-version "finevo_emnlp_v1" \
    --reflection-prompt-version "finevo_reflection_v1" \
    --decision-max-tokens 800 \
    --reflection-max-tokens 200

  echo "[$(date)] DONE GPT-4o ${setting}/${variant} seed13"
}

run_one "text-only" "baseline" "False" "E1_baseline"
run_one "finevo" "default" "True" "E1_finevo"
