#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <display-name> <openrouter-model-id> <workers>" >&2
  exit 2
fi

set -a
source .env
set +a

display="$1"
model="$2"
workers="$3"

run_one() {
  local seed="$1"
  local setting="$2"
  local variant="$3"
  local gap="$4"
  local tag="$5"

  local safe="${model//\//_}"
  safe="${safe//:/_}"
  local src="data/thirdparty-${safe}-${tag}-seed${seed}-100agents-240months"

  echo "[$(date)] START E1 ${display} ${setting}/${variant} seed=${seed} workers=${workers}"

  PYTHONUNBUFFERED=1 python simulate.py \
    --provider thirdparty \
    --model "$model" \
    --api_base "${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}" \
    --num_agents 100 \
    --episode_length 240 \
    --gap_fixes "$gap" \
    --workers "$workers" \
    --seed "$seed" \
    --tag "$tag" \
    --exp_id E1 \
    --setting "$setting" \
    --variant "$variant" \
    --temperature 0.2

  python export_emnlp_outputs.py "$src" \
    --runs-root runs \
    --exp-id E1 \
    --model "$display" \
    --setting "$setting" \
    --variant "$variant" \
    --seed "$seed" \
    --temperature 0.2 \
    --top-p 1.0 \
    --max-tokens 4000

  echo "[$(date)] DONE E1 ${display} ${setting}/${variant} seed=${seed}"
}

for seed in 13 21 42 87 2026; do
  run_one "$seed" "text-only" "baseline" "False" "E1_baseline"
  run_one "$seed" "finevo" "default" "True" "E1_finevo"
done
