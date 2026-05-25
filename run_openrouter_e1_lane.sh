#!/usr/bin/env bash
set -euo pipefail
set -x

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <display-name> <openrouter-model-id> <seed> <workers>" >&2
  echo "optional env: NUM_AGENTS=100 EPISODE_LENGTH=240" >&2
  exit 2
fi

cd "$(dirname "$0")"

set -a
source .env
set +a

display="$1"
model="$2"
seed="$3"
workers="$4"
num_agents="${NUM_AGENTS:-100}"
episode_length="${EPISODE_LENGTH:-240}"
api_access_date="$(date -u +%Y-%m-%d)"
code_commit="$(git rev-parse HEAD)"

run_one() {
  local setting="$1"
  local variant="$2"
  local gap="$3"
  local tag="$4"

  local safe="${model//\//_}"
  safe="${safe//:/_}"
  local src="data/thirdparty-${safe}-${tag}-seed${seed}-${num_agents}agents-${episode_length}months"

  echo "[$(date)] START E1 ${display} ${setting}/${variant} seed=${seed} workers=${workers}"

  PYTHONUNBUFFERED=1 python simulate.py \
    --provider thirdparty \
    --model "$model" \
    --api_base "${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}" \
    --num_agents "$num_agents" \
    --episode_length "$episode_length" \
    --gap_fixes "$gap" \
    --workers "$workers" \
    --seed "$seed" \
    --tag "$tag" \
    --exp_id E1 \
    --setting "$setting" \
    --variant "$variant" \
    --temperature 0.2 \
    --top_p 1.0

  if [[ ! -f "$src/summary.json" ]]; then
    echo "MISSING summary.json: $src" >&2
    exit 2
  fi

  python export_emnlp_outputs.py "$src" \
    --runs-root runs \
    --exp-id E1 \
    --model "$display" \
    --setting "$setting" \
    --variant "$variant" \
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

  echo "[$(date)] DONE E1 ${display} ${setting}/${variant} seed=${seed}"
}

run_one "text-only" "baseline" "False" "E1_baseline"
run_one "finevo" "default" "True" "E1_finevo"
