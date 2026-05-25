#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

GEMINI_LOG="logs/e1_gemini3flash_seed13_openrouter.log"
QWEN_LOG="logs/e1_qwen3_seed13_openrouter.log"

echo "[$(date)] Guard started: wait for Gemini Step >= 24 before launching Qwen."

while true; do
  if grep -qE "Traceback|MISSING summary|429|rate limit|Third-party API error|Exception" "$GEMINI_LOG" 2>/dev/null; then
    echo "[$(date)] Gemini log has error. Do not start Qwen."
    grep -E "Traceback|MISSING summary|429|rate limit|Third-party API error|Exception" "$GEMINI_LOG" | tail -20
    exit 1
  fi

  latest_step="$(grep -oE "Step [0-9]+/240" "$GEMINI_LOG" 2>/dev/null | awk '{print $2}' | cut -d/ -f1 | tail -1)"
  latest_step="${latest_step:-0}"

  echo "[$(date)] Gemini latest step: ${latest_step}/240"

  if [ "$latest_step" -ge 24 ]; then
    echo "[$(date)] Gemini is stable enough. Launch Qwen seed13 E1."
    break
  fi

  if grep -q "Experiment completed" "$GEMINI_LOG" 2>/dev/null; then
    echo "[$(date)] Gemini baseline completed. Launch Qwen seed13 E1."
    break
  fi

  sleep 600
done

if pgrep -f "simulate.py.*qwen/qwen3-235b-a22b-2507|run_openrouter_e1_lane.sh Qwen3-235B" >/dev/null; then
  echo "[$(date)] Qwen lane already appears to be running. Exit without launching duplicate."
  exit 0
fi

./run_openrouter_e1_lane.sh "Qwen3-235B" "qwen/qwen3-235b-a22b-2507" 13 2 > "$QWEN_LOG" 2>&1

echo "[$(date)] Qwen seed13 lane finished."
