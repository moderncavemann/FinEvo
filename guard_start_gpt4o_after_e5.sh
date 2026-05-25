#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "[$(date)] Guard started: wait for E3=50, E4=20, E5=35 before launching GPT-4o E1 seed13."

while true; do
  e3="$(find runs/E3 -name metrics_summary.csv 2>/dev/null | wc -l | tr -d ' ')"
  e4="$(find runs/E4 -name metrics_summary.csv 2>/dev/null | wc -l | tr -d ' ')"
  e5="$(find runs/E5 -name metrics_summary.csv 2>/dev/null | wc -l | tr -d ' ')"

  echo "[$(date)] Counts: E3=${e3} E4=${e4} E5=${e5}"

  if grep -RniE "Traceback|OpenAI error|insufficient_quota|unexpected keyword|MISSING summary" logs/gpt52_E3_openai.log logs/gpt52_E4_openai.log logs/gpt52_E5_openai.log 2>/dev/null | tail -20 | grep -q .; then
    echo "[$(date)] Recent OpenAI-related error found. Do not start GPT-4o."
    grep -RniE "Traceback|OpenAI error|insufficient_quota|unexpected keyword|MISSING summary" logs/gpt52_E3_openai.log logs/gpt52_E4_openai.log logs/gpt52_E5_openai.log 2>/dev/null | tail -20
    exit 1
  fi

  if [ "$e3" -ge 50 ] && [ "$e4" -ge 20 ] && [ "$e5" -ge 35 ]; then
    echo "[$(date)] E3/E4/E5 complete. Launch GPT-4o seed13 E1."
    break
  fi

  sleep 900
done

if pgrep -f "simulate.py.*--provider openai.*--model gpt-4o|run_gpt4o_seed13_e1_once.sh" >/dev/null; then
  echo "[$(date)] GPT-4o lane already appears to be running. Exit without launching duplicate."
  exit 0
fi

./run_gpt4o_seed13_e1_once.sh > logs/e1_gpt4o_seed13_openai.log 2>&1

echo "[$(date)] GPT-4o seed13 E1 finished."
