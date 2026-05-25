"""
EconAgent Simulation - ECCV 2026

Multi-LLM Economic Agent Simulation with GAP Fixes

Features:
- Multi-model support (OpenAI, Gemini, Local MLX/vLLM)
- Dual-track memory system (GAP 3 fix)
- Market sentiment module (GAP 1 & 2 fix)
- Strategy evolution with reflection (GAP 4 fix)

Usage:
    python simulate.py --model=gpt-5.2 --provider=openai --num_agents=100 --episode_length=240
    python simulate.py --model=mlx-community/Llama-3.3-70B-Instruct-4bit --provider=local --local_url=http://localhost:8002/v1
"""

import argparse
import csv
import fire
import hashlib
import os
import sys
import json
import random as py_random
import re
import pickle as pkl
import numpy as np
import yaml
from time import time
from collections import defaultdict, deque
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, List, Optional

import ai_economist.foundation as foundation

from llm_providers import (
    create_llm_provider, MultiModelLLM,
    OpenAIProvider, GeminiProvider, OllamaProvider, LocalAPIProvider, ThirdPartyProvider
)
from memory_module import DualTrackMemory
from market_sentiment import MarketSentiment
from utils import prettify_document, format_numbers, format_percentages, BRACKETS, WORLD_START_TIME

# Load config
with open('config.yaml', "r") as f:
    run_configuration = yaml.safe_load(f)
env_config = run_configuration.get('env')

# API Keys from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')


EVENT_FIELDS = [
    "month",
    "seed",
    "event_id",
    "event_text",
    "event_type",
    "source_macro_state",
    "sentiment_label",
    "v_t",
    "classifier_name",
    "classifier_confidence",
    "shuffled_from_month",
]

EVENT_LIBRARY = [
    ("credit_spread_widening", "Credit spreads widened as households became more cautious about near-term income.", -0.35, "negative"),
    ("hiring_plan_cut", "Several employers signaled slower hiring plans for the next quarter.", -0.30, "negative"),
    ("inflation_pressure", "Essential-goods prices were reported to be rising faster than expected.", -0.25, "negative"),
    ("consumer_confidence_drop", "A consumer survey showed weaker confidence and more precautionary saving.", -0.20, "negative"),
    ("steady_policy", "Policy rates and household credit conditions were broadly unchanged.", 0.00, "neutral"),
    ("wage_growth", "Wage growth improved modestly in service-sector jobs.", 0.20, "positive"),
    ("demand_recovery", "Retail demand recovered after several slow months.", 0.25, "positive"),
    ("employment_gain", "New job postings increased and layoffs remained limited.", 0.30, "positive"),
]

RANDOM_EVENT_LIBRARY = [
    "A finance newsletter compared the naming conventions of several market indexes.",
    "A bank announced a redesign of its mobile-app dashboard.",
    "A university seminar discussed historical approaches to accounting education.",
    "A trade magazine profiled office software used by payroll departments.",
    "A public dataset added metadata fields for archived economic reports.",
]


def _safe_slug(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace(" ", "_")


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _gini(values: List[float]) -> float:
    wealth = sorted(max(float(value), 0.0) for value in values)
    total = sum(wealth)
    n = len(wealth)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((idx + 1) * value for idx, value in enumerate(wealth))
    return 2 * weighted / (n * total) - (n + 1) / n


def build_event_schedule(event_variant: str, episode_length: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    """Build deterministic monthly event controls for EMNLP E4."""
    variant = (event_variant or "numeric-only").replace("_", "-")
    if variant == "none":
        variant = "numeric-only"

    base = []
    for month in range(episode_length):
        event_id, text, score, label = EVENT_LIBRARY[month % len(EVENT_LIBRARY)]
        base.append({
            "month": month,
            "seed": seed if seed is not None else "",
            "event_id": f"{event_id}_{month:03d}",
            "event_text": text,
            "event_type": "text-event",
            "source_macro_state": f"synthetic_cycle_phase_{month % len(EVENT_LIBRARY)}",
            "sentiment_label": label,
            "v_t": score,
            "classifier_name": "rule_based_event_fixture_v1",
            "classifier_confidence": 1.0,
            "shuffled_from_month": "",
        })

    if variant == "numeric-only":
        return [{
            "month": month,
            "seed": seed if seed is not None else "",
            "event_id": "",
            "event_text": "",
            "event_type": "numeric-only",
            "source_macro_state": "",
            "sentiment_label": "none",
            "v_t": 0.0,
            "classifier_name": "",
            "classifier_confidence": "",
            "shuffled_from_month": "",
        } for month in range(episode_length)]

    if variant == "text-event":
        return base

    if variant == "shuffled-event":
        rng = py_random.Random(seed if seed is not None else 0)
        order = list(range(episode_length))
        rng.shuffle(order)
        shuffled = []
        for month, source_month in enumerate(order):
            event = dict(base[source_month])
            event["month"] = month
            event["event_type"] = "shuffled-event"
            event["shuffled_from_month"] = source_month
            shuffled.append(event)
        return shuffled

    if variant == "random-event":
        rng = py_random.Random(seed if seed is not None else 0)
        return [{
            "month": month,
            "seed": seed if seed is not None else "",
            "event_id": f"random_irrelevant_{month:03d}",
            "event_text": rng.choice(RANDOM_EVENT_LIBRARY),
            "event_type": "random-event",
            "source_macro_state": "macro_irrelevant_fixture",
            "sentiment_label": "neutral",
            "v_t": 0.0,
            "classifier_name": "rule_based_event_fixture_v1",
            "classifier_confidence": 1.0,
            "shuffled_from_month": "",
        } for month in range(episode_length)]

    raise ValueError(f"Unknown event_variant: {event_variant}")


def agent_decision(
    env,
    obs,
    dialog_queue,
    dialog4ref_queue,
    memory_systems,
    sentiment_module,
    llm: MultiModelLLM,
    save_path,
    error_count,
    total_cost,
    use_gap_fixes: bool = True,
    use_sentiment: Optional[bool] = None,
    use_episodic: Optional[bool] = None,
    use_semantic: Optional[bool] = None,
    use_reflection: Optional[bool] = None,
    temperature: float = 0.0,
    prompt_style: str = "default",
    output_format: str = "json",
    event_entry: Optional[Dict[str, Any]] = None,
    action_log: Optional[List[Dict[str, Any]]] = None,
    api_error_log: Optional[List[Dict[str, Any]]] = None,
    memory_retrieval_log: Optional[List[Dict[str, Any]]] = None,
    semantic_rule_log: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    model_display: str = "",
    setting_label: str = "",
    variant_label: str = "",
    retrieval_k: int = 5,
    rule_budget: int = 3,
):
    """
    LLM-based Agent Decision Function with GAP Fixes

    Args:
        env: Environment instance
        obs: Current observation
        dialog_queue: Dialog history queue
        dialog4ref_queue: Reflection dialog queue
        memory_systems: Dict of DualTrackMemory instances
        sentiment_module: MarketSentiment instance
        llm: MultiModelLLM instance
        save_path: Path for saving dialogs
        error_count: Running error count
        total_cost: Running total cost
        use_gap_fixes: Whether to use GAP fixes

    Returns:
        actions: Agent actions dict
        error_count: Updated error count
        total_cost: Updated total cost
    """
    if use_sentiment is None:
        use_sentiment = use_gap_fixes
    if use_episodic is None:
        use_episodic = use_gap_fixes
    if use_semantic is None:
        use_semantic = use_gap_fixes
    if use_reflection is None:
        use_reflection = use_gap_fixes

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    curr_rates = obs['p']['PeriodicBracketTax-curr_rates']
    current_time = WORLD_START_TIME + relativedelta(months=env.world.timestep)
    current_time_str = current_time.strftime('%Y.%m')

    # Build economic state for memory retrieval
    year = (env.world.timestep - 1) // env.world.period
    if 0 <= year < len(env.world.unemployment):
        unemployment_rate = env.world.unemployment[year] / env.world.period / env.world.n_agents
    else:
        unemployment_rate = 0.05
    current_economic_state = {
        'timestamp': env.world.timestep,
        'price': env.world.price[-1],
        'interest_rate': env.world.interest_rate[-1],
        'unemployment_rate': unemployment_rate,
        'inflation': env.world.inflation[-1] if env.world.inflation else 0,
        'sentiment': sentiment_module.current_sentiment if use_sentiment else 0,
    }

    # Update sentiment module (GAP 1 & 2)
    if use_sentiment and env.world.timestep > 0:
        sentiment_module.update(
            env.world.price,
            env.world.unemployment,
            env.world.real_gdp_inflation,
            env.world.timestep,
            env.world.period,
        )
        if event_entry and event_entry.get("event_type") in {"text-event", "shuffled-event"}:
            sentiment_module.current_sentiment = float(np.clip(
                sentiment_module.current_sentiment + float(event_entry.get("v_t", 0.0)),
                -1,
                1,
            ))

    all_prompts = []
    prompt_hashes = []
    retrieval_traces: Dict[int, Dict[str, Any]] = {}

    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        skill = this_agent.state['skill']
        wealth = this_agent.inventory['Coin']
        consumption = this_agent.consumption['Coin']
        interest_rate = env.world.interest_rate[-1]
        price = env.world.price[-1]
        tax_paid = obs['p'][f'p{idx}']['PeriodicBracketTax-tax_paid']
        lump_sum = obs['p'][f'p{idx}']['PeriodicBracketTax-lump_sum']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        name = this_agent.endogenous['name']
        age = this_agent.endogenous['age']
        city = this_agent.endogenous['city']
        job = this_agent.endogenous['job']
        offer = this_agent.endogenous['offer']
        actions = env.dense_log['actions']
        states = env.dense_log['states']

        # Memory prompt (GAP 3)
        memory_prompt = ""
        if use_episodic or use_semantic:
            memory_system = memory_systems.get(idx)
            if memory_system:
                current_personal_state = {
                    **current_economic_state,
                    'wealth': wealth,
                    'income': this_agent.income.get('Coin', 0),
                    'employed': job != 'Unemployment',
                }
                memory_prompt, trace = memory_system.generate_memory_prompt(
                    current_personal_state,
                    retrieval_k=retrieval_k,
                    rule_budget=rule_budget,
                    include_episodic=bool(use_episodic),
                    include_semantic=bool(use_semantic),
                    return_trace=True,
                )
                retrieval_traces[idx] = trace

        # Sentiment prompt (GAP 2)
        sentiment_prompt = ""
        if use_sentiment:
            sentiment_prompt = sentiment_module.get_sentiment_prompt()

        # Build prompts
        problem_prompt = f'''You're {name}, a {age}-year-old individual living in {city}. A portion of your monthly income is taxed by the federal government through a tiered system, with tax revenue redistributed equally to all citizens. Now it's {current_time_str}.'''

        if job == 'Unemployment':
            job_prompt = f'''In the previous month, you were unemployed with no income. Now, you're invited to work as a(an) {offer} with monthly salary of ${skill*max_l:.2f}.'''
        else:
            if skill >= states[-1][str(idx)]['skill'] if states else skill:
                job_prompt = f'''In the previous month, you worked as a(an) {job}. If you continue working, your expected income will be ${skill*max_l:.2f} (increased due to labor market inflation).'''
            else:
                job_prompt = f'''In the previous month, you worked as a(an) {job}. If you continue working, your expected income will be ${skill*max_l:.2f} (decreased due to labor market deflation).'''

        if (consumption <= 0) and (len(actions) > 0) and (actions[-1].get('SimpleConsumption', 0) > 0):
            consumption_prompt = f'Your consumption was $0 due to goods shortage.'
        else:
            consumption_prompt = f'Your consumption was ${consumption:.2f}.'

        tax_prompt = f'''Tax deducted: ${tax_paid:.2f}. Redistribution received: ${lump_sum:.2f}. Tax brackets: {format_numbers(BRACKETS)}, rates: {format_numbers(curr_rates)}.'''

        if env.world.timestep == 0:
            price_prompt = f'Essential goods price: ${price:.2f}.'
        else:
            if price >= env.world.price[-2]:
                price_prompt = f'Inflation raised essential goods price to ${price:.2f}.'
            else:
                price_prompt = f'Deflation lowered essential goods price to ${price:.2f}.'

        # Combine all prompts
        full_prompt = f'''{prettify_document(problem_prompt)} {prettify_document(job_prompt)} {prettify_document(consumption_prompt)} {prettify_document(tax_prompt)} {prettify_document(price_prompt)}'''

        if use_sentiment and sentiment_prompt:
            full_prompt += f" {prettify_document(sentiment_prompt)}"

        if event_entry and event_entry.get("event_text"):
            full_prompt += f" Monthly market event: {prettify_document(event_entry['event_text'])}"

        full_prompt += f''' Your savings: ${wealth:.2f}. Interest rate: {interest_rate*100:.2f}%.'''

        if (use_episodic or use_semantic) and memory_prompt:
            full_prompt += f" {memory_prompt}"

        # Output format instruction (GAP 4: includes reflection)
        if prompt_style == "concise":
            full_prompt += " Keep reasoning concise and focus on current budget constraints."
        elif prompt_style == "risk_neutral":
            full_prompt += " Use a risk-neutral planning frame and maximize expected financial stability."
        elif prompt_style == "no_fairness":
            full_prompt += " Focus only on your own budget, prices, job prospects, and savings."

        if use_reflection and output_format == "json":
            full_prompt += ''' IMPORTANT: Respond with ONLY a JSON object containing: 1) "reflection": brief strategy thought (1 sentence), 2) "work": propensity 0-1 (intervals of 0.02), 3) "consumption": rate 0-1 (intervals of 0.02). Example: {"reflection": "High inflation means I should save more.", "work": 0.9, "consumption": 0.3}'''
        elif output_format == "natural_json":
            full_prompt += ''' You may give one brief sentence, but end with a JSON object containing "work" and "consumption" values in [0, 1].'''
        else:
            full_prompt += ''' IMPORTANT: Respond with ONLY a JSON object with "work" (0-1, intervals of 0.02) and "consumption" (0-1, intervals of 0.02). Example: {"work": 0.8, "consumption": 0.3}'''

        full_prompt = prettify_document(full_prompt)
        dialog_queue[idx].append({'role': 'user', 'content': full_prompt})
        dialog4ref_queue[idx].append({'role': 'user', 'content': full_prompt})
        all_prompts.append(list(dialog_queue[idx]))
        prompt_hashes.append(hashlib.sha256(json.dumps(list(dialog_queue[idx]), sort_keys=True).encode()).hexdigest())

    # Call LLM
    if env.world.timestep % 3 == 0 and env.world.timestep > 0:
        dialogs_to_send = [
            list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:]
            for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)
        ]
    else:
        dialogs_to_send = [list(dialogs) for dialogs in dialog_queue]

    results, cost = llm.get_multiple_completions(dialogs_to_send, temperature=temperature, max_tokens=800)
    total_cost += cost

    # Parse responses
    actions = {}
    reflections = {}

    for idx in range(env.num_agents):
        content = results[idx]
        this_agent = env.get_agent(str(idx))
        valid_json = True
        repair_attempts = 0
        clipped = False
        parsed = {}

        try:
            json_str = content
            # Strip <think>...</think> tags (Qwen3 thinking mode)
            if '<think>' in content:
                json_str = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            if '```json' in json_str:
                repair_attempts += 1
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                repair_attempts += 1
                json_str = json_str.split('```')[1].split('```')[0].strip()
            elif '{' in json_str:
                repair_attempts += 1
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                json_str = json_str[start:end]

            parsed = json.loads(json_str)

            reflection = parsed.get('reflection', '')
            reflections[idx] = reflection

            work_val = parsed.get('work', 1)
            consumption_val = parsed.get('consumption', 0.5)
            extracted_actions = [work_val, consumption_val]

            if not (0 <= work_val <= 1 and 0 <= consumption_val <= 1):
                extracted_actions = [1, 0.5]
                clipped = True
                error_count += 1

        except Exception as e:
            extracted_actions = [1, 0.5]
            reflections[idx] = ""
            valid_json = False
            error_count += 1

        if content == "Error" and api_error_log is not None:
            api_error_log.append({
                "month": env.world.timestep,
                "seed": seed if seed is not None else "",
                "model": model_display,
                "agent_id": idx,
                "error_type": "provider_error",
                "message": content,
            })

        work_propensity = float(extracted_actions[0])
        consumption_fraction = float(extracted_actions[1])

        if action_log is not None:
            trace = retrieval_traces.get(idx, {})
            action_log.append({
                "month": env.world.timestep,
                "seed": seed if seed is not None else "",
                "model": model_display,
                "setting": setting_label,
                "variant": variant_label,
                "agent_id": idx,
                "prompt_hash": prompt_hashes[idx] if idx < len(prompt_hashes) else "",
                "temperature": temperature,
                "top_p": 1.0,
                "raw_output": content,
                "parsed_action": {
                    "work": work_propensity,
                    "labor_hours": work_propensity * env._components_dict['SimpleLabor'].num_labor_hours,
                    "consumption_fraction": consumption_fraction,
                },
                "valid_json": valid_json,
                "repair_attempts": repair_attempts,
                "clipped": clipped,
                "rationale": parsed.get("reflection", "") if isinstance(parsed, dict) else "",
                "used_memory_ids": trace.get("retrieved_episode_ids", []) + trace.get("selected_rule_ids", []),
            })

        if memory_retrieval_log is not None and idx in retrieval_traces:
            trace = retrieval_traces[idx]
            memory_retrieval_log.append({
                "month": env.world.timestep,
                "seed": seed if seed is not None else "",
                "model": model_display,
                "agent_id": idx,
                "regime": "event" if event_entry and event_entry.get("event_text") else "numeric",
                "retrieved_episode_ids": trace.get("retrieved_episode_ids", []),
                "retrieval_scores": trace.get("retrieval_scores", []),
                "score_components": trace.get("score_components", []),
                "selected_rule_ids": trace.get("selected_rule_ids", []),
            })

        # Bernoulli sampling for work decision
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions

        # Update memory (GAP 3)
        if use_episodic or use_semantic:
            memory_system = memory_systems.get(idx)
            if memory_system and env.world.timestep > 0:
                prev_state = states[-1].get(str(idx), {}) if states else {}
                prev_wealth = prev_state.get('wealth', prev_state.get('inventory', {}).get('Coin', this_agent.inventory['Coin']))

                memory_system.add_episodic_memory(
                    timestamp=env.world.timestep,
                    economic_state=current_economic_state,
                    personal_state={
                        'wealth': this_agent.inventory['Coin'],
                        'income': this_agent.income.get('Coin', 0),
                        'employed': this_agent.endogenous['job'] != 'Unemployment',
                    },
                    decision={
                        'work': extracted_actions[0],
                        'consumption': extracted_actions[1] * 0.02,
                    },
                    outcome={
                        'wealth_change': this_agent.inventory['Coin'] - prev_wealth,
                    },
                    sentiment=sentiment_module.current_sentiment,
                    reflection=reflections.get(idx, ''),
                )

                # Quarterly consolidation
                if use_semantic and (env.world.timestep + 1) % 3 == 0:
                    memory_system.consolidate_to_semantic(env.world.timestep)
                    if semantic_rule_log is not None:
                        for rule in memory_system.semantic_memories.values():
                            semantic_rule_log.append({
                                "month": env.world.timestep,
                                "seed": seed if seed is not None else "",
                                "model": model_display,
                                "agent_id": idx,
                                "rule_id": rule.rule_id,
                                "condition": rule.condition,
                                "action_guidance": rule.strategy,
                                "rationale": "",
                                "regime": "numeric",
                                "confidence": rule.confidence,
                                "source_episode_ids": [f"E{episode}" for episode in rule.source_episodes],
                                "coded_category": [],
                                "validity_note": "supported_by_episode",
                            })

        dialog_queue[idx].append({'role': 'assistant', 'content': content})
        dialog4ref_queue[idx].append({'role': 'assistant', 'content': content})

    actions['p'] = [0]

    # Save dialog logs
    for idx, agent_dialog in enumerate(dialog_queue):
        with open(f'''{save_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
            for dialog in list(agent_dialog)[-2:]:
                f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')

    # Quarterly reflection
    if use_reflection and (env.world.timestep + 1) % 3 == 0:
        reflection_prompt = '''Given the previous quarter's economic environment, reflect on labor, consumption, and financial markets. What conclusions have you drawn? (Less than 200 words)'''
        reflection_prompt = prettify_document(reflection_prompt)

        for idx in range(env.num_agents):
            dialog4ref_queue[idx].append({'role': 'user', 'content': reflection_prompt})

        results, cost = llm.get_multiple_completions(
            [list(dialogs) for dialogs in dialog4ref_queue],
            temperature=temperature, max_tokens=200
        )
        total_cost += cost

        for idx in range(env.num_agents):
            dialog4ref_queue[idx].append({'role': 'assistant', 'content': results[idx]})

        for idx, agent_dialog in enumerate(dialog4ref_queue):
            with open(f'''{save_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
                for dialog in list(agent_dialog)[-2:]:
                    f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')

    return actions, error_count, total_cost


def run_experiment(
    model_name: str = "gpt-4o",
    provider_type: str = "openai",
    num_agents: int = 100,
    episode_length: int = 240,
    dialog_len: int = 3,
    use_gap_fixes: bool = True,
    max_price_inflation: float = 0.1,
    max_wage_inflation: float = 0.05,
    num_workers: int = 10,
    local_base_url: str = "http://localhost:8000/v1",
    api_key: str = "",
    api_base: str = "",
    seed: Optional[int] = None,
    tag: str = "",
    exp_id: str = "",
    setting: str = "",
    variant: str = "",
    use_sentiment: Optional[bool] = None,
    use_episodic: Optional[bool] = None,
    use_semantic: Optional[bool] = None,
    use_reflection: Optional[bool] = None,
    temperature: float = 0.0,
    prompt_style: str = "default",
    output_format: str = "json",
    event_variant: str = "numeric-only",
    retrieval_k: int = 5,
    rule_budget: int = 3,
    s_inertia: Optional[float] = None,
    s_inflation_sens: Optional[float] = None,
    s_unemp_sens: Optional[float] = None,
    s_gdp_sens: Optional[float] = None,
    s_no_macro: bool = False,
    s_random: bool = False,
):
    """
    Run experiment with specified model and settings

    Args:
        model_name: Model to use (e.g., gpt-5.2, gemini-3-pro-preview, mlx-community/Llama-3.3-70B-Instruct-4bit)
        provider_type: Provider type - "openai", "gemini", "ollama", or "local"
        num_agents: Number of agents in simulation
        episode_length: Simulation length in months
        dialog_len: Dialog history length
        use_gap_fixes: Whether to use GAP fixes (memory, sentiment, reflection)
        max_price_inflation: Maximum price inflation rate
        max_wage_inflation: Maximum wage inflation rate
        num_workers: Number of parallel workers for LLM calls
        local_base_url: Base URL for local API server (MLX, vLLM, LM Studio)

    Returns:
        summary: Experiment summary dict
    """
    print(f"\n{'='*60}")
    print(f"EconAgent Simulation - ECCV 2026")
    print(f"{'='*60}")
    print(f"Model: {provider_type}/{model_name}")
    print(f"GAP Fixes: {'Enabled' if use_gap_fixes else 'Disabled'}")
    print(f"Agents: {num_agents}, Episodes: {episode_length}")
    if seed is not None:
        print(f"Seed: {seed}")
    if provider_type == "local":
        print(f"Local API: {local_base_url}")
    print(f"{'='*60}\n")

    if seed is not None:
        np.random.seed(seed)
        py_random.seed(seed)

    if use_sentiment is None:
        use_sentiment = use_gap_fixes
    if use_episodic is None:
        use_episodic = use_gap_fixes
    if use_semantic is None:
        use_semantic = use_gap_fixes
    if use_reflection is None:
        use_reflection = use_gap_fixes

    # Create LLM provider
    if provider_type == "openai":
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        provider = OpenAIProvider(api_key=key, model=model_name)
    elif provider_type == "gemini":
        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        provider = GeminiProvider(api_key=key, model=model_name)
    elif provider_type == "thirdparty":
        key = api_key or OPENROUTER_API_KEY
        if not key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        provider = ThirdPartyProvider(
            api_key=key,
            model=model_name,
            base_url=api_base or OPENROUTER_BASE_URL,
        )
    elif provider_type == "ollama":
        provider = OllamaProvider(model=model_name)
    elif provider_type == "local":
        provider = LocalAPIProvider(model=model_name, base_url=local_base_url)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")

    llm = MultiModelLLM(provider, num_workers=num_workers)

    # Setup environment config
    env_config['n_agents'] = num_agents
    env_config['episode_length'] = episode_length
    env_config['flatten_masks'] = False
    env_config['flatten_observations'] = False
    env_config['components'][0]['SimpleLabor']['scale_obs'] = False
    env_config['components'][1]['PeriodicBracketTax']['scale_obs'] = False
    env_config['components'][3]['SimpleSaving']['scale_obs'] = False
    env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
    env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation

    # Initialize
    total_cost = 0
    error_count = 0
    dialog_queue = [deque(maxlen=dialog_len) for _ in range(num_agents)]
    dialog4ref_queue = [deque(maxlen=7) for _ in range(num_agents)]

    # Memory systems (GAP 3)
    memory_systems = {}
    if use_episodic or use_semantic:
        memory_systems = {
            idx: DualTrackMemory(agent_id=idx, episodic_capacity=24, semantic_capacity=10)
            for idx in range(num_agents)
        }

    # Sentiment module (GAP 1 & 2)
    sentiment_params = {
        "sentiment_inertia": 0.7 if s_inertia is None else s_inertia,
        "inflation_sensitivity": 0.15 if s_inflation_sens is None else s_inflation_sens,
        "unemployment_sensitivity": -0.2 if s_unemp_sens is None else s_unemp_sens,
        "gdp_sensitivity": 0.1 if s_gdp_sens is None else s_gdp_sens,
        "news_shock_std": 0.05,
        "sentiment_diffusion": 0.3,
    }
    if s_no_macro:
        sentiment_params["inflation_sensitivity"] = 0.0
        sentiment_params["unemployment_sensitivity"] = 0.0
        sentiment_params["gdp_sensitivity"] = 0.0
    if s_random:
        sentiment_params["sentiment_inertia"] = 0.0
        sentiment_params["inflation_sensitivity"] = 0.0
        sentiment_params["unemployment_sensitivity"] = 0.0
        sentiment_params["gdp_sensitivity"] = 0.0
        sentiment_params["news_shock_std"] = 0.35
    sentiment_module = MarketSentiment(n_agents=num_agents, **sentiment_params)
    sentiment_module.reset()
    event_schedule = build_event_schedule(event_variant, episode_length, seed)

    # Create environment
    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()

    # Setup save paths
    gap_suffix = "-gap_fixed" if use_gap_fixes else "-baseline"
    model_safe_name = _safe_slug(model_name)
    if tag:
        seed_suffix = f"-seed{seed}" if seed is not None else ""
        experiment_name = f'{provider_type}-{model_safe_name}-{tag}{seed_suffix}-{num_agents}agents-{episode_length}months'
    else:
        experiment_name = f'{provider_type}-{model_safe_name}{gap_suffix}-{num_agents}agents-{episode_length}months'

    save_path = './'
    if not os.path.exists(f'{save_path}data/{experiment_name}'):
        os.makedirs(f'{save_path}data/{experiment_name}')
    if not os.path.exists(f'{save_path}figs/{experiment_name}'):
        os.makedirs(f'{save_path}figs/{experiment_name}')

    action_log: List[Dict[str, Any]] = []
    api_error_log: List[Dict[str, Any]] = []
    memory_retrieval_log: List[Dict[str, Any]] = []
    semantic_rule_log: List[Dict[str, Any]] = []
    start_time = time()

    # Run simulation
    for step in range(episode_length):
        event_entry = event_schedule[step] if step < len(event_schedule) else None
        actions, error_count, total_cost = agent_decision(
            env, obs, dialog_queue, dialog4ref_queue,
            memory_systems, sentiment_module, llm,
            f'{save_path}data/{experiment_name}/dialogs',
            error_count, total_cost,
            use_gap_fixes=use_gap_fixes,
            use_sentiment=use_sentiment,
            use_episodic=use_episodic,
            use_semantic=use_semantic,
            use_reflection=use_reflection,
            temperature=temperature,
            prompt_style=prompt_style,
            output_format=output_format,
            event_entry=event_entry,
            action_log=action_log,
            api_error_log=api_error_log,
            memory_retrieval_log=memory_retrieval_log,
            semantic_rule_log=semantic_rule_log,
            seed=seed,
            model_display=model_name,
            setting_label=setting or ("finevo" if use_gap_fixes else "text-only"),
            variant_label=variant or ("default" if use_gap_fixes else "baseline"),
            retrieval_k=retrieval_k,
            rule_budget=rule_budget,
        )

        obs, rew, done, info = env.step(actions)

        if (step + 1) % 3 == 0:
            elapsed = time() - t
            print(f'Step {step+1}/{episode_length} done, {elapsed:.1f}s elapsed')
            print(f'  Errors: {error_count}, Cost: ${total_cost:.2f}')
            if use_gap_fixes:
                print(f'  Sentiment: {sentiment_module.current_sentiment:.3f}')
            t = time()

        # Save checkpoints
        if (step + 1) % 6 == 0 or step + 1 == episode_length:
            with open(f'{save_path}data/{experiment_name}/env_{step+1}.pkl', 'wb') as f:
                pkl.dump(env, f)
            with open(f'{save_path}data/{experiment_name}/dense_log_{step+1}.pkl', 'wb') as f:
                pkl.dump(env.dense_log, f)
            if memory_systems:
                with open(f'{save_path}data/{experiment_name}/memory_{step+1}.pkl', 'wb') as f:
                    pkl.dump(memory_systems, f)

    # Save final results
    with open(f'{save_path}data/{experiment_name}/dense_log.pkl', 'wb') as f:
        pkl.dump(env.dense_log, f)
    if memory_systems:
        with open(f'{save_path}data/{experiment_name}/memory_final.pkl', 'wb') as f:
            pkl.dump(memory_systems, f)

    _write_jsonl(f'{save_path}data/{experiment_name}/actions.jsonl', action_log)
    _write_jsonl(f'{save_path}data/{experiment_name}/api_errors.jsonl', api_error_log)
    _write_jsonl(f'{save_path}data/{experiment_name}/memory_retrieval.jsonl', memory_retrieval_log)
    _write_jsonl(f'{save_path}data/{experiment_name}/semantic_rules.jsonl', semantic_rule_log)
    _write_csv(f'{save_path}data/{experiment_name}/event_log.csv', event_schedule, EVENT_FIELDS)

    # Calculate final metrics
    final_wealths = [env.get_agent(str(i)).inventory['Coin'] for i in range(num_agents)]
    wall_time_min = (time() - start_time) / 60
    api_error_rate = len(api_error_log) / max(1, episode_length * num_agents)

    summary = {
        "model": f"{provider_type}/{model_name}",
        "gap_fixes": use_gap_fixes,
        "modules": {
            "sentiment": bool(use_sentiment),
            "episodic_memory": bool(use_episodic),
            "semantic_memory": bool(use_semantic),
            "reflection": bool(use_reflection),
        },
        "random_seed": seed if seed is not None else "",
        "ablation_tag": tag,
        "exp_id": exp_id,
        "setting": setting or ("finevo" if use_gap_fixes else "text-only"),
        "variant": variant or ("default" if use_gap_fixes else "baseline"),
        "temperature": temperature,
        "prompt_style": prompt_style,
        "output_format": output_format,
        "event_variant": event_variant,
        "sentiment_params": sentiment_params,
        "event_log": event_schedule,
        "num_agents": num_agents,
        "episode_length": episode_length,
        "total_cost": total_cost,
        "error_rate": error_count / (episode_length * num_agents),
        "api_error_rate": api_error_rate,
        "wall_time_min": wall_time_min,
        "final_metrics": {
            "avg_wealth": float(np.mean(final_wealths)),
            "median_wealth": float(np.median(final_wealths)),
            "std_wealth": float(np.std(final_wealths)),
            "min_wealth": float(np.min(final_wealths)),
            "max_wealth": float(np.max(final_wealths)),
            "gini": float(_gini(final_wealths)),
            "avg_unemployment": float(np.mean(env.world.unemployment) / env.world.period / num_agents),
            "avg_inflation": float(np.mean(env.world.inflation)) if env.world.inflation else 0,
            "sentiment_history": sentiment_module.sentiment_history if use_sentiment else [],
        }
    }

    with open(f'{save_path}data/{experiment_name}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment completed: {experiment_name}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Error rate: {error_count / (episode_length * num_agents) * 100:.2f}%")
    print(f"Avg wealth: ${np.mean(final_wealths):,.2f}")
    print(f"{'='*60}\n")

    return summary


def main(
    model: str = "gpt-4o",
    provider: str = "openai",
    num_agents: int = 100,
    episode_length: int = 240,
    gap_fixes: bool = True,
    workers: int = 10,
    local_url: str = "http://localhost:8000/v1",
    api_key: str = "",
    api_base: str = "",
    seed: Optional[int] = None,
    tag: str = "",
    exp_id: str = "",
    setting: str = "",
    variant: str = "",
    use_sentiment: Optional[bool] = None,
    use_episodic: Optional[bool] = None,
    use_semantic: Optional[bool] = None,
    use_reflection: Optional[bool] = None,
    temperature: float = 0.0,
    prompt_style: str = "default",
    output_format: str = "json",
    event_variant: str = "numeric-only",
    retrieval_k: int = 5,
    rule_budget: int = 3,
    s_inertia: Optional[float] = None,
    s_inflation_sens: Optional[float] = None,
    s_unemp_sens: Optional[float] = None,
    s_gdp_sens: Optional[float] = None,
    s_no_macro: bool = False,
    s_random: bool = False,
):
    """
    EconAgent Simulation - ECCV 2026

    Examples:
        # OpenAI GPT-5.2 with GAP fixes
        python simulate.py --model=gpt-5.2 --provider=openai --gap_fixes=True

        # OpenAI GPT-4o baseline (no GAP fixes)
        python simulate.py --model=gpt-4o --provider=openai --gap_fixes=False

        # Google Gemini 3 Pro
        python simulate.py --model=gemini-3-pro-preview --provider=gemini

        # Local Llama 3.3 70B via MLX
        python simulate.py --model=mlx-community/Llama-3.3-70B-Instruct-4bit --provider=local --local_url=http://localhost:8002/v1

        # Local Qwen3 32B via MLX
        python simulate.py --model=mlx-community/Qwen3-32B-4bit --provider=local --local_url=http://localhost:8001/v1
    """
    return run_experiment(
        model_name=model,
        provider_type=provider,
        num_agents=num_agents,
        episode_length=episode_length,
        use_gap_fixes=gap_fixes,
        num_workers=workers,
        local_base_url=local_url,
        api_key=api_key,
        api_base=api_base,
        seed=seed,
        tag=tag,
        exp_id=exp_id,
        setting=setting,
        variant=variant,
        use_sentiment=use_sentiment,
        use_episodic=use_episodic,
        use_semantic=use_semantic,
        use_reflection=use_reflection,
        temperature=temperature,
        prompt_style=prompt_style,
        output_format=output_format,
        event_variant=event_variant,
        retrieval_k=retrieval_k,
        rule_budget=rule_budget,
        s_inertia=s_inertia,
        s_inflation_sens=s_inflation_sens,
        s_unemp_sens=s_unemp_sens,
        s_gdp_sens=s_gdp_sens,
        s_no_macro=s_no_macro,
        s_random=s_random,
    )


if __name__ == "__main__":
    fire.Fire(main)
