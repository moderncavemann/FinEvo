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
import fire
import os
import sys
import json
import re
import pickle as pkl
import numpy as np
import yaml
from time import time
from collections import defaultdict, deque
from dateutil.relativedelta import relativedelta
from typing import Optional

import ai_economist.foundation as foundation

from llm_providers import (
    create_llm_provider, MultiModelLLM,
    OpenAIProvider, GeminiProvider, OllamaProvider, LocalAPIProvider
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
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    curr_rates = obs['p']['PeriodicBracketTax-curr_rates']
    current_time = WORLD_START_TIME + relativedelta(months=env.world.timestep)
    current_time_str = current_time.strftime('%Y.%m')

    # Build economic state for memory retrieval
    year = (env.world.timestep - 1) // env.world.period
    current_economic_state = {
        'timestamp': env.world.timestep,
        'price': env.world.price[-1],
        'interest_rate': env.world.interest_rate[-1],
        'unemployment_rate': env.world.unemployment[year] / env.world.period / env.world.n_agents if year < len(env.world.unemployment) else 0.05,
        'inflation': env.world.inflation[-1] if env.world.inflation else 0,
        'sentiment': sentiment_module.current_sentiment if use_gap_fixes else 0,
    }

    # Update sentiment module (GAP 1 & 2)
    if use_gap_fixes and env.world.timestep > 0:
        sentiment_module.update(
            env.world.price,
            env.world.unemployment,
            env.world.real_gdp_inflation,
            env.world.timestep,
            env.world.period,
        )

    all_prompts = []

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
        if use_gap_fixes:
            memory_system = memory_systems.get(idx)
            if memory_system:
                current_personal_state = {
                    **current_economic_state,
                    'wealth': wealth,
                    'income': this_agent.income.get('Coin', 0),
                    'employed': job != 'Unemployment',
                }
                memory_prompt = memory_system.generate_memory_prompt(current_personal_state)

        # Sentiment prompt (GAP 2)
        sentiment_prompt = ""
        if use_gap_fixes:
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

        if use_gap_fixes and sentiment_prompt:
            full_prompt += f" {prettify_document(sentiment_prompt)}"

        full_prompt += f''' Your savings: ${wealth:.2f}. Interest rate: {interest_rate*100:.2f}%.'''

        if use_gap_fixes and memory_prompt:
            full_prompt += f" {memory_prompt}"

        # Output format instruction (GAP 4: includes reflection)
        if use_gap_fixes:
            full_prompt += ''' IMPORTANT: Respond with ONLY a JSON object containing: 1) "reflection": brief strategy thought (1 sentence), 2) "work": propensity 0-1 (intervals of 0.02), 3) "consumption": rate 0-1 (intervals of 0.02). Example: {"reflection": "High inflation means I should save more.", "work": 0.9, "consumption": 0.3}'''
        else:
            full_prompt += ''' IMPORTANT: Respond with ONLY a JSON object with "work" (0-1, intervals of 0.02) and "consumption" (0-1, intervals of 0.02). Example: {"work": 0.8, "consumption": 0.3}'''

        full_prompt = prettify_document(full_prompt)
        dialog_queue[idx].append({'role': 'user', 'content': full_prompt})
        dialog4ref_queue[idx].append({'role': 'user', 'content': full_prompt})
        all_prompts.append(list(dialog_queue[idx]))

    # Call LLM
    if env.world.timestep % 3 == 0 and env.world.timestep > 0:
        dialogs_to_send = [
            list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:]
            for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)
        ]
    else:
        dialogs_to_send = [list(dialogs) for dialogs in dialog_queue]

    results, cost = llm.get_multiple_completions(dialogs_to_send, temperature=0, max_tokens=800)
    total_cost += cost

    # Parse responses
    actions = {}
    reflections = {}

    for idx in range(env.num_agents):
        content = results[idx]
        this_agent = env.get_agent(str(idx))

        try:
            json_str = content
            # Strip <think>...</think> tags (Qwen3 thinking mode)
            if '<think>' in content:
                json_str = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()
            elif '{' in json_str:
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
                error_count += 1

        except Exception as e:
            extracted_actions = [1, 0.5]
            reflections[idx] = ""
            error_count += 1

        # Bernoulli sampling for work decision
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions

        # Update memory (GAP 3)
        if use_gap_fixes:
            memory_system = memory_systems.get(idx)
            if memory_system and env.world.timestep > 0:
                prev_wealth = states[-1][str(idx)].get('wealth', 0) if states else this_agent.inventory['Coin']

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
                if (env.world.timestep + 1) % 3 == 0:
                    memory_system.consolidate_to_semantic(env.world.timestep)

        dialog_queue[idx].append({'role': 'assistant', 'content': content})
        dialog4ref_queue[idx].append({'role': 'assistant', 'content': content})

    actions['p'] = [0]

    # Save dialog logs
    for idx, agent_dialog in enumerate(dialog_queue):
        with open(f'''{save_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
            for dialog in list(agent_dialog)[-2:]:
                f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')

    # Quarterly reflection
    if (env.world.timestep + 1) % 3 == 0:
        reflection_prompt = '''Given the previous quarter's economic environment, reflect on labor, consumption, and financial markets. What conclusions have you drawn? (Less than 200 words)'''
        reflection_prompt = prettify_document(reflection_prompt)

        for idx in range(env.num_agents):
            dialog4ref_queue[idx].append({'role': 'user', 'content': reflection_prompt})

        results, cost = llm.get_multiple_completions(
            [list(dialogs) for dialogs in dialog4ref_queue],
            temperature=0, max_tokens=200
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
    if provider_type == "local":
        print(f"Local API: {local_base_url}")
    print(f"{'='*60}\n")

    # Create LLM provider
    if provider_type == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        provider = OpenAIProvider(api_key=OPENAI_API_KEY, model=model_name)
    elif provider_type == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        provider = GeminiProvider(api_key=GEMINI_API_KEY, model=model_name)
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
    if use_gap_fixes:
        memory_systems = {
            idx: DualTrackMemory(agent_id=idx, episodic_capacity=24, semantic_capacity=10)
            for idx in range(num_agents)
        }

    # Sentiment module (GAP 1 & 2)
    sentiment_module = MarketSentiment(n_agents=num_agents)
    sentiment_module.reset()

    # Create environment
    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()

    # Setup save paths
    gap_suffix = "-gap_fixed" if use_gap_fixes else "-baseline"
    model_safe_name = model_name.replace(":", "_").replace("/", "_")
    experiment_name = f'{provider_type}-{model_safe_name}{gap_suffix}-{num_agents}agents-{episode_length}months'

    save_path = './'
    if not os.path.exists(f'{save_path}data/{experiment_name}'):
        os.makedirs(f'{save_path}data/{experiment_name}')
    if not os.path.exists(f'{save_path}figs/{experiment_name}'):
        os.makedirs(f'{save_path}figs/{experiment_name}')

    # Run simulation
    for step in range(episode_length):
        actions, error_count, total_cost = agent_decision(
            env, obs, dialog_queue, dialog4ref_queue,
            memory_systems, sentiment_module, llm,
            f'{save_path}data/{experiment_name}/dialogs',
            error_count, total_cost,
            use_gap_fixes=use_gap_fixes,
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

    # Save final results
    with open(f'{save_path}data/{experiment_name}/dense_log.pkl', 'wb') as f:
        pkl.dump(env.dense_log, f)

    # Calculate final metrics
    final_wealths = [env.get_agent(str(i)).inventory['Coin'] for i in range(num_agents)]

    summary = {
        "model": f"{provider_type}/{model_name}",
        "gap_fixes": use_gap_fixes,
        "num_agents": num_agents,
        "episode_length": episode_length,
        "total_cost": total_cost,
        "error_rate": error_count / (episode_length * num_agents),
        "final_metrics": {
            "avg_wealth": float(np.mean(final_wealths)),
            "median_wealth": float(np.median(final_wealths)),
            "std_wealth": float(np.std(final_wealths)),
            "min_wealth": float(np.min(final_wealths)),
            "max_wealth": float(np.max(final_wealths)),
            "avg_unemployment": float(np.mean(env.world.unemployment) / env.world.period / num_agents),
            "avg_inflation": float(np.mean(env.world.inflation)) if env.world.inflation else 0,
            "sentiment_history": sentiment_module.sentiment_history if use_gap_fixes else [],
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
    )


if __name__ == "__main__":
    fire.Fire(main)
