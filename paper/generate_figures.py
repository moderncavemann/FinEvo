"""
FinEvo Paper Figure Generator
==============================
Generates all 7 figures for the ECCV paper.
Run from the ACL24-EconAgent directory:
    python paper/generate_figures.py

Output: figs/paper/*.pdf  and  figs/paper/*.png
"""

import pickle
import numpy as np
import os
import json
from math import pi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.6,
    'figure.dpi': 200,
})

# Line styles per model
STYLE = {
    "GPT-5.2 Baseline":               dict(color='#888888', ls='--', lw=1.5),
    "GPT-5.2 + FinEvo":               dict(color='#2196F3', ls='-',  lw=2.0),
    "Llama-4-Maverick-17B + FinEvo":  dict(color='#E91E63', ls='-',  lw=1.8),
    "Llama-3.3-70B + FinEvo":         dict(color='#FF9800', ls='-',  lw=1.8),
    "Qwen3-32B + FinEvo":             dict(color='#9C27B0', ls=':',  lw=1.5),
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def gini(x):
    x = np.sort(np.array([max(0, v) for v in x], dtype=float))
    n = len(x)
    if np.sum(x) == 0:
        return 0
    return (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def save(fig, name):
    os.makedirs("figs/paper", exist_ok=True)
    fig.savefig(f"figs/paper/{name}.pdf", bbox_inches='tight')
    fig.savefig(f"figs/paper/{name}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ figs/paper/{name}.pdf/.png")


# ── Load all experiment data ───────────────────────────────────────────────────
BASE = "data/"
EXPERIMENTS = [
    ("GPT-5.2 Baseline",              "openai-gpt-5.2-baseline-100agents-240months"),
    ("GPT-5.2 + FinEvo",              "openai-gpt-5.2-gap_fixed-100agents-240months"),
    ("Llama-4-Maverick-17B + FinEvo", "local-mlx-community_Llama-4-Maverick-17B-128E-Instruct-4bit-gap_fixed-100agents-240months"),
    ("Llama-3.3-70B + FinEvo",        "local-mlx-community_Llama-3.3-70B-Instruct-4bit-gap_fixed-100agents-240months"),
    ("Qwen3-32B + FinEvo",            "local-mlx-community_Qwen3-32B-4bit-gap_fixed-100agents-240months"),
]

print("Loading experiment data...")
data = {}
for name, folder in EXPERIMENTS:
    log_path = os.path.join(BASE, folder, "dense_log.pkl")
    sum_path = os.path.join(BASE, folder, "summary.json")
    with open(log_path, 'rb') as f:
        log = pickle.load(f)
    with open(sum_path) as f:
        summary = json.load(f)

    states   = log['states']
    world    = log['world']
    n_agents = len([k for k in states[0].keys() if k != 'p'])
    T        = len(states)

    wealth_ts  = [np.mean([states[t][str(i)]['inventory']['Coin']
                            for i in range(n_agents) if str(i) in states[t]])
                  for t in range(T)]
    gini_ts    = [gini([states[t][str(i)]['inventory']['Coin']
                        for i in range(n_agents) if str(i) in states[t]])
                  for t in range(T)]
    gdp_ts     = [w.get('Real GDP Growth', 0) * 100 for w in world]
    unemp_ts   = [w.get('Unemployment Rate', 0) * 100 for w in world]
    final_w    = [states[-1][str(i)]['inventory']['Coin']
                  for i in range(n_agents) if str(i) in states[-1]]
    sentiment  = summary['final_metrics'].get('sentiment_history', [0] * T)

    data[name] = dict(
        wealth_ts=wealth_ts, gini_ts=gini_ts,
        gdp_ts=gdp_ts, unemp_ts=unemp_ts,
        final_wealth=final_w, sentiment=sentiment,
        summary=summary,
    )
    print(f"  Loaded: {name}  (T={T}, agents={n_agents})")

months = np.arange(len(data["GPT-5.2 Baseline"]["wealth_ts"]))
print()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — 4-panel overview: wealth / GDP / unemployment / sentiment
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 1 — 4-panel overview...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax_w, ax_g, ax_u, ax_s = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

for name, d in data.items():
    st = STYLE[name]
    ax_w.plot(months, d['wealth_ts'], label=name, **st)
    ax_g.plot(months, d['gdp_ts'], **st)
    ax_u.plot(months, d['unemp_ts'], **st)
    if d['sentiment'] and len(d['sentiment']) == 240:
        ax_s.plot(np.arange(240), d['sentiment'], label=name, **st)

ax_w.set_title('(a) Average Agent Wealth over Time', fontweight='bold')
ax_w.set_ylabel('Avg. Wealth (Coin)'); ax_w.set_xlabel('Month')
ax_w.legend(fontsize=7.5)

ax_g.set_title('(b) Real GDP Growth Rate (%)', fontweight='bold')
ax_g.set_ylabel('GDP Growth (%/month)'); ax_g.set_xlabel('Month')
ax_g.axhline(0, color='k', lw=0.8)

ax_u.set_title('(c) Unemployment Rate (%)', fontweight='bold')
ax_u.set_ylabel('Unemployment Rate (%)'); ax_u.set_xlabel('Month')

ax_s.set_title('(d) Market Sentiment Index $S_t$ (FinEvo models only)', fontweight='bold')
ax_s.set_ylabel('Sentiment $S_t$ ∈ [−1, 1]'); ax_s.set_xlabel('Month')
ax_s.axhline(0, color='k', lw=0.8, ls=':')
ax_s.legend(fontsize=7.5)

plt.tight_layout()
save(fig, "fig1_overview_timeseries")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Lorenz curves
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 2 — Lorenz curves...")
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Equality')

for name, d in data.items():
    fw = np.sort(np.array([max(0, v) for v in d['final_wealth']]))
    n  = len(fw)
    cum_pop  = np.linspace(0, 1, n + 1)
    cum_wlth = np.concatenate([[0], np.cumsum(fw) / np.sum(fw)])
    g  = gini(fw)
    st = STYLE[name]
    ax.plot(cum_pop, cum_wlth, label=f"{name} (Gini={g:.3f})", **st)
    ax.fill_between(cum_pop, cum_pop, cum_wlth, alpha=0.04, color=st['color'])

ax.set_xlabel('Cumulative Share of Population')
ax.set_ylabel('Cumulative Share of Wealth')
ax.set_title('Lorenz Curves: Wealth Distribution (t=240)', fontweight='bold')
ax.legend(fontsize=8.5)
plt.tight_layout()
save(fig, "fig2_lorenz_curves")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Gini evolution over time
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 3 — Gini evolution...")
fig, ax = plt.subplots(figsize=(8, 4))

for name, d in data.items():
    ax.plot(months, d['gini_ts'], label=name, **STYLE[name])

ax.set_xlabel('Month')
ax.set_ylabel('Gini Coefficient')
ax.set_title('Gini Coefficient Evolution over 20 Years (100 Agents)', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.axhline(0.4, color='gray', lw=0.7, ls=':', alpha=0.7)
ax.text(5, 0.41, 'Gini=0.4 (moderate inequality)', fontsize=7.5, color='gray')
ax.legend(fontsize=8)
plt.tight_layout()
save(fig, "fig3_gini_evolution")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Cost vs Performance scatter
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 4 — Cost vs Performance...")
fig, ax = plt.subplots(figsize=(7, 5))

COST_PERF = [
    # (label,                       cost,    wealth_M, color,     high_err)
    ("GPT-5.2\nBaseline",           93.55,   0.428,    '#888888', False),
    ("GPT-5.2\n+ FinEvo",          142.59,   2.614,    '#2196F3', False),
    ("Llama-4-Maverick\n+ FinEvo",   0.0,    1.401,    '#E91E63', False),
    ("Llama-3.3-70B\n+ FinEvo",      0.0,    2.297,    '#FF9800', False),
    ("Qwen3-32B\n+ FinEvo*",         0.0,    0.027,    '#9C27B0', True),   # 42% error
]

for label, cost, wealth, color, high_err in COST_PERF:
    marker = 'x' if high_err else 'o'
    alpha  = 0.45 if high_err else 0.9
    ax.scatter(cost + 0.1, wealth, s=220, c=color, marker=marker, alpha=alpha, zorder=5)
    ax.annotate(label, (cost + 0.1, wealth),
                textcoords='offset points', xytext=(8, 4), fontsize=7.5)

ax.set_xscale('log')
ax.set_xlabel('API Cost (USD, log scale)')
ax.set_ylabel('Final Avg Wealth (Millions)')
ax.set_title('Cost–Performance Trade-off\n100 Agents / 240 Months', fontweight='bold')
ax.axvline(x=1.0, color='gray', ls=':', alpha=0.5)
ax.text(0.15, 0.05, 'Zero-cost\nopen-weight', fontsize=8,
        color='gray', transform=ax.transAxes)
plt.tight_layout()
save(fig, "fig4_cost_performance")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Ablation bar charts (10A / 24M)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 5 — Ablation bars...")
ABL = [
    # (label,                       wealth_K, gini,  unemp_pct)
    ("GPT-4.1-mini\nBaseline",       32.6,    0.542, 5.42),
    ("GPT-5.2\n+ FinEvo",           157.2,    0.269, 0.83),
    ("GPT-4o\n+ FinEvo",            229.7,    0.272, 0.00),
    ("Llama-4-Maverick\n+ FinEvo",  117.5,    0.544, 19.58),
    ("Qwen2.5-72B\n+ FinEvo",        59.1,    0.362, 1.67),
]
names      = [r[0] for r in ABL]
wealth_k   = [r[1] for r in ABL]
ginis      = [r[2] for r in ABL]
unemp      = [r[3] for r in ABL]
bar_colors = ['#888888', '#2196F3', '#4DB6AC', '#E91E63', '#9C27B0']
x = np.arange(len(names))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
panels = [
    (wealth_k, '(a) Avg Wealth (K Coin)',    'Avg Wealth (×1000)'),
    (ginis,    '(b) Gini Coefficient',        'Gini Coefficient'),
    (unemp,    '(c) Unemployment Rate (%)',   'Unemployment (%)'),
]
for ax, (vals, title, ylabel) in zip(axes, panels):
    ax.bar(x, vals, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel(ylabel)

# Moderate Gini reference line
axes[1].axhline(0.4, color='red', lw=0.8, ls='--', alpha=0.6, label='Moderate threshold')
axes[1].legend(fontsize=7.5)

plt.suptitle('Small-Scale Evaluation: 10 Agents / 24 Months', fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, "fig5_ablation_smallscale")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Baseline vs FinEvo head-to-head (GPT-5.2)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 6 — Baseline vs FinEvo...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

base_w   = data['GPT-5.2 Baseline']['wealth_ts']
finevo_w = data['GPT-5.2 + FinEvo']['wealth_ts']
base_u   = data['GPT-5.2 Baseline']['unemp_ts']
finevo_u = data['GPT-5.2 + FinEvo']['unemp_ts']

ax = axes[0]
ax.plot(months, base_w,   color='#888888', ls='--', lw=2,   label='GPT-5.2 Baseline')
ax.fill_between(months, base_w,   alpha=0.08, color='#888888')
ax.plot(months, finevo_w, color='#2196F3', ls='-',  lw=2.2, label='GPT-5.2 + FinEvo')
ax.fill_between(months, finevo_w, alpha=0.08, color='#2196F3')
ax.set_xlabel('Month'); ax.set_ylabel('Avg. Wealth (Coin)')
ax.set_title('(a) Wealth Accumulation: Baseline vs. FinEvo', fontweight='bold')
ax.legend()

ax = axes[1]
ax.plot(months, base_u,   color='#888888', ls='--', lw=2,   label='GPT-5.2 Baseline')
ax.fill_between(months, base_u,   alpha=0.08, color='#888888')
ax.plot(months, finevo_u, color='#2196F3', ls='-',  lw=2.2, label='GPT-5.2 + FinEvo')
ax.fill_between(months, finevo_u, alpha=0.08, color='#2196F3')
ax.set_xlabel('Month'); ax.set_ylabel('Unemployment Rate (%)')
ax.set_title('(b) Unemployment Rate: Baseline vs. FinEvo', fontweight='bold')
ax.legend()

plt.tight_layout()
save(fig, "fig6_baseline_vs_finevo")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Radar chart: multidimensional performance comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 7 — Radar chart...")

labels   = ['Avg. Wealth', 'Equality\n(1 − Gini)', 'Employment\nRate', 'GDP Growth']
num_vars = len(labels)

data_raw = {
    'GPT-5.2 Baseline':      [428417,   1 - 0.400, 100 - 15.61, -1.96],
    'GPT-5.2 + FinEvo':      [2614208,  1 - 0.388, 100 -  0.16,  0.14],
    'Llama-4-Mav + FinEvo':  [1401208,  1 - 0.759, 100 -  1.98,  0.46],
    'Llama-3.3-70B + FinEvo':[2297338,  1 - 0.519, 100 -  1.29,  0.10],
}

bounds = {
    'Avg. Wealth':         (0,   3_000_000),
    'Equality (1 − Gini)': (0.2, 0.7),
    'Employment Rate':     (70,  100),
    'GDP Growth':          (-2.5, 1.0),
}

data_norm = {}
for model, values in data_raw.items():
    norm_vals = []
    for i, val in enumerate(values):
        lo, hi = list(bounds.values())[i]
        norm_vals.append(max(0.0, min(1.0, (val - lo) / (hi - lo))))
    data_norm[model] = norm_vals

angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], labels, color='dimgrey', size=12, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75, 1.0], ['25%', '50%', '75%', 'Max'], color='grey', size=9)
plt.ylim(0, 1.1)

RADAR_STYLE = [
    ('GPT-5.2 Baseline',       '#888888', '--'),
    ('GPT-5.2 + FinEvo',       '#2196F3', '-'),
    ('Llama-4-Mav + FinEvo',   '#E91E63', '-'),
    ('Llama-3.3-70B + FinEvo', '#FF9800', '-'),
]
for model, color, ls in RADAR_STYLE:
    vals = data_norm[model] + data_norm[model][:1]
    ax.plot(angles, vals, linewidth=2.5, linestyle=ls, label=model, color=color)
    ax.fill(angles, vals, color=color, alpha=0.08)

plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.title('Multidimensional Macroeconomic Performance\n(100 Agents / 240 Months)',
          size=13, fontweight='bold', y=1.12)
plt.tight_layout()
save(fig, "fig7_radar_comparison")

print("\nDone! All 7 figures saved to figs/paper/")
