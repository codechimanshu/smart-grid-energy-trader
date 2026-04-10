"""
generate_dataset.py
===================
Generates the complete Smart Grid dataset.
Run once to create all dataset files inside the /dataset folder.

Usage:
    python generate_dataset.py
"""

import random, math, json, csv, os

os.makedirs("dataset", exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────

def get_weather(day, task_id, rng):
    if task_id == 3:
        return {1:'sunny',2:'sunny',3:'cloudy',
                4:'stormy',5:'stormy',6:'sunny',7:'sunny'}.get(day,'sunny')
    return rng.choice(['sunny','sunny','sunny','cloudy','cloudy'])

def get_solar(hour, weather, rng):
    if weather == 'stormy': return 0.0
    f = 0.3 if weather == 'cloudy' else 1.0
    c = max(0, math.sin(math.pi*(hour-6)/12)) if 6<=hour<=18 else 0
    return round(c*15.0*f + rng.uniform(0,0.5), 2)

def get_wind(weather, rng):
    if weather == 'stormy': return round(rng.uniform(8,14),2)
    if weather == 'cloudy': return round(rng.uniform(4,8),2)
    return round(rng.uniform(2,6),2)

def get_demand(hour, rng):
    base = 12.0 if (7<=hour<=9 or 18<=hour<=21) else 4.0 if hour<=5 else 7.0
    return round(base + rng.uniform(-1,1), 2)

def get_price(hour, day, task_id, rng):
    base = 3.0 if hour<6 else 6.0 if hour<10 else 5.0 if hour<17 else 12.0 if hour<21 else 7.0
    if task_id==3 and day==6: base *= 2.5
    return round(base + rng.uniform(-0.5,0.5), 2)

def rule_agent(s):
    p,bat,dem = s['grid_price'],s['battery_level'],s['home_demand']
    gen = s['solar_output']+s['wind_output']
    w,d = s['weather'],s['day']
    if d==3 and bat<80:    return 0,'STORE'
    if d==6 and p>=10:     return 2,'SELL'
    if p>=10 and bat>30:   return 2,'SELL'
    if dem>gen and bat>20: return 1,'DISTRIBUTE'
    if bat<20:             return 1,'DISTRIBUTE'
    if w=='stormy':        return 1,'DISTRIBUTE'
    return 0,'STORE'

def optimal_agent(s):
    p,bat,dem = s['grid_price'],s['battery_level'],s['home_demand']
    gen = s['solar_output']+s['wind_output']
    w,d = s['weather'],s['day']
    if d==3 and bat<95:           return 0,'STORE'
    if d in (4,5) and bat>10:     return 1,'DISTRIBUTE'
    if d==6 and p>=10:            return 2,'SELL'
    if p>=12 and bat>40:          return 2,'SELL'
    if p>=8  and bat>60:          return 2,'SELL'
    if gen>dem+2 and bat<85:      return 0,'STORE'
    if dem>gen:                    return 1,'DISTRIBUTE'
    return 0,'STORE'

def compute_reward(action, gen, demand, price, battery):
    MAX_BAT = 100.0
    reward = 0.0
    if action == 0:   # STORE
        space   = MAX_BAT - battery
        stored  = min(gen, space)
        waste   = gen - stored
        new_bat = min(MAX_BAT, battery + stored)
        serve   = min(demand, new_bat)
        new_bat -= serve
        short   = max(0, demand - serve)
        reward  = -(waste*0.05) - (short*0.3)
        new_bat = max(0, min(MAX_BAT, new_bat))
    elif action == 1: # DISTRIBUTE
        served  = min(gen, demand)
        surplus = gen - served
        short   = max(0, demand - served)
        space   = MAX_BAT - battery
        stored  = min(surplus, space)
        waste   = surplus - stored
        new_bat = min(MAX_BAT, battery + stored)
        reward  = -(short*0.3) - (waste*0.02)
    else:             # SELL
        revenue = gen * price
        serve   = min(demand, battery)
        new_bat = battery - serve
        short   = max(0, demand - serve)
        reward  = (revenue/100.0) - (short*0.5)
        new_bat = max(0, min(MAX_BAT, new_bat))
    if 30 <= battery <= 80:
        reward += 0.02
    return round(reward, 4)


# ════════════════════════════════════════════════════════════════════
# Dataset 1 — states.csv
# Every possible hour/day/weather combination with all state values
# ════════════════════════════════════════════════════════════════════
print("Generating dataset/states.csv ...")
rng = random.Random(42)
rows = []
for task_id in [1,2,3]:
    max_day = 1 if task_id==1 else 1 if task_id==2 else 7
    for day in range(1, max_day+1):
        for hour in range(24):
            weather = get_weather(day, task_id, rng)
            solar   = get_solar(hour, weather, rng)
            wind    = get_wind(weather, rng)
            demand  = get_demand(hour, rng)
            price   = get_price(hour, day, task_id, rng)
            rows.append({
                'task_id':      task_id,
                'day':          day,
                'hour':         hour,
                'weather':      weather,
                'solar_output': solar,
                'wind_output':  wind,
                'generation':   round(solar+wind,2),
                'grid_price':   price,
                'home_demand':  demand,
                'price_zone':   'peak' if 17<=hour<=20 else 'off_peak' if hour<6 else 'mid',
                'is_storm_day': int(task_id==3 and day in (4,5)),
                'is_spike_day': int(task_id==3 and day==6),
            })

with open("dataset/states.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"  → {len(rows)} rows")


# ════════════════════════════════════════════════════════════════════
# Dataset 2 — episodes.csv
# Full episodes with rule-based and optimal agent decisions + rewards
# ════════════════════════════════════════════════════════════════════
print("Generating dataset/episodes.csv ...")
erows = []
episode_id = 0

for seed in range(20):          # 20 different seeds
    for task_id in [1,2,3]:
        rng2 = random.Random(seed*100+task_id)
        max_steps = {1:1,2:24,3:168}[task_id]
        battery = 50.0
        episode_id += 1

        for step in range(max_steps):
            day  = (step//24)+1
            hour = step%24
            weather = get_weather(day, task_id, rng2)
            solar   = get_solar(hour, weather, rng2)
            wind    = get_wind(weather, rng2)
            demand  = get_demand(hour, rng2)
            price   = get_price(hour, day, task_id, rng2)
            gen     = solar+wind

            state = {
                'battery_level':battery,'solar_output':solar,
                'wind_output':wind,'grid_price':price,
                'home_demand':demand,'weather':weather,'day':day
            }

            rule_a, rule_name   = rule_agent(state)
            opt_a,  opt_name    = optimal_agent(state)
            rule_r  = compute_reward(rule_a, gen, demand, price, battery)
            opt_r   = compute_reward(opt_a,  gen, demand, price, battery)

            erows.append({
                'episode_id':    episode_id,
                'seed':          seed,
                'task_id':       task_id,
                'step':          step+1,
                'day':           day,
                'hour':          hour,
                'weather':       weather,
                'battery_level': round(battery,2),
                'solar_output':  solar,
                'wind_output':   wind,
                'generation':    round(gen,2),
                'grid_price':    price,
                'home_demand':   demand,
                'rule_action':   rule_a,
                'rule_action_name': rule_name,
                'rule_reward':   rule_r,
                'optimal_action':opt_a,
                'optimal_action_name': opt_name,
                'optimal_reward':opt_r,
                'reward_gap':    round(opt_r - rule_r, 4),
            })

            # advance battery using optimal action for next state
            if opt_a == 0:
                battery = min(100, battery + min(gen, 100-battery))
            elif opt_a == 1:
                surplus = max(0, gen-demand)
                battery = min(100, battery + min(surplus, 100-battery))
            else:
                battery = max(0, battery - min(demand, battery))
            battery = round(battery, 2)

with open("dataset/episodes.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=erows[0].keys())
    w.writeheader(); w.writerows(erows)
print(f"  → {len(erows)} rows")


# ════════════════════════════════════════════════════════════════════
# Dataset 3 — price_patterns.csv
# 24-hour price curves for different scenarios
# ════════════════════════════════════════════════════════════════════
print("Generating dataset/price_patterns.csv ...")
prows = []
for scenario in ['normal','storm_day','spike_day','weekend']:
    for hour in range(24):
        base = 3.0 if hour<6 else 6.0 if hour<10 else 5.0 if hour<17 else 12.0 if hour<21 else 7.0
        if scenario == 'spike_day':  base *= 2.5
        if scenario == 'storm_day':  base *= 0.8   # less demand
        if scenario == 'weekend':    base *= 0.9
        prows.append({
            'scenario':    scenario,
            'hour':        hour,
            'base_price':  round(base,2),
            'is_peak':     int(17<=hour<=20),
            'is_offpeak':  int(hour<6),
            'strategy':    'SELL' if base>=10 else 'STORE' if base<5 else 'DISTRIBUTE'
        })

with open("dataset/price_patterns.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=prows[0].keys())
    w.writeheader(); w.writerows(prows)
print(f"  → {len(prows)} rows")


# ════════════════════════════════════════════════════════════════════
# Dataset 4 — llm_prompts.jsonl
# Ready-to-use training examples in LLM chat format
# ════════════════════════════════════════════════════════════════════
print("Generating dataset/llm_prompts.jsonl ...")
rng3  = random.Random(99)
examples = []

SYSTEM = (
    "You are an AI agent managing a renewable energy microgrid. "
    "Each step you get the current state as JSON. "
    "Reply with ONLY a JSON object: {\"action\": <0|1|2>} "
    "0=STORE  1=DISTRIBUTE  2=SELL. No explanation."
)

for i in range(500):
    task_id = rng3.choice([1,1,2,2,3])
    day     = rng3.randint(1,7) if task_id==3 else 1
    hour    = rng3.randint(0,23)
    weather = get_weather(day, task_id, rng3)
    solar   = get_solar(hour, weather, rng3)
    wind    = get_wind(weather, rng3)
    demand  = get_demand(hour, rng3)
    price   = get_price(hour, day, task_id, rng3)
    battery = round(rng3.uniform(10,90),1)

    state = {
        'battery_level':battery,'solar_output':solar,'wind_output':wind,
        'grid_price':price,'home_demand':demand,'hour_of_day':hour,
        'day':day,'weather':weather,'task_id':task_id
    }
    s = {'battery_level':battery,'solar_output':solar,'wind_output':wind,
         'grid_price':price,'home_demand':demand,'weather':weather,'day':day}
    opt_a, opt_name = optimal_agent(s)

    # explanation for why this action is optimal
    if opt_a == 2:
        reason = f"Grid price is high ({price} Rs/kWh). Sell to maximize revenue."
    elif opt_a == 0:
        reason = f"Cheap energy available. Store in battery (currently {battery}%)."
    else:
        reason = f"Homes need {demand} kWh but only {round(solar+wind,1)} kWh generating."

    examples.append({
        "messages": [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":f"Current state:\n{json.dumps(state,indent=2)}"},
            {"role":"assistant","content":json.dumps({"action":opt_a})}
        ],
        "optimal_action": opt_a,
        "optimal_action_name": opt_name,
        "reason": reason,
        "task_id": task_id,
        "metadata": {"hour":hour,"day":day,"weather":weather,"price":price}
    })

with open("dataset/llm_prompts.jsonl","w") as f:
    for ex in examples:
        f.write(json.dumps(ex)+"\n")
print(f"  → {len(examples)} examples")


# ════════════════════════════════════════════════════════════════════
# Dataset 5 — dataset_card.json  (HuggingFace dataset card info)
# ════════════════════════════════════════════════════════════════════
card = {
    "name": "smart-grid-energy-trader-dataset",
    "description": "Training and evaluation dataset for the Smart Grid Energy Trader OpenEnv environment.",
    "version": "1.0.0",
    "files": {
        "states.csv":        "All possible environment states across all tasks",
        "episodes.csv":      "Full episode rollouts with rule-based and optimal agent decisions",
        "price_patterns.csv":"24-hour price curves for 4 market scenarios",
        "llm_prompts.jsonl": "500 LLM training examples in OpenAI chat format"
    },
    "stats": {
        "states_rows":        len(rows),
        "episodes_rows":      len(erows),
        "price_pattern_rows": len(prows),
        "llm_examples":       len(examples),
    },
    "tasks": ["Single Decision (easy)","24-Hour Day Manager (medium)","7-Day Storm Survival (hard)"],
    "actions": {"0":"STORE","1":"DISTRIBUTE","2":"SELL"},
    "license": "MIT"
}

with open("dataset/dataset_card.json","w") as f:
    json.dump(card, f, indent=2)


# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════
print()
print("="*50)
print("  Dataset generation complete!")
print("="*50)
for fname, desc in card["files"].items():
    size = os.path.getsize(f"dataset/{fname}")
    print(f"  {fname:<25} {size:>8,} bytes")
print(f"\n  Total examples: {sum(card['stats'].values()):,}")
print("="*50)
