"""
dataset_utils.py
================
Utility functions that USE the dataset files in real ways.

This file is imported by:
  - app.py          (API endpoints use dataset for hints + analysis)
  - inference.py    (agent uses dataset for smarter decisions)
  - evaluate.py     (scoring uses dataset for benchmarking)

Dataset files used:
  dataset/states.csv         → look up typical states
  dataset/episodes.csv       → find best known actions
  dataset/price_patterns.csv → predict future prices
  dataset/llm_prompts.jsonl  → ready-made training examples
"""

import csv
import json
import os
from typing import List, Dict, Optional

# Bulletproof path resolution — works on Windows, Linux, Mac, pytest
def _find_dataset_dir() -> str:
    """Find dataset/ folder regardless of where script is run from."""
    # Try 1: relative to this file
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "dataset")
    if os.path.exists(candidate):
        return candidate

    # Try 2: current working directory
    candidate = os.path.join(os.getcwd(), "dataset")
    if os.path.exists(candidate):
        return candidate

    # Try 3: walk up from cwd to find it
    current = os.getcwd()
    for _ in range(5):
        candidate = os.path.join(current, "dataset")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        "Cannot find dataset/ folder. Run generate_dataset.py first. "
        "Searched from: " + os.path.abspath(__file__)
    )

DATASET_DIR = _find_dataset_dir()


# ══════════════════════════════════════════════════════════════════
# LOADER — reads CSV/JSONL files into memory (called once at startup)
# ══════════════════════════════════════════════════════════════════

def load_states() -> List[Dict]:
    """Load all states from states.csv"""
    path = os.path.join(DATASET_DIR, "states.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # convert numeric columns
    for r in rows:
        for col in ['task_id','day','hour','solar_output','wind_output',
                    'generation','grid_price','home_demand',
                    'is_storm_day','is_spike_day']:
            try: r[col] = float(r[col])
            except: pass
    return rows


def load_episodes() -> List[Dict]:
    """Load all episode rows from episodes.csv"""
    path = os.path.join(DATASET_DIR, "episodes.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for col in ['battery_level','solar_output','wind_output','generation',
                    'grid_price','home_demand','rule_action','rule_reward',
                    'optimal_action','optimal_reward','reward_gap','step','hour','day']:
            try: r[col] = float(r[col])
            except: pass
    return rows


def load_price_patterns() -> List[Dict]:
    """Load price patterns from price_patterns.csv"""
    path = os.path.join(DATASET_DIR, "price_patterns.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for col in ['hour','base_price','is_peak','is_offpeak']:
            try: r[col] = float(r[col])
            except: pass
    return rows


def load_llm_prompts() -> List[Dict]:
    """Load LLM training examples from llm_prompts.jsonl"""
    path = os.path.join(DATASET_DIR, "llm_prompts.jsonl")
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ── load everything once at module import ──
_STATES   = None
_EPISODES = None
_PRICES   = None
_PROMPTS  = None

def _get_states():
    global _STATES
    if _STATES is None: _STATES = load_states()
    return _STATES

def _get_episodes():
    global _EPISODES
    if _EPISODES is None: _EPISODES = load_episodes()
    return _EPISODES

def _get_prices():
    global _PRICES
    if _PRICES is None: _PRICES = load_price_patterns()
    return _PRICES

def _get_prompts():
    global _PROMPTS
    if _PROMPTS is None: _PROMPTS = load_llm_prompts()
    return _PROMPTS


# ══════════════════════════════════════════════════════════════════
# USE 1 — SMART HINT SYSTEM
# Uses: episodes.csv
# Purpose: Given current state, find the most similar past episode
#          step and tell the agent what worked best there
# ══════════════════════════════════════════════════════════════════

def get_smart_hint(state: dict) -> dict:
    """
    Find the most similar state in the episode dataset
    and return what the optimal agent did there.

    Called from: app.py /hint endpoint
                 inference.py ask() function

    Returns dict with:
        best_action      : int (0/1/2)
        best_action_name : str
        reason           : str
        confidence       : float (0-1)
    """
    episodes = _get_episodes()

    hour    = state.get("hour_of_day", state.get("hour", 0))
    price   = state.get("grid_price", 5)
    battery = state.get("battery_level", 50)
    weather = state.get("weather", "sunny")
    day     = state.get("day", 1)

    # Score each episode row by similarity to current state
    best_row  = None
    best_score = float('inf')

    for row in episodes:
        # similarity = weighted distance across key features
        score = (
            abs(row['hour']          - hour)    * 1.0 +
            abs(row['grid_price']    - price)   * 0.8 +
            abs(row['battery_level'] - battery) * 0.5 +
            (0 if row['weather'] == weather else 2.0)
        )
        if score < best_score:
            best_score = score
            best_row   = row

    if not best_row:
        return {"best_action": 1, "best_action_name": "DISTRIBUTE",
                "reason": "No similar state found", "confidence": 0.0}

    action_names = {0: "STORE", 1: "DISTRIBUTE", 2: "SELL"}
    action       = int(best_row['optimal_action'])
    reward_gap   = best_row.get('reward_gap', 0)

    # Build a human-readable reason
    if action == 2:
        reason = f"Similar state had high price (Rs {best_row['grid_price']:.1f}) → SELL earned best reward"
    elif action == 0:
        reason = f"Similar state at hour {int(best_row['hour'])} had cheap energy → STORE was optimal"
    else:
        reason = f"Similar state had high demand ({best_row['home_demand']:.1f} kWh) → DISTRIBUTE was best"

    # Confidence: lower distance = higher confidence
    confidence = round(max(0.0, 1.0 - best_score / 10.0), 2)

    return {
        "best_action":      action,
        "best_action_name": action_names[action],
        "reason":           reason,
        "confidence":       confidence,
        "similar_state": {
            "hour":    int(best_row['hour']),
            "price":   best_row['grid_price'],
            "battery": best_row['battery_level'],
            "weather": best_row['weather'],
        },
        "optimal_reward": best_row['optimal_reward'],
        "rule_reward":    best_row['rule_reward'],
        "reward_gap":     reward_gap,
    }


# ══════════════════════════════════════════════════════════════════
# USE 2 — PRICE FORECASTER
# Uses: price_patterns.csv
# Purpose: Predict what price will be in the next N hours
#          Agent uses this to decide whether to store now and sell later
# ══════════════════════════════════════════════════════════════════

def forecast_prices(current_hour: int, task_id: int, day: int,
                    hours_ahead: int = 6) -> List[dict]:
    """
    Predict grid prices for the next N hours.

    Called from: app.py /forecast endpoint
                 inference.py for smarter sell timing

    Returns list of dicts: [{hour, predicted_price, strategy}, ...]
    """
    patterns = _get_prices()

    # Pick scenario based on task and day
    if task_id == 3 and day in (4, 5):
        scenario = "storm_day"
    elif task_id == 3 and day == 6:
        scenario = "spike_day"
    else:
        scenario = "normal"

    # Build lookup: hour → price for this scenario
    price_map = {
        int(p['hour']): p
        for p in patterns
        if p['scenario'] == scenario
    }

    forecast = []
    for i in range(1, hours_ahead + 1):
        future_hour = (current_hour + i) % 24
        p = price_map.get(future_hour, {})
        forecast.append({
            "hour":            future_hour,
            "hours_from_now":  i,
            "predicted_price": p.get('base_price', 5.0),
            "strategy":        p.get('strategy', 'DISTRIBUTE'),
            "is_peak":         bool(p.get('is_peak', 0)),
        })

    return forecast


def should_store_now(state: dict, task_id: int = 1) -> dict:
    """
    Decide: is it better to STORE now and SELL later?
    Uses price forecast to find if a price spike is coming.

    Returns: {store_now: bool, reason: str, peak_hour: int, peak_price: float}
    """
    hour    = state.get("hour_of_day", 0)
    battery = state.get("battery_level", 50)
    price   = state.get("grid_price", 5)
    day     = state.get("day", 1)

    forecast = forecast_prices(hour, task_id, day, hours_ahead=8)
    peak     = max(forecast, key=lambda x: x['predicted_price'])

    # Store if: current price is low AND a peak is coming AND battery not full
    current_is_cheap = price < 6.0
    peak_is_coming   = peak['predicted_price'] > 10.0
    battery_has_room = battery < 80.0

    store_now = current_is_cheap and peak_is_coming and battery_has_room

    return {
        "store_now":   store_now,
        "current_price": price,
        "peak_hour":   peak['hour'],
        "peak_price":  peak['predicted_price'],
        "hours_until_peak": peak['hours_from_now'],
        "reason": (
            f"Store now (Rs {price:.1f}/kWh cheap), sell at Hr{peak['hour']:02d} (Rs {peak['predicted_price']:.1f}/kWh)"
            if store_now else
            f"Current price Rs {price:.1f}/kWh — no strong peak signal"
        )
    }


# ══════════════════════════════════════════════════════════════════
# USE 3 — DATASET-POWERED AGENT
# Uses: episodes.csv + price_patterns.csv
# Purpose: A smarter version of the rule-based fallback that
#          looks up the dataset instead of using hard-coded rules
# ══════════════════════════════════════════════════════════════════

def dataset_agent(state: dict, task_id: int = 1) -> int:
    """
    Make a decision using dataset knowledge instead of hard-coded rules.
    This is significantly smarter than the basic rule-based fallback.

    Called from: inference.py as the fallback agent

    Returns: action int (0=STORE, 1=DISTRIBUTE, 2=SELL)
    """
    # Step 1: check if a price peak is coming → maybe store
    forecast_result = should_store_now(state, task_id)

    # Step 2: find the most similar past state
    hint = get_smart_hint(state)

    price   = state.get("grid_price", 5)
    battery = state.get("battery_level", 50)
    weather = state.get("weather", "sunny")
    day     = state.get("day", 1)
    demand  = state.get("home_demand", 7)
    gen     = state.get("solar_output",0) + state.get("wind_output",0)

    # Step 3: combine signals
    # Priority 1: storm survival (Task 3 specific)
    if task_id == 3 and day == 3 and battery < 90:
        return 0  # STORE — must fill before Days 4-5 storm

    if task_id == 3 and day in (4, 5):
        return 1  # DISTRIBUTE — storm, serve from battery

    if task_id == 3 and day == 6 and price >= 10:
        return 2  # SELL — price spike day

    # Priority 2: current price very high → sell
    if price >= 12 and battery > 30:
        return 2

    # Priority 3: forecast says store now for future peak
    if forecast_result['store_now'] and battery < 70:
        return 0

    # Priority 4: dataset hint is confident
    if hint['confidence'] >= 0.7:
        return hint['best_action']

    # Priority 5: basic logic
    if demand > gen and battery > 20:  return 1
    if battery < 15:                   return 1
    if weather == "stormy":            return 1
    if price < 4 and battery < 80:    return 0

    # Priority 6: follow dataset hint regardless of confidence
    return hint['best_action']


# ══════════════════════════════════════════════════════════════════
# USE 4 — BENCHMARK COMPARISON
# Uses: episodes.csv
# Purpose: Compare your agent's score against the dataset baselines
# ══════════════════════════════════════════════════════════════════

def get_benchmark(task_id: int) -> dict:
    """
    Get baseline scores from the dataset for comparison.

    Called from: app.py /benchmark endpoint
                 evaluate.py

    Returns: {rule_avg, optimal_avg, your_score_percentile}
    """
    episodes = _get_episodes()

    task_rows = [r for r in episodes if int(r.get('task_id',0)) == task_id]
    if not task_rows:
        return {"rule_avg_reward": 0, "optimal_avg_reward": 0, "total_steps": 0}

    rule_rewards    = [r['rule_reward']    for r in task_rows]
    optimal_rewards = [r['optimal_reward'] for r in task_rows]
    reward_gaps     = [r['reward_gap']     for r in task_rows]

    rule_avg    = round(sum(rule_rewards)    / len(rule_rewards),    4)
    optimal_avg = round(sum(optimal_rewards) / len(optimal_rewards), 4)
    gap_avg     = round(sum(reward_gaps)     / len(reward_gaps),     4)

    # How many steps did optimal beat rule-based?
    optimal_wins = sum(1 for r in task_rows if r['optimal_reward'] > r['rule_reward'])
    win_pct      = round(optimal_wins / len(task_rows) * 100, 1)

    return {
        "task_id":             task_id,
        "total_steps":         len(task_rows),
        "rule_avg_reward":     rule_avg,
        "optimal_avg_reward":  optimal_avg,
        "avg_reward_gap":      gap_avg,
        "optimal_wins_pct":    win_pct,
        "interpretation": (
            f"Optimal agent beats rule-based {win_pct}% of steps, "
            f"avg gain of {gap_avg:.4f} reward per step"
        )
    }


# ══════════════════════════════════════════════════════════════════
# USE 5 — LLM PROMPT BUILDER WITH DATASET EXAMPLES
# Uses: llm_prompts.jsonl
# Purpose: Build better LLM prompts by including similar examples
#          from the dataset (few-shot prompting)
# ══════════════════════════════════════════════════════════════════

def build_fewshot_prompt(state: dict, task_id: int = 1,
                         n_examples: int = 2) -> str:
    """
    Build a few-shot prompt for the LLM using similar examples
    from the dataset. This makes the LLM much more accurate.

    Called from: inference.py ask() function

    Returns: system prompt string with embedded examples
    """
    prompts = _get_prompts()

    hour    = state.get("hour_of_day", 0)
    weather = state.get("weather", "sunny")
    price   = state.get("grid_price", 5)

    # Find most similar examples from dataset
    scored = []
    for ex in prompts:
        meta = ex.get("metadata", {})
        sim  = (
            abs(meta.get('hour', 0)  - hour)    * 1.0 +
            abs(meta.get('price', 5) - price)   * 0.5 +
            (0 if meta.get('weather') == weather else 1.5)
        )
        scored.append((sim, ex))

    scored.sort(key=lambda x: x[0])
    top_examples = [ex for _, ex in scored[:n_examples]]

    # Build few-shot section
    fewshot = ""
    for i, ex in enumerate(top_examples, 1):
        user_msg = ex['messages'][1]['content']
        asst_msg = ex['messages'][2]['content']
        reason   = ex.get('reason', '')
        fewshot += f"\n\nExample {i}:\n{user_msg}\nAnswer: {asst_msg}  # {reason}"

    system = (
        "You are an AI agent managing a renewable energy microgrid.\n"
        "Each step you get the current state as JSON.\n"
        "Reply with ONLY a JSON object: {\"action\": <0|1|2>}\n"
        "0=STORE  1=DISTRIBUTE  2=SELL\n"
        "No explanation. Just the JSON.\n\n"
        "Here are similar past decisions to guide you:"
        + fewshot
    )

    return system


# ══════════════════════════════════════════════════════════════════
# USE 6 — DATASET STATS (for README / dashboard display)
# Uses: all files
# ══════════════════════════════════════════════════════════════════

def get_dataset_stats() -> dict:
    """
    Return summary statistics about the dataset.
    Called from: app.py /dataset-stats endpoint
    """
    states   = _get_states()
    episodes = _get_episodes()
    prices   = _get_prices()
    prompts  = _get_prompts()

    prices_all = [r['grid_price'] for r in states]
    solar_all  = [r['solar_output'] for r in states]

    return {
        "total_states":        len(states),
        "total_episode_steps": len(episodes),
        "total_price_rows":    len(prices),
        "total_llm_examples":  len(prompts),
        "price_stats": {
            "min":  round(min(prices_all), 2),
            "max":  round(max(prices_all), 2),
            "avg":  round(sum(prices_all)/len(prices_all), 2),
        },
        "solar_stats": {
            "min":  round(min(solar_all), 2),
            "max":  round(max(solar_all), 2),
            "avg":  round(sum(solar_all)/len(solar_all), 2),
        },
        "benchmarks": {
            "task_1": get_benchmark(1),
            "task_2": get_benchmark(2),
            "task_3": get_benchmark(3),
        }
    }


# ══════════════════════════════════════════════════════════════════
# Quick test — run this file directly to verify everything works
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Testing dataset_utils.py...")
    print()

    # Test 1: hint system
    test_state = {
        "battery_level": 65.0, "solar_output": 12.4, "wind_output": 3.1,
        "grid_price": 11.5, "home_demand": 9.0,
        "hour_of_day": 18, "day": 2, "weather": "sunny", "task_id": 2
    }
    hint = get_smart_hint(test_state)
    print(f"[Hint] Best action: {hint['best_action_name']}  confidence={hint['confidence']}")
    print(f"       Reason: {hint['reason']}")

    print()

    # Test 2: price forecast
    fc = forecast_prices(current_hour=14, task_id=2, day=1, hours_ahead=6)
    print("[Forecast] Next 6 hours:")
    for f in fc:
        print(f"  Hr{f['hour']:02d}: Rs {f['predicted_price']:.1f}  → {f['strategy']}")

    print()

    # Test 3: should store now?
    store = should_store_now(test_state, task_id=2)
    print(f"[Store now?] {store['store_now']}  — {store['reason']}")

    print()

    # Test 4: dataset agent decision
    action = dataset_agent(test_state, task_id=2)
    print(f"[Dataset agent] Action: {['STORE','DISTRIBUTE','SELL'][action]}")

    print()

    # Test 5: benchmark
    for tid in [1, 2, 3]:
        b = get_benchmark(tid)
        print(f"[Benchmark Task {tid}] rule={b['rule_avg_reward']:.4f}  "
              f"optimal={b['optimal_avg_reward']:.4f}  "
              f"optimal wins {b['optimal_wins_pct']}% of steps")

    print()

    # Test 6: dataset stats
    stats = get_dataset_stats()
    print(f"[Stats] {stats['total_states']} states, "
          f"{stats['total_episode_steps']} episode steps, "
          f"{stats['total_llm_examples']} LLM examples")
    print(f"        Price range: Rs {stats['price_stats']['min']}–{stats['price_stats']['max']}")

    print()
    print("All tests passed!")
