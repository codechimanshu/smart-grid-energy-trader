"""
tests/test_dataset_utils.py
Tests for dataset_utils.py
"""

import pytest
from dataset_utils import (
    get_smart_hint, forecast_prices, should_store_now,
    get_benchmark, get_dataset_stats,
    build_fewshot_prompt, dataset_agent,
)

# ── shared sample states ──────────────────────────────────────────────

HIGH_PRICE = {
    "battery_level":65.0,"solar_output":12.4,"wind_output":3.1,
    "grid_price":11.5,"home_demand":9.0,"hour_of_day":18,
    "day":2,"weather":"sunny","task_id":2,
}
LOW_PRICE = {**HIGH_PRICE, "grid_price":2.5, "hour_of_day":2}
STORMY    = {**HIGH_PRICE, "weather":"stormy","day":4,
             "solar_output":0.0,"grid_price":4.0}


# ── get_smart_hint ────────────────────────────────────────────────────

def test_hint_has_required_keys():
    h = get_smart_hint(HIGH_PRICE)
    for k in ["best_action","best_action_name","reason","confidence"]:
        assert k in h

def test_hint_action_valid():
    assert get_smart_hint(HIGH_PRICE)["best_action"] in (0,1,2)

def test_hint_name_matches_action():
    h = get_smart_hint(HIGH_PRICE)
    assert h["best_action_name"] == {0:"STORE",1:"DISTRIBUTE",2:"SELL"}[h["best_action"]]

def test_hint_confidence_in_range():
    assert 0.0 <= get_smart_hint(HIGH_PRICE)["confidence"] <= 1.0

def test_hint_high_price_suggests_sell():
    assert get_smart_hint(HIGH_PRICE)["best_action"] == 2

def test_hint_reason_is_nonempty_string():
    r = get_smart_hint(HIGH_PRICE)["reason"]
    assert isinstance(r, str) and len(r) > 5


# ── forecast_prices ───────────────────────────────────────────────────

def test_forecast_correct_length():
    assert len(forecast_prices(14, task_id=2, day=1, hours_ahead=6)) == 6

def test_forecast_item_keys():
    for item in forecast_prices(14, task_id=2, day=1, hours_ahead=3):
        for k in ["hour","predicted_price","strategy","is_peak"]:
            assert k in item

def test_forecast_strategy_values():
    for item in forecast_prices(0, task_id=1, day=1, hours_ahead=24):
        assert item["strategy"] in ("STORE","DISTRIBUTE","SELL")

def test_forecast_spike_day_higher_than_normal():
    normal = forecast_prices(14, task_id=3, day=1, hours_ahead=6)
    spike  = forecast_prices(14, task_id=3, day=6, hours_ahead=6)
    avg_n  = sum(f["predicted_price"] for f in normal) / 6
    avg_s  = sum(f["predicted_price"] for f in spike)  / 6
    assert avg_s > avg_n


# ── should_store_now ──────────────────────────────────────────────────

def test_store_now_has_required_keys():
    r = should_store_now(HIGH_PRICE, task_id=2)
    for k in ["store_now","reason","peak_hour","peak_price"]:
        assert k in r

def test_store_now_is_bool():
    assert isinstance(should_store_now(HIGH_PRICE)["store_now"], bool)

def test_store_now_reason_is_string():
    r = should_store_now(HIGH_PRICE)["reason"]
    assert isinstance(r, str) and len(r) > 5


# ── get_benchmark ─────────────────────────────────────────────────────

def test_benchmark_required_keys():
    b = get_benchmark(1)
    for k in ["task_id","total_steps","rule_avg_reward",
              "optimal_avg_reward","optimal_wins_pct"]:
        assert k in b

def test_benchmark_all_tasks():
    for tid in [1,2,3]:
        b = get_benchmark(tid)
        assert b["task_id"] == tid
        assert b["total_steps"] > 0

def test_benchmark_win_pct_valid():
    for tid in [1,2,3]:
        assert 0.0 <= get_benchmark(tid)["optimal_wins_pct"] <= 100.0


# ── get_dataset_stats ─────────────────────────────────────────────────

def test_stats_counts_positive():
    s = get_dataset_stats()
    assert s["total_states"]        > 0
    assert s["total_episode_steps"] > 0
    assert s["total_llm_examples"]  > 0

def test_stats_price_range_valid():
    p = get_dataset_stats()["price_stats"]
    assert p["min"] > 0
    assert p["max"] > p["min"]

def test_stats_has_all_benchmarks():
    b = get_dataset_stats()["benchmarks"]
    for k in ["task_1","task_2","task_3"]:
        assert k in b


# ── build_fewshot_prompt ──────────────────────────────────────────────

def test_fewshot_is_nonempty_string():
    p = build_fewshot_prompt(HIGH_PRICE, task_id=2)
    assert isinstance(p, str) and len(p) > 100

def test_fewshot_contains_action_codes():
    p = build_fewshot_prompt(HIGH_PRICE)
    assert "0" in p and "1" in p and "2" in p

def test_fewshot_contains_examples():
    assert "Example" in build_fewshot_prompt(HIGH_PRICE, n_examples=2)


# ── dataset_agent ─────────────────────────────────────────────────────

def test_agent_returns_valid_action():
    for state in [HIGH_PRICE, LOW_PRICE, STORMY]:
        assert dataset_agent(state, task_id=2) in (0,1,2)

def test_agent_distributes_during_storm():
    assert dataset_agent(STORMY, task_id=3) == 1

def test_agent_stores_before_storm():
    state = {**HIGH_PRICE,"day":3,"battery_level":50.0,
             "task_id":3,"grid_price":5.0}
    assert dataset_agent(state, task_id=3) == 0

def test_agent_sells_on_spike_day():
    state = {**HIGH_PRICE,"day":6,"grid_price":28.0,
             "battery_level":70.0,"task_id":3}
    assert dataset_agent(state, task_id=3) == 2
