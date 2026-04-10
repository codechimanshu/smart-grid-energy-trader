"""
tests/test_environment.py
Tests for environment.py
Run: pytest tests/ -v
"""

import pytest
from environment import SmartGridEnv


@pytest.fixture
def env1():
    e = SmartGridEnv(task_id=1, seed=42)
    e.reset()
    return e

@pytest.fixture
def env2():
    e = SmartGridEnv(task_id=2, seed=42)
    e.reset()
    return e

@pytest.fixture
def env3():
    e = SmartGridEnv(task_id=3, seed=42)
    e.reset()
    return e


# ── reset ────────────────────────────────────────────────────────────

def test_reset_returns_state(env1):
    state = env1.reset()
    for key in ["battery_level","solar_output","wind_output",
                "grid_price","home_demand","hour_of_day","day","weather"]:
        assert key in state

def test_reset_battery_starts_at_50(env1):
    assert env1.reset()["battery_level"] == 50.0

def test_reset_weather_is_valid(env1):
    assert env1.reset()["weather"] in ("sunny","cloudy","stormy")

def test_reset_hour_starts_at_0(env1):
    assert env1.reset()["hour_of_day"] == 0

def test_reset_day_starts_at_1(env1):
    assert env1.reset()["day"] == 1

def test_reset_clears_previous_episode(env2):
    env2.step(2); env2.step(1)
    state = env2.reset()
    assert state["battery_level"] == 50.0
    assert state["hour_of_day"]   == 0


# ── step ─────────────────────────────────────────────────────────────

def test_step_returns_required_keys(env1):
    r = env1.step(1)
    for key in ["observation","reward","done","info"]:
        assert key in r

def test_step_reward_is_float(env1):
    assert isinstance(env1.step(0)["reward"], float)

def test_step_done_is_bool(env1):
    assert isinstance(env1.step(1)["done"], bool)

def test_step_observation_has_all_keys(env2):
    obs = env2.step(1)["observation"]
    for key in ["battery_level","solar_output","wind_output",
                "grid_price","home_demand","hour_of_day","day","weather"]:
        assert key in obs

def test_step_battery_stays_in_range(env2):
    for action in [0,1,2,0,2,1]:
        bat = env2.step(action)["observation"]["battery_level"]
        assert 0.0 <= bat <= 100.0

def test_step_invalid_action_does_not_crash(env1):
    r = env1.step(99)
    assert "observation" in r

def test_step_all_three_actions_work(env2):
    for action in [0,1,2]:
        env2.reset()
        r = env2.step(action)
        assert r["reward"] is not None


# ── task 1 (1 step) ──────────────────────────────────────────────────

def test_task1_done_after_1_step(env1):
    assert env1.step(2)["done"] is True

def test_task1_score_valid(env1):
    env1.step(2)
    assert 0.0 <= env1.score() <= 1.0


# ── task 2 (24 steps) ────────────────────────────────────────────────

def test_task2_not_done_after_1_step(env2):
    assert env2.step(1)["done"] is False

def test_task2_done_after_24_steps(env2):
    r = None
    for _ in range(24):
        r = env2.step(1)
    assert r["done"] is True

def test_task2_score_valid(env2):
    for _ in range(24): env2.step(1)
    assert 0.0 <= env2.score() <= 1.0

def test_task2_hour_advances(env2):
    for expected in range(1, 6):
        env2.step(1)
        assert env2.state()["hour_of_day"] == expected


# ── task 3 (168 steps) ───────────────────────────────────────────────

def test_task3_storm_days_zero_solar(env3):
    for _ in range(72): env3.step(1)   # advance to day 4
    state = env3.state()
    assert state["day"]     == 4
    assert state["weather"] == "stormy"
    assert state["solar_output"] == 0.0

def test_task3_done_after_168_steps(env3):
    r = None
    for _ in range(168): r = env3.step(1)
    assert r["done"] is True

def test_task3_score_valid(env3):
    for _ in range(168): env3.step(0)
    assert 0.0 <= env3.score() <= 1.0


# ── determinism ───────────────────────────────────────────────────────

def test_same_seed_same_state():
    e1 = SmartGridEnv(2, seed=42); s1 = e1.reset()
    e2 = SmartGridEnv(2, seed=42); s2 = e2.reset()
    assert s1["solar_output"] == s2["solar_output"]
    assert s1["grid_price"]   == s2["grid_price"]

def test_different_seeds_differ():
    s1 = SmartGridEnv(2, seed=1).reset()
    s2 = SmartGridEnv(2, seed=99).reset()
    assert s1 != s2
