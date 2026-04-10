"""
tests/test_tasks.py
Tests for tasks.py
"""

import pytest
from tasks import TASKS, get_prompt, run_grader


def test_three_tasks_defined():
    assert len(TASKS) == 3

def test_task_ids_are_1_2_3():
    assert [t["id"] for t in TASKS] == [1, 2, 3]

def test_task_difficulties():
    assert [t["difficulty"] for t in TASKS] == ["easy","medium","hard"]

def test_task_max_steps():
    steps = {t["id"]: t["max_steps"] for t in TASKS}
    assert steps == {1:1, 2:24, 3:168}


# ── get_prompt ────────────────────────────────────────────────────────

def test_prompt_is_string():
    for tid in [1,2,3]:
        p = get_prompt(tid)
        assert isinstance(p, str) and len(p) > 50

def test_prompt_contains_actions():
    for tid in [1,2,3]:
        p = get_prompt(tid)
        assert "0" in p and "1" in p and "2" in p

def test_prompt_mentions_task_name():
    assert "Single Decision"  in get_prompt(1)
    assert "24-Hour"          in get_prompt(2)
    assert "Storm"            in get_prompt(3)


# ── run_grader ────────────────────────────────────────────────────────

def test_grader_has_score_key():
    assert "score" in run_grader(1, [2])

def test_grader_score_in_range():
    for tid, actions in [(1,[2]),(2,[1]*24),(3,[0]*168)]:
        sc = run_grader(tid, actions)["score"]
        assert 0.0 <= sc <= 1.0, f"Task {tid} score out of range: {sc}"

def test_grader_task2_runs_24_steps():
    assert run_grader(2, [1]*24)["steps"] == 24

def test_grader_task3_runs_168_steps():
    assert run_grader(3, [0]*168)["steps"] == 168

def test_grader_short_actions_do_not_crash():
    r = run_grader(2, [1,2])   # only 2 actions for 24-step task
    assert r["steps"] == 24

def test_grader_empty_actions_do_not_crash():
    assert "score" in run_grader(1, [])

def test_grader_returns_revenue():
    r = run_grader(2, [2]*24)
    assert "revenue_rs" in r
    assert r["revenue_rs"] >= 0
