"""
tasks.py
========
Defines the 3 tasks and their graders.
"""

from environment import SmartGridEnv

TASKS = [
    {
        "id": 1,
        "name": "Single Decision",
        "difficulty": "easy",
        "max_steps": 1,
        "description": (
            "Make ONE decision. Grid price is shown in state. "
            "If grid_price >= 8 → SELL (action 2) earns most revenue. "
            "If grid_price < 5  → STORE (action 0) saves energy cheaply. "
            "Otherwise          → DISTRIBUTE (action 1) serves homes. "
            "Respond with ONLY: {\"action\": <0|1|2>}"
        ),
    },
    {
        "id": 2,
        "name": "24-Hour Day Manager",
        "difficulty": "medium",
        "max_steps": 24,
        "description": (
            "Manage the microgrid for 24 hours (one step per hour). "
            "Grid price peaks at 17:00–20:00 (up to 12 Rs/kWh). "
            "Solar only generates between 06:00 and 18:00. "
            "Home demand peaks morning (07–09) and evening (18–21). "
            "Strategy: STORE at cheap hours, SELL at peak price, "
            "DISTRIBUTE when demand is high and price is low. "
            "Respond with ONLY: {\"action\": <0|1|2>}"
        ),
    },
    {
        "id": 3,
        "name": "7-Day Storm Survival",
        "difficulty": "hard",
        "max_steps": 168,
        "description": (
            "Manage the microgrid for 7 days (168 hours). "
            "IMPORTANT: Days 4 and 5 have a STORM — zero solar, only wind. "
            "You MUST pre-fill battery to ~100% by end of Day 3. "
            "Day 6 has a 2.5x price SPIKE — sell aggressively that day. "
            "Score = storm survival 40% + revenue 40% + efficiency 20%. "
            "Respond with ONLY: {\"action\": <0|1|2>}"
        ),
    },
]


def get_prompt(task_id: int) -> str:
    task = next(t for t in TASKS if t["id"] == task_id)
    return (
        f"You are an AI agent managing a renewable energy microgrid.\n\n"
        f"TASK {task['id']} ({task['difficulty'].upper()}): {task['name']}\n\n"
        f"{task['description']}\n\n"
        f"Actions: 0=STORE  1=DISTRIBUTE  2=SELL\n"
        f"Reply with ONLY a JSON object. Example: {{\"action\": 2}}"
    )


def run_grader(task_id: int, actions: list) -> dict:
    """
    Run a full episode with the given action list.
    Returns score 0.0–1.0 plus breakdown.
    """
    env = SmartGridEnv(task_id=task_id, seed=42)
    env.reset()
    log = []
    i   = 0
    while not env.done:
        a = actions[i] if i < len(actions) else 1
        r = env.step(a)
        log.append({"step": i+1,
                    "action": r["info"].get("action"),
                    "reward": r["reward"],
                    "battery": r["info"].get("battery_level")})
        i += 1

    return {
        "task_id":    task_id,
        "score":      env.score(),
        "revenue_rs": round(env.total_revenue, 2),
        "shortage":   round(env.total_shortage, 2),
        "waste":      round(env.total_waste, 2),
        "steps":      env.steps_done,
        "log":        log[:5],   # first 5 steps for judges
    }
