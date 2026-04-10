"""
inference.py
============
Baseline LLM agent — runs all 3 tasks and prints scores.

REQUIRED environment variables:
    OPENAI_API_KEY   Your OpenAI API key
    MODEL_NAME       e.g. gpt-4.1-mini (optional)

Run:
    python inference.py
"""

import os, re, json, time, textwrap
from typing import List
from openai import OpenAI
from environment import SmartGridEnv
from tasks import TASKS, get_prompt
from dataset_utils import dataset_agent

# ── config ───────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "gpt-4.1-mini"
API_KEY      = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not set. Please set it before running.")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM = textwrap.dedent("""
You manage a renewable energy microgrid.
Each step you get the current state as JSON.

Respond ONLY with:
{"action": 0}  or  {"action": 1}  or  {"action": 2}

0 = STORE
1 = DISTRIBUTE
2 = SELL

No explanation. No extra text.
""").strip()

# ── helpers ───────────────────────────────────────────────────────────

def parse_action(text: str) -> int:
    """Robust parsing of model output"""
    try:
        data = json.loads(text.strip())
        action = int(data.get("action", 1))
        return max(0, min(2, action))
    except Exception:
        # fallback regex
        m = re.search(r'\b([012])\b', text or "")
        return int(m.group(1)) if m else 1


def ask(step: int, state: dict, history: List[str], task_id: int) -> int:
    """Call LLM with retry + fallback"""

    hist_text = "\n".join(history[-3:]) or "None"
    user_msg = (
        f"Step {step}\n\n"
        f"History:\n{hist_text}\n\n"
        f"Current state:\n{json.dumps(state)}"
    )

    for attempt in range(3):  # retry logic
        try:
            rsp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=20,
            )

            text = rsp.choices[0].message.content or ""
            return parse_action(text)

        except Exception as e:
            if attempt == 2:
                print(f"    [LLM failed → fallback] {e}")
                return dataset_agent(state, task_id)
            time.sleep(1)  # wait before retry

    return 1  # safety fallback


# ── run one task ─────────────────────────────────────────────────────

def run_task(task_id: int) -> dict:
    task = next(t for t in TASKS if t["id"] == task_id)

    print(f"\n{'='*52}")
    print(f"  Task {task_id}: {task['name']} [{task['difficulty'].upper()}]")
    print(f"{'='*52}")

    env = SmartGridEnv(task_id=task_id, seed=42)
    state = env.reset()
    history: List[str] = []
    step = 0

    while not state.get("done"):
        step += 1

        action = ask(step, state, history, task_id)
        action_name = ["STORE", "DISTRIBUTE", "SELL"][action]

        result = env.step(action)
        state = result["observation"]
        reward = result["reward"]
        info = result["info"]

        if step % 6 == 1 or task_id == 1:
            print(
                f"  [{step:3d}] Day{info.get('day',1)} "
                f"Hr{info.get('hour',0):02d} | "
                f"{action_name:10s} | "
                f"Bat:{info.get('battery_level',0):5.1f}% | "
                f"Reward:{reward:+.3f}"
            )

        history.append(
            f"Step {step}: {action_name} → reward {reward:+.3f}, "
            f"battery {info.get('battery_level',0):.1f}%"
        )

        if result["done"]:
            break

        time.sleep(0.01)

    sc = env.score()

    print(
        f"\n  Score: {sc:.3f} | Revenue: Rs{env.total_revenue:.0f} "
        f"| Shortage: {env.total_shortage:.1f} | Steps: {step}"
    )

    return {
        "task_id": task_id,
        "name": task["name"],
        "difficulty": task["difficulty"],
        "score": sc,
        "steps": step,
    }


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*52)
    print("  SMART GRID ENERGY TRADER — inference.py")
    print("="*52)
    print(f"  Model  : {MODEL_NAME}")
    print(f"  API URL: {API_BASE_URL}")

    results = [run_task(tid) for tid in [1, 2, 3]]

    print("\n" + "="*52)
    print("  FINAL SCORES")
    print("="*52)

    for r in results:
        print(f"  Task {r['task_id']} {r['name']:<24} {r['score']:.3f}")

    avg = sum(r["score"] for r in results) / 3

    print(f"  {'─'*40}")
    print(f"  Average score {avg:.3f}")
    print("="*52)

    with open("inference_results.json", "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "results": results,
                "average_score": round(avg, 3),
            },
            f,
            indent=2,
        )

    print("\n  Saved → inference_results.json")


if __name__ == "__main__":
    main()