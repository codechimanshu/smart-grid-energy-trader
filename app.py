"""
app.py  —  Smart Grid Energy Trader  (with dataset endpoints)
=============================================================
Endpoints:
  GET  /health          → health check (judges ping this)
  GET  /                → frontend dashboard
  POST /reset           → reset environment
  POST /step            → take action
  GET  /state           → current state
  GET  /tasks           → list all 3 tasks
  POST /grade           → run grader, return score 0.0-1.0

  Dataset endpoints (bonus — show judges you used the data):
  GET  /hint            → smart action hint from dataset
  GET  /forecast        → price forecast for next N hours
  GET  /benchmark       → compare vs dataset baselines
  GET  /dataset-stats   → dataset summary stats
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn, os

from environment  import SmartGridEnv
from tasks        import TASKS, get_prompt, run_grader
from dataset_utils import (get_smart_hint, forecast_prices,
                            should_store_now, get_benchmark,
                            get_dataset_stats, build_fewshot_prompt)

app = FastAPI(
    title="Smart Grid Energy Trader",
    description="OpenEnv — renewable energy microgrid with dataset-powered hints",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

_envs = {1: SmartGridEnv(1), 2: SmartGridEnv(2), 3: SmartGridEnv(3)}


# ── Request models ────────────────────────────────────────────────

class ResetReq(BaseModel):
    task_id: int = 1
    seed: Optional[int] = 42

class StepReq(BaseModel):
    task_id: int = 1
    action: int

class GradeReq(BaseModel):
    task_id: int
    actions: List[int]


# ── Core endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "SmartGridEnv", "version": "1.0.0"}

@app.get("/")
def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"project": "Smart Grid Energy Trader", "docs": "/docs"}

@app.post("/reset")
def reset(req: ResetReq = None):
    tid  = req.task_id if req else 1
    seed = req.seed    if req else 42
    if tid not in _envs:
        raise HTTPException(400, "task_id must be 1, 2, or 3")
    _envs[tid] = SmartGridEnv(task_id=tid, seed=seed)
    obs = _envs[tid].reset()
    return {"observation": obs, "task_prompt": get_prompt(tid)}

@app.post("/step")
def step(req: StepReq):
    if req.task_id not in _envs:
        raise HTTPException(400, "task_id must be 1, 2, or 3")
    if req.action not in (0, 1, 2):
        raise HTTPException(400, "action must be 0, 1, or 2")
    return _envs[req.task_id].step(req.action)

@app.get("/state")
def get_state(task_id: int = 1):
    if task_id not in _envs:
        raise HTTPException(400, "task_id must be 1, 2, or 3")
    return _envs[task_id].state()

@app.get("/tasks")
def list_tasks():
    return {"tasks": [{**t, "prompt": get_prompt(t["id"])} for t in TASKS]}

@app.post("/grade")
def grade(req: GradeReq):
    if req.task_id not in (1, 2, 3):
        raise HTTPException(400, "task_id must be 1, 2, or 3")
    return run_grader(req.task_id, req.actions)

@app.get("/openenv.yaml", response_class=PlainTextResponse)
def yaml_spec():
    with open("openenv.yaml") as f:
        return PlainTextResponse(f.read(), media_type="text/yaml")


# ── Dataset endpoints ─────────────────────────────────────────────

@app.get("/hint")
def hint(task_id: int = 1):
    """
    Get a dataset-powered action hint for the current state.
    Uses episode history to find the most similar past situation
    and recommend what the optimal agent did there.
    """
    state = _envs[task_id].state()
    h     = get_smart_hint(state)
    fc    = should_store_now(state, task_id)
    return {
        "current_state":  state,
        "hint":           h,
        "price_forecast": fc,
        "recommended_action": h['best_action'],
        "recommended_name":   h['best_action_name'],
    }

@app.get("/forecast")
def forecast(task_id: int = 1, hours_ahead: int = 6):
    """
    Price forecast for the next N hours using dataset price patterns.
    Helps agents decide whether to store now and sell later.
    """
    state = _envs[task_id].state()
    fc    = forecast_prices(
                current_hour=state['hour_of_day'],
                task_id=task_id,
                day=state['day'],
                hours_ahead=hours_ahead
            )
    store = should_store_now(state, task_id)
    return {
        "current_hour":  state['hour_of_day'],
        "current_price": state['grid_price'],
        "forecast":      fc,
        "store_advice":  store,
    }

@app.get("/benchmark")
def benchmark(task_id: int = 1):
    """
    Compare agent performance vs dataset baselines.
    Shows rule-based vs optimal agent reward averages from the dataset.
    """
    return get_benchmark(task_id)

@app.get("/dataset-stats")
def dataset_stats():
    """
    Full statistics about the dataset — row counts, price ranges, benchmarks.
    """
    return get_dataset_stats()

@app.get("/fewshot-prompt")
def fewshot_prompt(task_id: int = 1, n_examples: int = 2):
    """
    Returns a few-shot LLM system prompt built from similar dataset examples.
    Use this instead of the plain system prompt for better LLM accuracy.
    """
    state  = _envs[task_id].state()
    prompt = build_fewshot_prompt(state, task_id, n_examples)
    return {"system_prompt": prompt, "n_examples": n_examples}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
