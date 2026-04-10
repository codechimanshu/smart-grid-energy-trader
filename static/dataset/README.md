# Smart Grid Energy Trader — Dataset

Complete training and evaluation dataset for the Smart Grid OpenEnv environment.

## Files

| File | Rows | Description |
|------|------|-------------|
| `states.csv` | 216 | All environment states across all tasks |
| `episodes.csv` | 3,860 | Full episode rollouts with agent decisions |
| `price_patterns.csv` | 96 | 24-hour price curves for 4 market scenarios |
| `llm_prompts.jsonl` | 500 | LLM training examples in OpenAI chat format |

## states.csv columns

| Column | Type | Description |
|--------|------|-------------|
| task_id | int | 1=easy, 2=medium, 3=hard |
| day | int | Day number (1–7) |
| hour | int | Hour (0–23) |
| weather | str | sunny / cloudy / stormy |
| solar_output | float | kWh from solar this hour |
| wind_output | float | kWh from wind this hour |
| generation | float | Total kWh generated |
| grid_price | float | Rs per kWh |
| home_demand | float | kWh homes need |
| price_zone | str | peak / mid / off_peak |
| is_storm_day | int | 1 if stormy (Task 3 Days 4–5) |
| is_spike_day | int | 1 if price spike (Task 3 Day 6) |

## episodes.csv columns

All state columns above, plus:

| Column | Type | Description |
|--------|------|-------------|
| episode_id | int | Unique episode identifier |
| seed | int | Random seed used |
| battery_level | float | Battery % at this step |
| rule_action | int | Rule-based agent decision |
| rule_action_name | str | STORE / DISTRIBUTE / SELL |
| rule_reward | float | Reward from rule-based action |
| optimal_action | int | Optimal agent decision |
| optimal_action_name | str | STORE / DISTRIBUTE / SELL |
| optimal_reward | float | Reward from optimal action |
| reward_gap | float | How much better optimal was |

## llm_prompts.jsonl format

```json
{
  "messages": [
    {"role": "system",    "content": "You manage a microgrid..."},
    {"role": "user",      "content": "Current state:\n{...}"},
    {"role": "assistant", "content": "{\"action\": 2}"}
  ],
  "optimal_action": 2,
  "optimal_action_name": "SELL",
  "reason": "Grid price is high (11.5 Rs/kWh). Sell to maximize revenue.",
  "task_id": 2
}
```

## Regenerate

```bash
python generate_dataset.py
```
