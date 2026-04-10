---
title: Smartgrid Openenv
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Smart Grid Energy Trader — OpenEnv

AI agent environment for renewable energy microgrid management.

## The Problem

A village microgrid runs on solar panels + a wind turbine.
Every hour the agent picks one action:

| Action | Code | What happens |
|--------|------|-------------|
| STORE | 0 | Charge the battery for later |
| DISTRIBUTE | 1 | Send power directly to homes |
| SELL | 2 | Sell to the city grid for money |

## Tasks

| # | Name | Difficulty | Steps |
|---|------|-----------|-------|
| 1 | Single Decision | Easy | 1 |
| 2 | 24-Hour Day Manager | Medium | 24 |
| 3 | 7-Day Storm Survival | Hard | 168 |

## State (what the agent sees)

```json
{
  "battery_level": 65.0,
  "solar_output":  12.4,
  "wind_output":    3.1,
  "grid_price":     8.5,
  "home_demand":    9.0,
  "hour_of_day":   14,
  "day":            2,
  "weather":      "sunny"
}