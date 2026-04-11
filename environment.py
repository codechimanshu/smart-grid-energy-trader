"""
environment.py
==============
Smart Grid Energy Trader — OpenEnv Environment

The agent manages a small village microgrid powered by solar + wind.
Every hour it picks ONE action:
    0 = STORE      → charge the battery
    1 = DISTRIBUTE → send power to homes
    2 = SELL       → sell to the city grid for money

State the agent sees every step:
    battery_level  : 0–100 %
    solar_output   : kWh from solar panels this hour
    wind_output    : kWh from wind turbine this hour
    grid_price     : rupees per kWh the city will pay right now
    home_demand    : kWh that local homes need this hour
    hour_of_day    : 0–23
    day            : 1–7
    weather        : sunny / cloudy / stormy
"""

import random
import math


class SmartGridEnv:

    MAX_BATTERY = 100.0   # kWh total battery capacity

    def __init__(self, task_id: int = 1, seed: int = 42):
        self.task_id = task_id
        self.seed    = seed
        self.reset()

    # ── public API ──────────────────────────────────────────────────

    def reset(self):
        self._rng         = random.Random(self.seed)
        self.hour         = 0
        self.day          = 1
        self.battery      = 50.0
        self.total_revenue  = 0.0
        self.total_shortage = 0.0
        self.total_waste    = 0.0
        self.steps_done   = 0
        self.done         = False
        self._history     = []

        if self.task_id == 3:
            self._week_weather = {
                1: "sunny", 2: "sunny", 3: "cloudy",
                4: "stormy", 5: "stormy",   # challenge days
                6: "sunny",  7: "sunny",
            }
        return self.state()

    def step(self, action: int):
        if self.done:
            return {"observation": self.state(), "reward": 0.0,
                    "done": True, "info": {"msg": "call reset() first"}}

        if action not in (0, 1, 2):
            action = 1  # safe default

        obs    = self.state()
        gen    = obs["solar_output"] + obs["wind_output"]
        demand = obs["home_demand"]
        price  = obs["grid_price"]
        reward = 0.0
        info   = {"action": ["STORE","DISTRIBUTE","SELL"][action],
                  "day": self.day, "hour": self.hour}

        if action == 0:   # STORE
            space   = self.MAX_BATTERY - self.battery
            stored  = min(gen, space)
            waste   = gen - stored
            self.battery += stored
            self.total_waste += waste * 0.05
            # still need to serve homes from battery
            serve   = min(demand, self.battery)
            self.battery -= serve
            short   = max(0.0, demand - serve)
            self.total_shortage += short
            reward  = -(waste * 0.05) - (short * 0.3)

        elif action == 1:  # DISTRIBUTE
            served  = min(gen, demand)
            surplus = gen - served
            short   = max(0.0, demand - served)
            space   = self.MAX_BATTERY - self.battery
            stored  = min(surplus, space)
            waste   = surplus - stored
            self.battery += stored
            self.total_waste    += waste * 0.02
            self.total_shortage += short
            reward  = -(short * 0.3) - (waste * 0.02)

        else:              # SELL
            revenue = gen * price
            self.total_revenue += revenue
            serve   = min(demand, self.battery)
            self.battery -= serve
            short   = max(0.0, demand - serve)
            self.total_shortage += short * 0.5
            reward  = (revenue / 100.0) - (short * 0.5)

        # clamp battery
        self.battery = max(0.0, min(self.MAX_BATTERY, self.battery))

        # small bonus for healthy battery (30–80%)
        if 30 <= self.battery <= 80:
            reward += 0.02

        self.steps_done += 1
        self._history.append({"action": info["action"], "reward": round(reward,4)})

        # advance clock
        self.hour += 1
        if self.hour == 24:
            self.hour = 0
            self.day += 1

        if self.steps_done >= self._max_steps():
            self.done = True

        info["battery_level"]  = round(self.battery, 1)
        info["total_revenue"]  = round(self.total_revenue, 2)
        return {"observation": self.state(),
                "reward":      round(reward, 4),
                "done":        self.done,
                "info":        info}

    def state(self):
        solar, wind = self._gen(self.hour, self.day)
        return {
            "battery_level": round(self.battery, 2),
            "solar_output":  round(solar, 2),
            "wind_output":   round(wind, 2),
            "grid_price":    round(self._price(self.hour, self.day), 2),
            "home_demand":   round(self._demand(self.hour), 2),
            "hour_of_day":   self.hour,
            "day":           self.day,
            "weather":       self._weather(self.day),
            "done":          self.done,
            "task_id":       self.task_id,
        }

    def score(self):
        """Returns bounds-clamped score strictly between 0 and 1 after episode ends."""
        if not self.done:
            return 0.001

        raw = 0.001
        if self.task_id == 1:
            if self._history:
                act = self._history[-1]["action"]
                p   = self._price(0, 1)
                if p >= 8.0 and act == "SELL":        raw = 0.999
                elif p < 5.0  and act == "STORE":     raw = 0.999
                elif act == "DISTRIBUTE":             raw = 0.5
                else:                                 raw = 0.2

        elif self.task_id == 2:
            max_rev = 24 * 15.0 * 12.0 * 0.3
            rev_s   = min(1.0, self.total_revenue / max_rev)
            short_s = min(1.0, self.total_shortage / 50.0)
            raw = round(max(0.0, rev_s - short_s * 0.5), 3)

        else:   # task 3
            survived = 1.0 if self.total_shortage < 10.0 \
                       else max(0.0, 1 - self.total_shortage / 100)
            rev_s    = min(1.0, self.total_revenue / 5000.0)
            waste_s  = 1.0 - min(1.0, self.total_waste / 20.0)
            raw = round(survived * 0.4 + rev_s * 0.4 + waste_s * 0.2, 3)

        return max(0.001, min(0.999, float(raw)))

    # ── internals ────────────────────────────────────────────────────

    def _max_steps(self):
        return {1: 1, 2: 24, 3: 168}[self.task_id]

    def _weather(self, day):
        if self.task_id == 3:
            return self._week_weather.get(day, "sunny")
        return self._rng.choice(["sunny","sunny","sunny","cloudy","cloudy"])

    def _gen(self, hour, day):
        w = self._weather(day)
        if w == "stormy":
            solar = 0.0
            wind  = self._rng.uniform(8, 14)
        elif w == "cloudy":
            curve = max(0, math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            solar = curve * 15.0 * 0.3 + self._rng.uniform(0, 0.3)
            wind  = self._rng.uniform(4, 8)
        else:
            curve = max(0, math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            solar = curve * 15.0 + self._rng.uniform(0, 0.5)
            wind  = self._rng.uniform(2, 6)
        return round(solar, 2), round(wind, 2)

    def _demand(self, hour):
        if 7 <= hour <= 9 or 18 <= hour <= 21:
            base = 12.0
        elif hour <= 5:
            base = 4.0
        else:
            base = 7.0
        return round(base + self._rng.uniform(-1, 1), 2)

    def _price(self, hour, day):
        if   hour < 6:             base = 3.0
        elif hour < 10:            base = 6.0
        elif hour < 17:            base = 5.0
        elif hour < 21:            base = 12.0   # evening peak
        else:                      base = 7.0
        if self.task_id == 3 and day == 6:
            base *= 2.5            # price spike on day 6
        return round(base + self._rng.uniform(-0.5, 0.5), 2)
