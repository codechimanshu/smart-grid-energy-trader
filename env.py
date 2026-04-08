import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SmartGridEnv(gym.Env):
    def __init__(self):
        super(SmartGridEnv, self).__init__()
        # Actions: 0=Power Village, 1=Sell to Grid, 2=Store in Battery
        self.action_space = spaces.Discrete(3)
        # Observation: [Solar_Generation, Battery_Level, Village_Demand, Grid_Price]
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.state = None
        self.step_count = 0
        self.max_steps = 24 # Simulate one full day (24 hours)

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initial state: 10kW Solar, 50% Battery, 8kW Demand, $5/kW Price
        self.state = np.array([10.0, 50.0, 8.0, 5.0], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        gen, battery, demand, price = self.state
        reward = 0
        done = False
        
        # Calculate action results
        if action == 0:  # Power Village
            net = gen - demand
            if net < 0 and battery < abs(net): # Total Blackout
                reward = -20.0 
                battery = 0
            elif net < 0: # Use battery to prevent blackout
                reward = 5.0
                battery += net
            else: # Powered purely by solar
                reward = 10.0
                battery = min(100, battery + net)
                
        elif action == 1:  # Sell to Grid
            reward = price * 2.0  # Profit multiplier
            battery = max(0, battery - 10) # Drain battery to sell more
            
        elif action == 2:  # Store Energy
            battery = min(100, battery + gen)
            reward = 2.0 # Small reward for safe behavior
        
        self.step_count += 1
        if self.step_count >= self.max_steps: 
            done = True
            
        # Simulate next hour (randomized for complexity)
        next_gen = np.clip(np.random.normal(10, 3), 0, 20)
        next_demand = np.clip(np.random.normal(8, 2), 2, 15)
        next_price = np.clip(np.random.normal(5, 4), 1, 15)
        
        self.state = np.array([next_gen, battery, next_demand, next_price], dtype=np.float32)
        return self.state, reward, done, False, {}