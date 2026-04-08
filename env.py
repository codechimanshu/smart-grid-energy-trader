import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SmartGridEnv(gym.Env):
    def __init__(self):
        super(SmartGridEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.state = None
        self.step_count = 0
        self.max_steps = 24
        self.current_task = "normal-day"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "task" in options:
            self.current_task = options["task"]
            
        # Change initial conditions based on the scenario
        if self.current_task == "summer-peak":
            self.state = np.array([15.0, 50.0, 10.0, 8.0], dtype=np.float32) # High Gen, High Demand
        elif self.current_task == "winter-storm":
            self.state = np.array([2.0, 20.0, 12.0, 15.0], dtype=np.float32) # Low Gen, Very High Demand
        else: # mild-spring
            self.state = np.array([10.0, 60.0, 6.0, 4.0], dtype=np.float32) # Easy day
            
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        gen, battery, demand, price = self.state
        reward = 0
        done = False
        
        if action == 0:
            net = gen - demand
            if net < 0 and battery < abs(net):
                reward = -20.0 
                battery = 0
            elif net < 0:
                reward = 5.0
                battery += net
            else:
                reward = 10.0
                battery = min(100, battery + net)
        elif action == 1:
            reward = price * 2.0
            battery = max(0, battery - 10)
        elif action == 2:
            battery = min(100, battery + gen)
            reward = 2.0
        
        self.step_count += 1
        if self.step_count >= self.max_steps: 
            done = True
            
        # Make the winter storm much harder
        if self.current_task == "winter-storm":
            next_gen = np.clip(np.random.normal(3, 2), 0, 10)
        else:
            next_gen = np.clip(np.random.normal(10, 3), 0, 20)
            
        next_demand = np.clip(np.random.normal(8, 2), 2, 15)
        next_price = np.clip(np.random.normal(5, 4), 1, 15)
        
        self.state = np.array([next_gen, battery, next_demand, next_price], dtype=np.float32)
        return self.state, reward, done, False, {}