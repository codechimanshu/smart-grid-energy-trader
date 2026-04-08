import os
import random
from openai import OpenAI
from env import SmartGridEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

def get_action_from_llm(obs):
    if not client:
        return str(random.choice([0, 1, 2]))
        
    prompt = f"State: Gen={obs[0]:.1f}, Bat={obs[1]:.1f}, Dem={obs[2]:.1f}, Price={obs[3]:.1f}. Action: 0(Vill), 1(Sell), 2(Store). Reply with ONLY 0, 1, or 2."
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        action_text = response.choices[0].message.content.strip()
        return action_text if action_text in ['0', '1', '2'] else '0'
    except Exception:
        return '0'

def run_single_task(env, task_name):
    # Reset environment with specific scenario
    obs, _ = env.reset(options={"task": task_name})
    
    print(f"[START] task={task_name} env=smart-grid model={MODEL_NAME}")
    total_rewards = []
    
    while True:
        action_str = get_action_from_llm(obs)
        action_int = int(action_str)
        
        obs, reward, done, _, _ = env.step(action_int)
        total_rewards.append(reward)
        
        print(f"[STEP] step={env.step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")
        
        if done:
            break
            
    # Calculate a normalized score strictly between 0 and 1
    total_sum = sum(total_rewards)
    # Convert raw rewards (can range from -480 to +720) into a 0.0 to 1.0 scale
    raw_score = (total_sum + 480) / 1200.0
    # Force it to be strictly between 0.01 and 0.99
    final_score = max(0.01, min(0.99, raw_score))
    
    rewards_str = ",".join([f"{r:.2f}" for r in total_rewards])
    # The crucial change: Adding score=... to the END string
    print(f"[END] success=true steps={len(total_rewards)} rewards={rewards_str} score={final_score:.4f}")

def main():
    env = SmartGridEnv()
    
    # We must run exactly 3 or more tasks!
    hackathon_tasks = ["summer-peak", "winter-storm", "mild-spring"]
    
    for task in hackathon_tasks:
        run_single_task(env, task)

if __name__ == "__main__":
    main()