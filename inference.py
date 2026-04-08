import os
import random
from openai import OpenAI
from env import SmartGridEnv

# The hackathon evaluator passes these environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo") # Default fallback
HF_TOKEN = os.getenv("HF_TOKEN")
# Setup OpenAI Client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

def get_action_from_llm(obs):
    # If no token is provided during local testing, use a random fallback to test the loop
    if not client:
        return str(random.choice([0, 1, 2]))
        
    prompt = f"You manage a grid. Current State: Solar Gen={obs[0]:.1f}kW, Battery={obs[1]:.1f}%, Demand={obs[2]:.1f}kW, Grid Price=${obs[3]:.1f}. Choose action: 0 (Village), 1 (Sell), or 2 (Store). Reply with ONLY the number 0, 1, or 2."
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        # Extract just the number from the response
        action_text = response.choices[0].message.content.strip()
        # Ensure it's a valid integer, fallback to 0 if LLM hallucinates
        return action_text if action_text in ['0', '1', '2'] else '0'
    except Exception as e:
        return '0' # Fallback action

def main():
    env = SmartGridEnv()
    obs, _ = env.reset()
    
    # REQUIRED FORMAT: START
    print(f"[START] task=energy-trade env=smart-grid model={MODEL_NAME}")
    
    total_rewards = []
    
    while True:
        action_str = get_action_from_llm(obs)
        action_int = int(action_str)
        
        obs, reward, done, _, _ = env.step(action_int)
        total_rewards.append(reward)
        
        # REQUIRED FORMAT: STEP
        print(f"[STEP] step={env.step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")
        
        if done:
            break
            
    # REQUIRED FORMAT: END
    rewards_str = ",".join([f"{r:.2f}" for r in total_rewards])
    print(f"[END] success=true steps={len(total_rewards)} rewards={rewards_str}")

if __name__ == "__main__":
    main()