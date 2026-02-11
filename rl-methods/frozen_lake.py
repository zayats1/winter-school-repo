import gymnasium as gym
import numpy as np
import random
import time
import os
from datetime import datetime
from tqdm import tqdm

# Fix for macOS/Headless server crash
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
ENV_NAME = "FrozenLake-v1"
# ENV_NAME = "Taxi-v3"

IS_SLIPPERY = True

# HYPERPARAMETERS
TRAIN_EPISODES = 8000
MAX_STEPS = 50
LEARNING_RATE = 0.5         # High rate for deterministic FrozenLake
DISCOUNT_RATE = 0.95        # We care about the future reward (reaching the goal)
EPSILON_START = 1.0
EPSILON_DECAY = 0.01       # Slower decay: explores for ~1000 episodes
EPSILON_MIN = 0.01

def save_plots(rewards, env_name):
    """
    Generates and saves a graph of the training progress.
    """
    plt.figure(figsize=(10, 5))
    
    # 1. Plot raw rewards (cyan)
    plt.plot(rewards, color='cyan', alpha=0.3, label='Raw Reward')
    
    # 2. Moving Average (blue)
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, color='blue', linewidth=2, label='Moving Avg (50 eps)')

    plt.title(f"Agent Learning Progress ({env_name})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    folder_path = "rl-methods/experiments"
    os.makedirs(folder_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Fix: Dynamic filename based on environment name
    filename = f"{folder_path}/{env_name.lower()}_metrics_{timestamp}.png"
    
    plt.savefig(filename)
    print(f"📊 Metrics saved to {filename}")
    plt.close()

def watch_agent(qtable=None, delay=0.2):
    """
    Runs one episode visually. 
    """
    # Note: render_mode="human" allows us to see the game pop up
    env = gym.make(ENV_NAME, render_mode="human", is_slippery=IS_SLIPPERY) if ENV_NAME == "FrozenLake-v1" else gym.make(ENV_NAME, render_mode="human")
    
    state, info = env.reset()
    done = False
    total_reward = 0
    
    print("\n🎬 Simulation Started...")
    
    for step in range(MAX_STEPS):
        if qtable is None:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state, :])
            
        new_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = new_state
        
        time.sleep(delay)
        
        if done:
            break
            
    print(f"🏁 Episode finished. Total Score: {total_reward}\n")
    env.close()

def train_agent():
    """
    Trains the agent using Q-Learning.
    """
    # Note: We turn off slippery mode for easier debugging of logic
    if ENV_NAME == "FrozenLake-v1":
        env = gym.make(ENV_NAME, is_slippery=IS_SLIPPERY, render_mode=None)
    else:
        env = gym.make(ENV_NAME, render_mode=None)
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))
    
    epsilon = EPSILON_START
    rewards_history = [] 

    print(f"🔄 Training {ENV_NAME} for {TRAIN_EPISODES} episodes...")
    
    for _ in tqdm(range(TRAIN_EPISODES)):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        for _ in range(MAX_STEPS):
            # 1. Choose Action (Epsilon-Greedy)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore
            else:
                action = np.argmax(qtable[state, :]) # Exploit

            # 2. Execute Action
            new_state, reward, terminated, truncated, info = env.step(action)
            
            # 3. Update Q-Table (The Bellman Equation)
            current_q = qtable[state, action]
            max_future_q = np.max(qtable[new_state, :])
            
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q - current_q)
            qtable[state, action] = new_q
            
            state = new_state
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards_history.append(episode_reward)
        epsilon = max(EPSILON_MIN, epsilon - EPSILON_DECAY)
        
    env.close()
    save_plots(rewards_history, ENV_NAME)
    return qtable

if __name__ == "__main__":
    print(f"🧊 Q-Learning Setup: {ENV_NAME}")
    
    # 1. Watch Random (Fail)
    input("\n❌ Press [Enter] to watch UNTRAINED agent (Random moves)...")
    watch_agent(qtable=None, delay=0.1)
    
    # 2. Train
    input("💪 Press [Enter] to TRAIN...")
    trained_qtable = train_agent()
    print("✅ Training Complete!")

    # 3. Watch Intelligent (Success)
    while True:
        input("🏆 Press [Enter] to watch TRAINED agent...")
        watch_agent(qtable=trained_qtable, delay=0.3)