import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

# Fix for macOS crash (zsh: trace trap)
import matplotlib
matplotlib.use('Agg') # Must be before importing pyplot
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

ENV_NAME = "Taxi-v3"
TRAIN_EPISODES = 2000
MAX_STEPS = 100
LEARNING_RATE = 0.7
DISCOUNT_RATE = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.005
EPSILON_MIN = 0.01

def save_plots(rewards):
    """
    Generates and saves a graph of the training progress.
    """
    plt.figure(figsize=(10, 5))
    
    # 1. Plot raw rewards (light blue)
    plt.plot(rewards, color='cyan', alpha=0.3, label='Raw Reward')
    
    # 2. Calculate and plot Moving Average (dark blue)
    # This smooths out the noise to show the actual learning trend
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, color='blue', linewidth=2, label='Moving Avg (50 eps)')

    plt.title(f"Agent Learning Progress ({ENV_NAME})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 1. Define the directory path
    folder_path = "rl-methods/experiments"
    
    # 2. Create the directory if it doesn't exist (CRITICAL STEP)
    os.makedirs(folder_path, exist_ok=True)
    
    # 3. Generate a clean timestamp (YearMonthDay_HourMinuteSecond)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 4. Save with the new dynamic filename
    filename = f"{folder_path}/taxi_q-learning_metrics_{timestamp}.png"
    
    plt.savefig(filename)
    print(f"📊 Metrics saved to {filename}")
    plt.close()

def watch_agent(qtable=None, delay=0.1):
    """
    Runs one episode visually. 
    """
    env = gym.make(ENV_NAME, render_mode="human")
    state, info = env.reset()
    done = False
    total_reward = 0
    
    print("\n🎬 Simulation Started...")
    
    for step in range(MAX_STEPS):
        if qtable is None:
            action = env.action_space.sample() # Untrained
        else:
            action = np.argmax(qtable[state, :]) # Trained
            
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
    Trains the agent and tracks rewards.
    """
    env = gym.make(ENV_NAME, render_mode=None)
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))
    
    epsilon = EPSILON_START
    rewards_history = []  # 📉 To store metrics

    print(f"🔄 Training for {TRAIN_EPISODES} episodes...")
    
    for _ in tqdm(range(TRAIN_EPISODES)):
        state, info = env.reset()
        done = False
        episode_reward = 0  # Track score for this specific game
        
        for _ in range(MAX_STEPS):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)
            
            # Q-Learning Update
            current_q = qtable[state, action]
            max_future_q = np.max(qtable[new_state, :])
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q - current_q)
            qtable[state, action] = new_q
            
            state = new_state
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # End of Episode: Save metric and decay epsilon
        rewards_history.append(episode_reward)
        epsilon = max(EPSILON_MIN, epsilon - EPSILON_DECAY)
        
    env.close()
    
    # Save the plot now that training is done
    save_plots(rewards_history)
    
    return qtable

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"🚕 Q-Learning with Metrics ({ENV_NAME})")
    
    # 1. Watch Random
    input("\n❌ Press [Enter] to watch UNTRAINED agent...")
    watch_agent(qtable=None, delay=0.1)
    
    # 2. Train & Plot
    input("💪 Press [Enter] to TRAIN and generate plots...")
    trained_qtable = train_agent()
    print("✅ Training Complete! Check 'training_metrics.png'.")

    # 3. Watch Trained
    while True:
        input("🏆 Press [Enter] to watch TRAINED agent...")
        watch_agent(qtable=trained_qtable, delay=0.2)