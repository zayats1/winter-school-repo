import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==========================================
# 0. CONFIGURATION (REFACTORED CONSTANTS)
# ==========================================
class SumoConfig:
    # Environment Physics
    ARENA_RADIUS = 250.0
    AGENT_RADIUS = 20.0
    FRICTION = 0.94
    REPEL_FORCE = 8.0    # Blows agents apart on collision
    SNAP_DISTANCE = 2.0  # Teleports agents apart slightly to prevent clipping
    
    # Stamina Logic
    MAX_STAMINA = 80.0
    AGENT_REGEN = 0.2
    BOT_REGEN = 0.6
    DRAIN_MULT = 3.2
    
    # Rewards (The "Suicide Prevention" Balance)
    WIN_REWARD = 100.0
    LOSS_PENALTY = -100.0   # High penalty prevents "rage-quitting"
    STALEMATE_BASE = -30.0   # Better than a loss, worse than a win
    PROXIMITY_WEIGHT = 15.0 # High weight rewards chasing the bot
    SURVIVAL_COST = 0.0

    # Training
    MAX_STEPS = 400
    N_ITERATIONS = 30
    N_SAMPLES = 100
    ELITE_PERCENT = 0.1
    EXPLORATION_NOISE = 0.89 # Prevents stagnation

# ==========================================
# 1. THE ENVIRONMENT
# ==========================================
class DungeonSumoEnv(gym.Env):
    metadata = {"render_modes": [None, "human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.screen_size = 600
        self.center = np.array([300.0, 300.0])
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # RANDOM START: Prevents the agent from learning a single static path
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.agent_pos = self.center + np.array([np.cos(angle)*120, np.sin(angle)*120])
        
        # Bot starts roughly opposite
        bot_angle = angle + self.np_random.uniform(np.pi * 0.8, np.pi * 1.2)
        self.bot_pos = self.center + np.array([np.cos(bot_angle)*120, np.sin(bot_angle)*120])
        
        self.agent_vel = np.zeros(2)
        self.bot_vel = np.zeros(2)
        self.agent_stamina = SumoConfig.MAX_STAMINA
        self.bot_stamina = SumoConfig.MAX_STAMINA
        return self._get_obs(), {}

    def _get_obs(self):
       # Normalize relative positions so 1.0 is the edge of the arena
        rel_agent = (self.agent_pos - self.center) / SumoConfig.ARENA_RADIUS
        rel_bot = (self.bot_pos - self.center) / SumoConfig.ARENA_RADIUS
        
        # Normalize velocities (Assuming max speed is around 15-20)
        norm_vel = self.agent_vel / 20.0 
        
        return np.concatenate([
            rel_agent,       # (2,)
            norm_vel,        # (2,)
            rel_bot,         # (2,)
            [self.agent_stamina / 100.0, self.bot_stamina / 100.0] # (2,)
        ], dtype=np.float32).flatten()

    def step(self, action):
        self.current_step += 1
        
        # 1. Efficiency & Movement
        agent_eff = 0.5 + 0.5 * (self.agent_stamina / SumoConfig.MAX_STAMINA)
        move_force = np.clip(action, -1, 1) * 1.2 * agent_eff
        self.agent_vel += move_force
        self.agent_stamina = max(0, self.agent_stamina - np.linalg.norm(move_force) * SumoConfig.DRAIN_MULT)

        # 3. Bot AI (Check if this is pushing you away!)
        diff = self.agent_pos - self.bot_pos
        dist_to_agent = np.linalg.norm(diff)
        
        if dist_to_agent > 0:
            bot_eff = 0.5 + 0.5 * (self.bot_stamina / SumoConfig.MAX_STAMINA)
            # Ensure the bot is pushing TOWARD you, not away
            # If 'diff' is (Agent - Bot), then bot_force should be +diff
            bot_force = (diff / dist_to_agent) * 0.35 * bot_eff
            self.bot_vel += bot_force

        # 3. Physics & Friction
        self.agent_stamina = min(SumoConfig.MAX_STAMINA, self.agent_stamina + SumoConfig.AGENT_REGEN)
        self.bot_stamina = min(SumoConfig.MAX_STAMINA, self.bot_stamina + SumoConfig.BOT_REGEN)
        self.agent_vel *= SumoConfig.FRICTION
        self.bot_vel *= SumoConfig.FRICTION
        self.agent_pos += self.agent_vel
        self.bot_pos += self.bot_vel

        # 4. Collision (Explosive repulsion)
        dist_col = np.linalg.norm(self.agent_pos - self.bot_pos)
        min_dist = SumoConfig.AGENT_RADIUS * 2
        if dist_col < min_dist:
            normal = (self.agent_pos - self.bot_pos) / (dist_col + 1e-5)
            self.agent_pos += normal * SumoConfig.SNAP_DISTANCE
            self.bot_pos -= normal * SumoConfig.SNAP_DISTANCE
            self.agent_vel += normal * SumoConfig.REPEL_FORCE
            self.bot_vel -= normal * SumoConfig.REPEL_FORCE

        # 5. Reward & Termination
        a_dist = np.linalg.norm(self.agent_pos - self.center)
        b_dist = np.linalg.norm(self.bot_pos - self.center)
        
        terminated = a_dist > SumoConfig.ARENA_RADIUS or b_dist > SumoConfig.ARENA_RADIUS
        truncated = self.current_step >= SumoConfig.MAX_STEPS

        if b_dist > SumoConfig.ARENA_RADIUS:
            reward = SumoConfig.WIN_REWARD
        elif a_dist > SumoConfig.ARENA_RADIUS:
            reward = SumoConfig.LOSS_PENALTY
        elif truncated:
            # Reward being closer to the enemy at the buzzer
            proximity = 1.0 - (dist_col / (SumoConfig.ARENA_RADIUS * 2))
            reward = SumoConfig.STALEMATE_BASE + (proximity * SumoConfig.PROXIMITY_WEIGHT)
        else:
            reward = SumoConfig.SURVIVAL_COST

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()
        self.screen.fill((30, 30, 30))
        pygame.draw.circle(self.screen, (60, 60, 60), self.center.astype(int), int(SumoConfig.ARENA_RADIUS))
        pygame.draw.circle(self.screen, (0, 200, 255), self.agent_pos.astype(int), int(SumoConfig.AGENT_RADIUS))
        pygame.draw.circle(self.screen, (255, 100, 0), self.bot_pos.astype(int), int(SumoConfig.AGENT_RADIUS))
        pygame.display.flip()
        self.clock.tick(60)

# ==========================================
# 2. TRAINING (CEM)
# ==========================================
def train_agent():
    envs = gym.vector.AsyncVectorEnv([lambda: DungeonSumoEnv() for _ in range(8)])
    obs_dim, act_dim = 8, 2
    
    mean_w = np.zeros((act_dim, obs_dim))
    std_w = np.ones((act_dim, obs_dim))
    mean_b = np.zeros(act_dim)
    std_b = np.ones(act_dim)

    for i in range(SumoConfig.N_ITERATIONS):
        batch_w = np.random.normal(mean_w, std_w, (SumoConfig.N_SAMPLES, act_dim, obs_dim))
        batch_b = np.random.normal(mean_b, std_b, (SumoConfig.N_SAMPLES, act_dim))
        rewards = np.zeros(SumoConfig.N_SAMPLES)


        # Process Vectorized Chunks
        for chunk_start in range(0, SumoConfig.N_SAMPLES, 8):
            curr_size = min(8, SumoConfig.N_SAMPLES - chunk_start) # Calculate actual size
            obs, _ = envs.reset()
            dones = np.zeros(8, dtype=bool)
            c_rewards = np.zeros(8)
            
            # Weights for just this chunk (e.g., 4 agents)
            w_c = batch_w[chunk_start:chunk_start + curr_size]
            b_c = batch_b[chunk_start:chunk_start + curr_size]

            while not np.all(dones[:curr_size]):
                # FIX: Slice 'obs' to match the number of weight sets 'w_c'
                # obs[:curr_size] ensures we only compute actions for active agents
                actions_batch = np.tanh(np.einsum('nij,nj->ni', w_c, obs[:curr_size]) + b_c)
                
                # We must pass 8 actions to envs.step(), so we pad with zeros
                full_actions = np.zeros((8, 2))
                full_actions[:curr_size] = actions_batch

                obs, r_step, term, trunc, _ = envs.step(full_actions)
                
                # Update rewards only for the indices in this chunk
                c_rewards[~dones] += r_step[~dones]
                dones |= (term | trunc)
            
            # Store the results back into the main rewards array
            rewards[chunk_start:chunk_start + curr_size] = c_rewards[:curr_size]

        # Elite selection
        n_elite = int(SumoConfig.N_SAMPLES * SumoConfig.ELITE_PERCENT)
        elites = np.argsort(rewards)[-n_elite:]
        mean_w, std_w = batch_w[elites].mean(axis=0), batch_w[elites].std(axis=0) + SumoConfig.EXPLORATION_NOISE
        mean_b, std_b = batch_b[elites].mean(axis=0), batch_b[elites].std(axis=0) + SumoConfig.EXPLORATION_NOISE

        print(f"Iter {i+1:02d} | Max: {np.max(rewards):>7.2f} | Avg: {np.mean(rewards):>7.2f}")

    envs.close()
    return mean_w, mean_b

if __name__ == "__main__":
    best_w, best_b = train_agent()
    
    # Final Visual Demo
    viz_env = DungeonSumoEnv(render_mode="human")
    for _ in range(5):
        obs, _ = viz_env.reset()
        done = False
        while not done:
            action = np.tanh(np.dot(best_w, obs) + best_b)
            obs, _, term, trunc, _ = viz_env.step(action)
            viz_env.render()
            done = term or trunc
    viz_env.close()
    
