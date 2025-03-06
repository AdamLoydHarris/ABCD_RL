import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
# ------------------------------
# Environment Definition
# ------------------------------
class GridMazeEnv(gym.Env):
    """
    A 3x3 grid maze where the agent must collect rewards in a cyclic order: A->B->C->D->A->...
    
    The observation is a 15-dimensional vector:
      - 9 dims: one-hot encoding of the current location
      - 4 dims: one-hot encoding of the previous action
      - 1 dim: binary flag indicating whether a reward was just received
      - 1 dim: buzzer flag that now turns on ONLY when the agent collects reward A
    """
    def __init__(self, reward_orders=None, training=True, fixed_reward_order=None, max_steps=200):
        super(GridMazeEnv, self).__init__()
        self.grid_size = 3
        self.num_cells = self.grid_size * self.grid_size
        self.max_steps = max_steps
        
        # Action space: 0=up, 1=down, 2=left, 3=right.
        self.action_space = spaces.Discrete(4)
        # Observation space: 15 dims (9 location, 4 past action, 1 reward flag, 1 buzzer).
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        
        # If no reward orders provided, generate 30 random ones.
        if reward_orders is None:
            self.all_reward_orders = []
            while len(self.all_reward_orders) < 30:
                order = random.sample(range(self.num_cells), 4)
                if order not in self.all_reward_orders:
                    self.all_reward_orders.append(order)
        else:
            self.all_reward_orders = reward_orders
        
        self.training = training
        self.fixed_reward_order = fixed_reward_order
        
        # Mapping for reward labels
        self.label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        
        self.reset()
    
    def reset(self):
        # Reset agent to a fixed starting position (e.g., bottom-right).
        self.agent_pos = [self.grid_size - 1, self.grid_size - 1]
        # current_goal_idx indicates which reward (0=A,1=B,2=C,3=D) is next.
        self.current_goal_idx = 0
        self.steps = 0
        
        # Previous action: zero vector (length 4).
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.last_reward = 0  # binary flag for reward
        # IMPORTANT: We no longer set the buzzer to 1 at the start of the episode.
        self.buzzer = 0.0
        
        # Choose the reward order (training: random from set, test: fixed).
        if self.training:
            self.reward_order = random.choice(self.all_reward_orders)
        else:
            if self.fixed_reward_order is not None:
                self.reward_order = self.fixed_reward_order
            else:
                self.reward_order = random.choice(self.all_reward_orders)
        
        return self._get_obs()
    
    def _get_obs(self):
        # One-hot encode current location
        cell = self.agent_pos[0] * self.grid_size + self.agent_pos[1]
        loc_one_hot = np.zeros(self.num_cells, dtype=np.float32)
        loc_one_hot[cell] = 1.0
        
        # Concatenate: location, previous action, reward flag, buzzer flag
        obs = np.concatenate([
            loc_one_hot,
            self.prev_action,
            np.array([self.last_reward], dtype=np.float32),
            np.array([self.buzzer], dtype=np.float32)
        ])
        return obs
    
    def step(self, action):
        self.steps += 1
        # Reset reward flag each step
        self.last_reward = 0
        
        # By default, buzzer is off each step unless we collect reward A below.
        self.buzzer = 0.0
        
        # Update previous action (one-hot).
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_action[action] = 1.0
        
        # Compute new position.
        row, col = self.agent_pos
        if action == 0:    # up
            new_row = max(row - 1, 0)
            new_col = col
        elif action == 1:  # down
            new_row = min(row + 1, self.grid_size - 1)
            new_col = col
        elif action == 2:  # left
            new_row = row
            new_col = max(col - 1, 0)
        elif action == 3:  # right
            new_row = row
            new_col = min(col + 1, self.grid_size - 1)
        else:
            new_row, new_col = row, col
        
        self.agent_pos = [new_row, new_col]
        cell = new_row * self.grid_size + new_col
        
        reward = 0.0
        reward_label = None
        
        # Check if agent is at the target
        if cell == self.reward_order[self.current_goal_idx]:
            reward = 1.0
            self.last_reward = 1
            reward_label = self.label_mapping[self.current_goal_idx]
            
            # Cycle: after D, go back to A
            self.current_goal_idx = (self.current_goal_idx + 1) % len(self.reward_order)
            
            # If we got reward A, turn buzzer on
            if reward_label == 'A':
                self.buzzer = 1.0
        
        # Episode ends when max steps reached
        done = self.steps >= self.max_steps
        info = {}
        if reward_label is not None:
            info['reward_label'] = reward_label
        
        return self._get_obs(), reward, done, info

# ------------------------------
# Actor-Critic Model with GRU
# ------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_actions=4):
        super(ActorCritic, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)  # Added fully connected layer
        self.relu = nn.ReLU()  # Added ReLU activation
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x, hidden):
        # x: (batch_size, input_size); add time dimension.
        x = x.unsqueeze(1)
        out, new_hidden = self.gru(x, hidden)
        out = out.squeeze(1)
        out = self.relu(self.fc(out))  # Pass through fully connected layer and ReLU
        action_logits = self.actor(out)
        state_value = self.critic(out)
        return action_logits, state_value, new_hidden
    
    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)

# ------------------------------
# Training Loop
# ------------------------------
def train_agent(env, model, optimizer, num_episodes=1000, gamma=0.99):
    model.train()
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        hidden = model.init_hidden(batch_size=1)
        done = False
        
        log_probs = []
        values = []
        rewards = []
        
        while not done:
            logits, value, hidden = model(obs_tensor, hidden)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            obs_next, reward, done, _ = env.step(action.item())
            obs_next_tensor = torch.from_numpy(obs_next).float().unsqueeze(0)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            obs_tensor = obs_next_tensor
        
        # Compute discounted returns.
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()
        values = torch.cat(values).squeeze()
        log_probs = torch.stack(log_probs)
        
        advantage = returns - values
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
    
    return episode_rewards

# ------------------------------
# Evaluation Function: Record Detailed Timepoint Data
# ------------------------------
def evaluate_agent(env, model):
    """
    Runs one test episode and returns a list of dictionaries (one per time step).
    Each dictionary contains:
      - time: current time step index.
      - activation: GRU hidden state (numpy array).
      - observation: the raw observation vector (15 dims).
      - location: decoded from the first 9 dims (index of active cell).
      - past_action: decoded from the next 4 dims (index; -1 if no action).
      - reward_flag: the binary reward flag (from obs).
      - buzzer: the buzzer flag (from obs).
      - reward: the reward received after the action.
      - reward_label: (if any) the label (A, B, C, or D).
    """
    model.eval()
    obs = env.reset()
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
    hidden = model.init_hidden(batch_size=1)
    done = False
    timepoint_data = []
    t = 0
    
    while not done:
        logits, value, hidden = model(obs_tensor, hidden)
        activation = hidden.detach().cpu().numpy()[0, 0, :].copy()
        
        # Decode observation:
        # Observation: [location (9), past_action (4), reward_flag (1), buzzer (1)]
        obs_np = obs.copy()
        location = int(np.argmax(obs_np[:9]))
        if np.sum(obs_np[9:13]) == 0:
            past_action = -1
        else:
            past_action = int(np.argmax(obs_np[9:13]))
        reward_flag = float(obs_np[13])
        buzzer = float(obs_np[14])
        
        current_data = {
            "time": t,
            "activation": activation,
            "observation": obs_np.copy(),
            "location": location,
            "past_action": past_action,
            "reward_flag": reward_flag,
            "buzzer": buzzer
        }
        
        action = torch.distributions.Categorical(logits=logits).sample()
        obs_next, reward, done, info = env.step(action.item())
        current_data["reward"] = reward
        current_data["reward_label"] = info.get('reward_label', None)
        
        timepoint_data.append(current_data)
        obs = obs_next
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        t += 1
    return timepoint_data

# ------------------------------
# Plotting Function for Test Sessions
# ------------------------------
def plot_test_session(activations, reward_events, unit_order=None, session_idx=0):
    """
    Plots a heatmap of GRU activations (rank-ordered if unit_order is provided)
    and overlays vertical lines for reward events.
    
    Reward colors: A - red, B - yellow, C - green, D - blue.
    """
    if unit_order is not None:
        activations = activations[:, unit_order]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(activations.T, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.xlabel("Time Step")
    plt.ylabel("GRU Unit (Rank-ordered)")
    plt.title(f"Test Session {session_idx+1} - GRU Activations")
    plt.colorbar(label='Activation')
    
    reward_colors = {'A': 'red', 'B': 'yellow', 'C': 'green', 'D': 'blue'}
    for t, label in reward_events:
        if label in reward_colors:
            plt.axvline(x=t, color=reward_colors[label], linestyle='--', linewidth=2)
    plt.show()

# ------------------------------
# Main Script
# ------------------------------
if __name__ == "__main__":
    # Training Setup:
    num_training_orders = 40
    training_reward_orders = []
    num_cells = 9
    while len(training_reward_orders) < num_training_orders:
        order = random.sample(range(num_cells), 4)
        if order not in training_reward_orders:
            training_reward_orders.append(order)

    with open("gru_outputs/training_tasks.txt", "w") as f:
        for order in training_reward_orders:
            f.write(f"{order}\n")
    
    train_env = GridMazeEnv(reward_orders=training_reward_orders, training=True, max_steps=300)
    model = ActorCritic(input_size=15, hidden_size=300, num_actions=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_episodes = 100_000
    episode_rewards = train_agent(train_env, model, optimizer, num_episodes=num_episodes, gamma=0.99)

    torch.save(model.state_dict(), "gru_outputs/gru_actor_critic_ABCD.pth")

    # Save episode rewards to a file.
    np.save("gru_outputs/episode_rewards.npy", episode_rewards)

    # Plot training reward over episodes.
    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, label='Total Reward', c='gray', linewidth=0.5)
    
    rolling_avg = np.convolve(episode_rewards, np.ones(200)/200, mode='valid')
    plt.plot(range(199, len(episode_rewards)), rolling_avg, color='red', label='Rolling Average (200 episodes)')
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward over Episodes (Training)")
    plt.legend()
    plt.grid(True)
    plt.savefig("gru_outputs/training_rewards_plot.svg")
    plt.show()
    # ------------------------------
    # Test Sessions on Unseen tasks
    # ------------------------------
    # Generate 5 unseen tasks (not in the training set).
    unseen_orders = []
    while len(unseen_orders) < 5:
        order = random.sample(range(num_cells), 4)
        if order not in training_reward_orders and order not in unseen_orders:
            unseen_orders.append(order)
    print("Unseen Test Reward Orders:", unseen_orders)
    
    with open("gru_outputs/test_tasks.txt", "w") as f:
        for order in unseen_orders:
            f.write(f"{order}\n")
    
    test_results = []
    # Run 5 test sessions.
    for idx, test_order in enumerate(unseen_orders):
        test_env = GridMazeEnv(reward_orders=training_reward_orders, training=False,
                               fixed_reward_order=test_order, max_steps=200)
        # Get detailed timepoint data.
        timepoint_data = evaluate_agent(test_env, model)
        test_results.append(timepoint_data)
    
    # For each session, extract activations and reward events.
    sessions_activations = []
    sessions_reward_events = []
    for session in test_results:
        activations = np.array([tp['activation'] for tp in session])
        reward_events = [(tp['time'], tp['reward_label']) for tp in session if tp['reward'] > 0]
        sessions_activations.append(activations)
        sessions_reward_events.append(reward_events)
    
    # Rank order GRU units by peak activation time in the first test session.
    first_activations = sessions_activations[0]  # shape: (T, hidden_size)
    peak_times = np.argmax(first_activations, axis=0)
    unit_order = np.argsort(peak_times)
    
    # Plot heatmaps for each test session using the same unit order.
    for idx, (acts, events) in enumerate(zip(sessions_activations, sessions_reward_events)):
        plot_test_session(acts, events, unit_order=unit_order, session_idx=idx)
    
    # Optionally, save the detailed test data for future analysis.

    with open("gru_outputs/test_sessions_data.pkl", "wb") as f:
        pickle.dump(test_results, f)
