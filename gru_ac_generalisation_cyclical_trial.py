import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------------
# Environment Definition
# ------------------------------
class GridMazeEnv(gym.Env):
    """
    A 3x3 grid maze where the agent must collect rewards in a cyclic order.
    The rewards are given according to a reward order (a list of 4 unique cell indices)
    that the agent must follow in order: A, B, C, D, and then cycle back to A.
    
    The observation is now a 15-dimensional vector:
      - 9 dims: one-hot encoding of the current location.
      - 4 dims: one-hot encoding of the previous action.
      - 1 dim: binary flag indicating whether a reward was just received.
      - 1 dim: buzzer flag indicating the start of a new trial.
      
    During training, the environment randomly picks one of a set of reward orders.
    At test time, a fixed (unseen) reward order is provided.
    """
    def __init__(self, reward_orders=None, training=True, fixed_reward_order=None, max_steps=200):
        super(GridMazeEnv, self).__init__()
        self.grid_size = 3            # 3x3 grid; cells 0-8 (row-major)
        self.num_cells = self.grid_size * self.grid_size
        self.max_steps = max_steps    # Maximum steps per episode (e.g. simulating 20 minutes)
        
        # Action space: 0=up, 1=down, 2=left, 3=right.
        self.action_space = spaces.Discrete(4)
        # Observation: 9 (location) + 4 (prev action) + 1 (reward flag) + 1 (buzzer) = 15 dims.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        
        # If no reward orders provided, generate 30 random orders (each order is a list of 4 unique cells).
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
        
        # Mapping for reward labels.
        self.label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        self.reset()
    
    def reset(self):
        # Reset agent to a fixed starting position (e.g., bottom-right).
        self.agent_pos = [self.grid_size - 1, self.grid_size - 1]
        # current_goal_idx indicates which reward (0=A,1=B,2=C,3=D) is expected next.
        self.current_goal_idx = 0  
        self.steps = 0
        
        # Previous action: initially a zero vector (length 4).
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.last_reward = 0  # binary flag for whether reward was just received.
        # Buzzer: set to 1 on reset (signalling the start of a new trial).
        self.buzzer = 1.0
        
        # Choose reward order.
        if self.training:
            self.reward_order = random.choice(self.all_reward_orders)
        else:
            if self.fixed_reward_order is not None:
                self.reward_order = self.fixed_reward_order
            else:
                self.reward_order = random.choice(self.all_reward_orders)
        
        return self._get_obs()
    
    def _get_obs(self):
        # One-hot encode current location.
        cell = self.agent_pos[0] * self.grid_size + self.agent_pos[1]
        loc_one_hot = np.zeros(self.num_cells, dtype=np.float32)
        loc_one_hot[cell] = 1.0
        
        # Concatenate: location, previous action, reward flag, buzzer flag.
        obs = np.concatenate([
            loc_one_hot,
            self.prev_action,
            np.array([self.last_reward], dtype=np.float32),
            np.array([self.buzzer], dtype=np.float32)
        ])
        return obs
    
    def step(self, action):
        self.steps += 1
        self.last_reward = 0  # reset reward flag at start of step
        
        # After the first step, turn off the buzzer.
        if self.steps > 1:
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
        # Check if the agent is at the current target location.
        if cell == self.reward_order[self.current_goal_idx]:
            reward = 1.0
            self.last_reward = 1
            reward_label = self.label_mapping[self.current_goal_idx]
            # Update cyclically: after D, go back to A.
            self.current_goal_idx = (self.current_goal_idx + 1) % len(self.reward_order)
        
        # Episode terminates only when max_steps is reached.
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
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x, hidden):
        # x: (batch_size, input_size) -> add time dimension.
        x = x.unsqueeze(1)
        out, new_hidden = self.gru(x, hidden)
        out = out.squeeze(1)
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
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        hidden = model.init_hidden(batch_size=1)
        done = False
        
        log_probs = []
        values = []
        rewards = []
        
        while not done:
            logits, value, hidden = model(obs, hidden)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            obs_next, reward, done, _ = env.step(action.item())
            obs_next = torch.from_numpy(obs_next).float().unsqueeze(0)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            obs = obs_next
        
        # Compute cumulative discounted returns.
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
# Evaluation & GRU Activation Recording
# ------------------------------
def evaluate_agent(env, model):
    """
    Runs one test episode and returns:
      - activations: a (T x hidden_size) array of GRU activations at each time step.
      - reward_events: a list of tuples (time_step, reward_label) indicating when rewards occurred.
    """
    model.eval()
    obs = env.reset()
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    hidden = model.init_hidden(batch_size=1)
    done = False
    hidden_states = []
    reward_events = []  # list of (time step, reward label)
    t = 0
    while not done:
        logits, value, hidden = model(obs, hidden)
        # Record hidden state (shape: [hidden_size])
        hidden_states.append(hidden.detach().cpu().numpy()[0, 0, :].copy())
        action = torch.distributions.Categorical(logits=logits).sample()
        obs, reward, done, info = env.step(action.item())
        if reward > 0:
            reward_label = info.get('reward_label', '')
            reward_events.append((t, reward_label))
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        t += 1
    return np.array(hidden_states), reward_events

# ------------------------------
# Plotting Function for Test Sessions
# ------------------------------
def plot_test_session(activations, reward_events, unit_order=None, session_idx=0):
    """
    Plots a heatmap of GRU activations (with units rank-ordered if provided)
    and overlays vertical lines at time bins where rewards occurred.
    
    Reward color coding: A - red, B - yellow, C - green, D - blue.
    """
    # If a specific ordering is provided, reorder columns.
    if unit_order is not None:
        activations = activations[:, unit_order]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(activations.T, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.xlabel("Time Step")
    plt.ylabel("GRU Unit (Rank-ordered)")
    plt.title(f"Test Session {session_idx+1} - GRU Activations")
    cbar = plt.colorbar(label='Activation')
    
    # Define colors for each reward.
    reward_colors = {'A': 'red', 'B': 'yellow', 'C': 'green', 'D': 'blue'}
    for t, label in reward_events:
        if label in reward_colors:
            plt.axvline(x=t, color=reward_colors[label], linestyle='--', linewidth=2)
    
    plt.show()

# ------------------------------
# Main Script
# ------------------------------
if __name__ == "__main__":
    # ------------------------------
    # Training Setup
    # ------------------------------
    num_training_orders = 30
    training_reward_orders = []
    num_cells = 9
    while len(training_reward_orders) < num_training_orders:
        order = random.sample(range(num_cells), 4)
        if order not in training_reward_orders:
            training_reward_orders.append(order)
    
    with open("training_reward_orders.txt", "w") as f:
        for order in training_reward_orders:
            f.write(f"{order}\n")

    train_env = GridMazeEnv(reward_orders=training_reward_orders, training=True, max_steps=200)
    model = ActorCritic(input_size=15, hidden_size=200, num_actions=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_episodes = 25_000
    episode_rewards = train_agent(train_env, model, optimizer, num_episodes=num_episodes, gamma=0.99)
    np.save("episode_rewards.npy", episode_rewards)
    torch.save(model.state_dict(), "gru_actor_critic_gridmaze.pth")
    # Plot reward per episode.
    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, label='Total Reward')
    
    # Compute and plot rolling average.
    rolling_avg = np.convolve(episode_rewards, np.ones(200)/200, mode='valid')
    plt.plot(range(199, len(episode_rewards)), rolling_avg, color='red', label='Rolling Average (200 episodes)')
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward over Episodes (Training)")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_rewards_plot.png")
    plt.show()
    # ------------------------------
    # Test Sessions on Unseen Reward Orders
    # ------------------------------
    # Generate 5 unseen reward orders (that are not in the training set).
    unseen_orders = []
    while len(unseen_orders) < 5:
        order = random.sample(range(num_cells), 4)
        if order not in training_reward_orders and order not in unseen_orders:
            unseen_orders.append(order)
    print("Unseen Test Reward Orders:", unseen_orders)

    with open("unseen_test_reward_orders.txt", "w") as f:
        for order in unseen_orders:
            f.write(f"{order}\n")
    
    test_results = []
    # Run 5 test sessions.
    for idx, test_order in enumerate(unseen_orders):
        test_env = GridMazeEnv(reward_orders=training_reward_orders, training=False,
                               fixed_reward_order=test_order, max_steps=200)
        activations, reward_events = evaluate_agent(test_env, model)
        test_results.append({'activations': activations, 'reward_events': reward_events})
    
    # Rank order the GRU units based on the first test session.
    first_activations = test_results[0]['activations']  # shape: (T, hidden_size)
    # For each unit, find the time index of its peak activation.
    peak_times = np.argmax(first_activations, axis=0)
    # Get sort order of units based on peak time.
    unit_order = np.argsort(peak_times)
    
    # Plot each test session heatmap with reward markers.
    for idx, result in enumerate(test_results):
        plot_test_session(result['activations'], result['reward_events'], unit_order=unit_order, session_idx=idx)
