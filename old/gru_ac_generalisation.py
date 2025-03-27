import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Custom 3x3 Grid Maze Environment with multiple reward orders and buzzer signal.
class GridMazeEnv(gym.Env):
    """
    A 3x3 grid maze where the agent must collect rewards at four locations
    in a specified order. The observation is now a 15-dimensional vector:
      - 9 dims: one-hot encoding of the current location.
      - 4 dims: one-hot encoding of the previous action.
      - 1 dim: binary flag indicating whether a reward was just received.
      - 1 dim: buzzer flag indicating the start of a new trial.
    
    The environment is parameterised by a list of reward orders.
    For training, one of these orders is randomly chosen at reset.
    For testing (when training=False), a fixed reward order can be used.
    """
    def __init__(self, reward_orders=None, training=True, fixed_reward_order=None, max_steps=50):
        super(GridMazeEnv, self).__init__()
        self.grid_size = 3  # 3x3 grid; cells 0-8 (row-major)
        self.num_cells = self.grid_size * self.grid_size
        self.max_steps = max_steps
        
        # Action space: 0=up, 1=down, 2=left, 3=right.
        self.action_space = spaces.Discrete(4)
        # Observation space: 9 (location) + 4 (prev action) + 1 (reward flag) + 1 (buzzer) = 15 dims.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        
        # If no reward orders are provided, generate 30 random orders (each order is a list of 4 unique cells).
        if reward_orders is None:
            self.all_reward_orders = []
            while len(self.all_reward_orders) < 30:
                order = random.sample(range(self.num_cells), 4)
                # Avoid duplicates in the list of orders.
                if order not in self.all_reward_orders:
                    self.all_reward_orders.append(order)
        else:
            self.all_reward_orders = reward_orders
        
        self.training = training
        self.fixed_reward_order = fixed_reward_order
        
        # The current reward order (will be set in reset).
        self.reward_order = None
        self.reset()
    
    def reset(self):
        # Reset agent to a fixed starting position (e.g., bottom-right).
        self.agent_pos = [self.grid_size - 1, self.grid_size - 1]
        self.current_goal_idx = 0  # index of the next reward to collect.
        self.steps = 0
        
        # Previous action: initially a zero vector (length 4).
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.last_reward = 0  # binary flag for reward.
        
        # Buzzer flag: set to 1 at the start of a new trial.
        self.buzzer = 1.0
        
        # Choose a reward order.
        if self.training:
            self.reward_order = random.choice(self.all_reward_orders)
        else:
            if self.fixed_reward_order is not None:
                self.reward_order = self.fixed_reward_order
            else:
                self.reward_order = random.choice(self.all_reward_orders)
        
        return self._get_obs()
    
    def _get_obs(self):
        # One-hot encode the agent's current cell.
        cell = self.agent_pos[0] * self.grid_size + self.agent_pos[1]
        loc_one_hot = np.zeros(self.num_cells, dtype=np.float32)
        loc_one_hot[cell] = 1.0
        
        # Observation: concatenate location, previous action, reward flag, and buzzer flag.
        obs = np.concatenate([
            loc_one_hot, 
            self.prev_action, 
            np.array([self.last_reward], dtype=np.float32),
            np.array([self.buzzer], dtype=np.float32)
        ])
        return obs
    
    def step(self, action):
        self.steps += 1
        self.last_reward = 0  # reset reward flag
        
        # After the first step, turn off the buzzer signal.
        self.buzzer = 0.0
        
        # Update previous action vector.
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_action[action] = 1.0
        
        # Compute new position based on action.
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
            new_row, new_col = row, col  # fallback for illegal action
        
        self.agent_pos = [new_row, new_col]
        cell = new_row * self.grid_size + new_col
        
        reward = 0.0
        done = False
        
        # Check if the agent is at the expected reward location.
        if self.current_goal_idx < len(self.reward_order) and cell == self.reward_order[self.current_goal_idx]:
            reward = 1.0
            self.last_reward = 1  # set reward flag for observation.
            self.current_goal_idx += 1
        
        # End episode if all rewards are collected or max steps reached.
        if self.current_goal_idx == len(self.reward_order) or self.steps >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, {}

# Actor-Critic network with a GRU layer.
class ActorCritic(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_actions=4):
        super(ActorCritic, self).__init__()
        self.hidden_size = hidden_size
        
        # GRU layer to handle the sequential aspect.
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        # Actor head: produces logits over actions.
        self.actor = nn.Linear(hidden_size, num_actions)
        # Critic head: produces a state-value estimate.
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x, hidden):
        # x shape: (batch_size, input_size); add time dimension (seq_len=1)
        x = x.unsqueeze(1)
        out, new_hidden = self.gru(x, hidden)
        out = out.squeeze(1)  # shape: (batch_size, hidden_size)
        action_logits = self.actor(out)
        state_value = self.critic(out)
        return action_logits, state_value, new_hidden
    
    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)

# Training loop for the actor-critic agent.
def train_agent(env, model, optimizer, num_episodes=1000, gamma=0.99):
    model.train()
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        obs = torch.from_numpy(obs).float().unsqueeze(0)  # shape: (1, input_size)
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
        
        # Compute cumulative discounted rewards.
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()
        values = torch.cat(values).squeeze()
        log_probs = torch.stack(log_probs)
        
        # Advantage estimation.
        advantage = returns - values
        
        # Loss: actor loss (policy gradient) plus critic loss.
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

# Evaluation function to record GRU activations on a test trial.
def evaluate_agent(env, model):
    model.eval()
    obs = env.reset()
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    hidden = model.init_hidden(batch_size=1)
    done = False
    hidden_states = []  # to record GRU activations
    
    while not done:
        logits, value, hidden = model(obs, hidden)
        # Record the hidden state from the GRU (shape: [1,1,hidden_size] -> [hidden_size])
        hidden_states.append(hidden.detach().cpu().numpy()[0,0,:].copy())
        # Sample an action.
        action = torch.distributions.Categorical(logits=logits).sample()
        obs, reward, done, _ = env.step(action.item())
        obs = torch.from_numpy(obs).float().unsqueeze(0)
    
    return np.array(hidden_states)

if __name__ == "__main__":
    # Generate 30 random training reward orders.
    num_training_orders = 300
    training_reward_orders = []
    num_cells = 9
    while len(training_reward_orders) < num_training_orders:
        order = random.sample(range(num_cells), 4)
        if order not in training_reward_orders:
            training_reward_orders.append(order)
    
    # Create training environment.
    train_env = GridMazeEnv(reward_orders=training_reward_orders, training=True, max_steps=50)
    
    # Instantiate the model and optimizer.
    model = ActorCritic(input_size=15, hidden_size=128, num_actions=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train the agent.
    num_episodes = 10000
    episode_rewards = train_agent(train_env, model, optimizer, num_episodes=num_episodes, gamma=0.99)
    
    # Plot the reward over episodes.
    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward over Episodes")
    plt.grid(True)
    plt.show()
    
    # Define an unseen (test) reward order.
    # Ensure that this test order is not in the training set.
    test_reward_order = [i for i in range(9) if i not in training_reward_orders[0]][:4]
    # For clarity, here we simply choose a fixed order (you may design a different unseen order).
    test_reward_order = [0, 2, 5, 7]  # Example unseen order.
    
    # Create a test environment with the fixed reward order.
    test_env = GridMazeEnv(reward_orders=training_reward_orders, training=False, fixed_reward_order=test_reward_order, max_steps=200)
    
    # Evaluate the agent on the unseen task and record GRU activations.
    gru_activations = evaluate_agent(test_env, model)
    
    # Plot GRU activations over time (e.g., as a heatmap).
    # Rank-order the rows by the index of their peak activation.
    peak_indices = np.argmax(gru_activations, axis=0)
    sorted_indices = np.argsort(peak_indices)
    sorted_activations = gru_activations[:, sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.imshow(sorted_activations.T, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Activation')
    plt.xlabel("Time Step")
    plt.ylabel("Hidden Unit (Sorted by Peak Activation)")
    plt.title("GRU Hidden State Activations Over Time (Test Trial)")
    plt.show()
