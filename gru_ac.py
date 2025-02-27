import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Custom 3x3 Grid Maze Environment
class GridMazeEnv(gym.Env):
    """
    A 3x3 grid maze where the agent must collect rewards at four specific locations
    (A, B, C, D) in order. The grid cells are numbered (row-major) from 0 to 8.
    Reward sequence (1-indexed positions: 1,5,7,3) corresponds to 0-indexed cells:
      - A: cell 0 (top-left)
      - B: cell 4 (center)
      - C: cell 6 (bottom-left)
      - D: cell 2 (top-right)
    The agent’s observation is a concatenated vector of:
      - One-hot representation of current location (9 dims)
      - One-hot representation of previous action (4 dims)
      - Binary flag indicating whether a reward was just received (1 dim)
    The hidden state (current reward target) is not revealed to the agent.
    """
    def __init__(self, max_steps=50):
        super(GridMazeEnv, self).__init__()
        self.grid_size = 3  # 3x3 grid
        self.num_cells = self.grid_size * self.grid_size
        self.max_steps = max_steps
        
        # Define action space: 0=up, 1=down, 2=left, 3=right.
        self.action_space = spaces.Discrete(4)
        # Observation space: 9 (location one-hot) + 4 (prev action one-hot) + 1 (reward flag)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)
        
        # Reward locations in order: A, B, C, D.
        self.reward_order = [0, 4, 6, 2]
        self.reset()
    
    def reset(self):
        # Start the agent in a fixed starting position (e.g., bottom-right)
        self.agent_pos = [self.grid_size - 1, self.grid_size - 1]
        # Index of the next reward to collect
        self.current_goal_idx = 0
        # Reset step counter
        self.steps = 0
        # No previous action on the first step: use zero one-hot vector of length 4.
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.last_reward = 0  # binary indicator
        return self._get_obs()
    
    def _get_obs(self):
        # One-hot encoding of the agent's current cell.
        cell = self.agent_pos[0] * self.grid_size + self.agent_pos[1]
        loc_one_hot = np.zeros(self.num_cells, dtype=np.float32)
        loc_one_hot[cell] = 1.0
        
        # Observation: concat current location, previous action, and reward flag.
        obs = np.concatenate([loc_one_hot, self.prev_action, np.array([self.last_reward], dtype=np.float32)])
        return obs
    
    def step(self, action):
        self.steps += 1
        self.last_reward = 0  # reset reward flag
        
        # Update previous action vector (one-hot)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_action[action] = 1.0
        
        # Determine new position based on action
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
            new_row, new_col = row, col  # illegal action fallback
        
        self.agent_pos = [new_row, new_col]
        cell = new_row * self.grid_size + new_col
        
        reward = 0.0
        done = False
        
        # Check if agent is at the current target reward location.
        if self.current_goal_idx < len(self.reward_order) and cell == self.reward_order[self.current_goal_idx]:
            reward = 1.0  # assign reward
            self.last_reward = 1  # set reward flag for observation
            self.current_goal_idx += 1
        
        # Episode ends if all rewards are collected or max steps reached.
        if self.current_goal_idx == len(self.reward_order) or self.steps >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, {}

# Actor-Critic network with GRU
class ActorCritic(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, num_actions=4):
        super(ActorCritic, self).__init__()
        self.hidden_size = hidden_size
        
        # GRU to maintain recurrent hidden state. 
        # Input is the observation at the current time step.
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        # Actor head: outputs logits over actions.
        self.actor = nn.Linear(hidden_size, num_actions)
        # Critic head: outputs a state-value estimate.
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x, hidden):
        """
        Forward pass for a single time step.
        
        Args:
            x: Input observation (batch_size x input_size)
            hidden: Previous hidden state (1 x batch_size x hidden_size)
        Returns:
            action_logits: (batch_size x num_actions)
            state_value: (batch_size x 1)
            new_hidden: updated hidden state (1 x batch_size x hidden_size)
        """
        # Add time dimension: (batch_size, seq_len=1, input_size)
        x = x.unsqueeze(1)
        out, new_hidden = self.gru(x, hidden)
        # Remove sequence dimension: (batch_size x hidden_size)
        out = out.squeeze(1)
        action_logits = self.actor(out)
        state_value = self.critic(out)
        return action_logits, state_value, new_hidden
    
    def init_hidden(self, batch_size=1):
        # Initialize hidden state to zeros.
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
            # Forward pass through the model.
            logits, value, hidden = model(obs, hidden)
            # Sample an action from the categorical distribution.
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Execute action in the environment.
            obs_next, reward, done, _ = env.step(action.item())
            obs_next = torch.from_numpy(obs_next).float().unsqueeze(0)
            
            # Store log probability, value, and reward.
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            obs = obs_next
        
        # Compute cumulative discounted rewards and advantage.
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
        
        # Compute actor (policy) and critic (value) losses.
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

if __name__ == "__main__":
    # Create environment and model.
    env = GridMazeEnv(max_steps=50)
    model = ActorCritic(input_size=14, hidden_size=128, num_actions=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train the agent.
    episode_rewards = train_agent(env, model, optimizer, num_episodes=1000, gamma=0.99)
    
    # Optionally: save the trained model.
    torch.save(model.state_dict(), "actor_critic_gridmaze.pth")
