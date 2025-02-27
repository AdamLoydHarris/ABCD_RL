import gym
from gym import spaces
import numpy as np

class GridMazeWithCorridors(gym.Env):
    def __init__(self):
        super(GridMazeWithCorridors, self).__init__()
        self.grid_size = 3  # 3x3 grid
        self.state_size = (self.grid_size + 1) * (self.grid_size + 1)  # 4x4 grid due to corridors
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

        # Define reward locations and the order
        self.reward_locs = [(0, 0), (0, 2), (2, 0), (2, 2)]  # A, B, C, D
        self.current_target_idx = 0
        self.current_target = self.reward_locs[self.current_target_idx]

        # Initialize state
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.reset()

    def reset(self):
        self.current_pos = (2, 2)  # Start in the middle
        self.current_target_idx = 0
        self.current_target = self.reward_locs[self.current_target_idx]
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros(self.state_size)
        grid_pos = self.current_pos[0] * self.grid_size + self.current_pos[1]
        obs[grid_pos] = 1.0
        return obs

    def step(self, action):
        x, y = self.current_pos

        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.grid_size - 1, y + 1)

        self.current_pos = (x, y)
        done = False
        reward = 0

        if self.current_pos == self.current_target:
            reward = 1.0
            self.current_target_idx = (self.current_target_idx + 1) % len(self.reward_locs)
            self.current_target = self.reward_locs[self.current_target_idx]

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.current_pos] = 1
        print(grid)


# import numpy as np
# import gym
# from gym import spaces

# class GridMazeWithCorridorsEnv(gym.Env):
#     def __init__(self):
#         super(GridMazeWithCorridorsEnv, self).__init__()

#         # Define grid size with corridors
#         self.grid_size = 5  # 5x5 grid including corridors
        
#         # Define action space: 0=Up, 1=Right, 2=Down, 3=Left
#         self.action_space = spaces.Discrete(4)
        
#         # Define observation space: agent's position (x, y)
#         self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        
#         # Define reward positions (A, B, C, D) corresponding to 'T' spots
#         self.rewards_positions = {
#             'A': (0, 0),
#             'B': (0, 4),
#             'C': (4, 4),
#             'D': (4, 0)
#         }
        
#         # Define the valid positions (Towers and Corridors)
#         self.valid_positions = set([
#             (0, 0), (0, 2), (0, 4),
#             (1, 1), (1, 3),
#             (2, 0), (2, 2), (2, 4),
#             (3, 1), (3, 3),
#             (4, 0), (4, 2), (4, 4)
#         ])
        
#         # Define the correct sequence
#         self.sequence = ['A', 'B', 'C', 'D']
#         self.current_index = 0  # Start with 'A'
        
#         # Initialize agent's starting position
#         self.agent_position = np.array([2, 2])  # Start in the center (at a 'T')
        
#         # Define maximum steps per episode
#         self.max_steps = 100
#         self.current_steps = 0

#     def reset(self):
#         # Reset agent's position and sequence index
#         self.agent_position = np.array([2, 2])
#         self.current_index = 0
#         self.current_steps = 0
#         return self.agent_position

#     def step(self, action):
#         self.current_steps += 1

#         # Define action effects
#         action_effects = {
#             0: (-1, 0),  # Up
#             1: (0, 1),   # Right
#             2: (1, 0),   # Down
#             3: (0, -1)   # Left
#         }
        
#         # Update agent's position based on action
#         new_position = self.agent_position + action_effects[action]
        
#         # Check if the new position is within bounds and is a valid position
#         if tuple(new_position) in self.valid_positions:
#             self.agent_position = new_position
        
#         # Check if the agent is on the correct reward location
#         current_target = self.sequence[self.current_index]
#         if tuple(self.agent_position) == self.rewards_positions[current_target]:
#             reward = 1  # Correct reward location
#             self.current_index = (self.current_index + 1) % len(self.sequence)  # Move to the next in the sequence
#         else:
#             reward = 0  # No reward

#         # Check if the episode should terminate
#         done = self.current_steps >= self.max_steps

#         return self.agent_position, reward, done, {}

#     def render(self, mode='human'):
#         grid = np.full((self.grid_size, self.grid_size), '-', dtype=str)

#         # Place Towers (rewards)
#         for key, position in self.rewards_positions.items():
#             grid[position] = key
        
#         # Place agent
#         x, y = self.agent_position
#         grid[x, y] = 'Agt'
        
#         print(grid)

#     def close(self):
#         pass
