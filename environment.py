import numpy as np
import gym
from gym import spaces

class GridMazeWithCorridorsEnv(gym.Env):
    def __init__(self):
        super(GridMazeWithCorridorsEnv, self).__init__()

        # Define grid size with corridors
        self.grid_size = 5  # 5x5 grid including corridors
        
        # Define action space: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_space = spaces.Discrete(4)
        
        # Define observation space: agent's position (x, y)
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        
        # Define reward positions (A, B, C, D) corresponding to 'T' spots
        self.rewards_positions = {
            'A': (0, 0),
            'B': (0, 4),
            'C': (4, 4),
            'D': (4, 0)
        }
        
        # Define the valid positions (Towers and Corridors)
        self.valid_positions = set([
            (0, 0), (0, 2), (0, 4),
            (1, 1), (1, 3),
            (2, 0), (2, 2), (2, 4),
            (3, 1), (3, 3),
            (4, 0), (4, 2), (4, 4)
        ])
        
        # Define the correct sequence
        self.sequence = ['A', 'B', 'C', 'D']
        self.current_index = 0  # Start with 'A'
        
        # Initialize agent's starting position
        self.agent_position = np.array([2, 2])  # Start in the center (at a 'T')
        
        # Define maximum steps per episode
        self.max_steps = 100
        self.current_steps = 0

    def reset(self):
        # Reset agent's position and sequence index
        self.agent_position = np.array([2, 2])
        self.current_index = 0
        self.current_steps = 0
        return self.agent_position

    def step(self, action):
        self.current_steps += 1

        # Define action effects
        action_effects = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        
        # Update agent's position based on action
        new_position = self.agent_position + action_effects[action]
        
        # Check if the new position is within bounds and is a valid position
        if tuple(new_position) in self.valid_positions:
            self.agent_position = new_position
        
        # Check if the agent is on the correct reward location
        current_target = self.sequence[self.current_index]
        if tuple(self.agent_position) == self.rewards_positions[current_target]:
            reward = 1  # Correct reward location
            self.current_index = (self.current_index + 1) % len(self.sequence)  # Move to the next in the sequence
        else:
            reward = 0  # No reward

        # Check if the episode should terminate
        done = self.current_steps >= self.max_steps

        return self.agent_position, reward, done, {}

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '-', dtype=str)

        # Place Towers (rewards)
        for key, position in self.rewards_positions.items():
            grid[position] = key
        
        # Place agent
        x, y = self.agent_position
        grid[x, y] = 'Agt'
        
        print(grid)

    def close(self):
        pass
