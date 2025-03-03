from model import RecurrentActorCritic
from a2c_agent import A2CAgent
from environment import GridMazeWithCorridors
import torch.optim as optim

# Initialize environment
env = GridMazeWithCorridors()

# Define the model
state_size = env.observation_space.shape[0]  # This should be 16
input_size = state_size + 2  # 16 (state) + 2 (previous action + previous reward)
hidden_size = 128
num_actions = env.action_space.n

model = RecurrentActorCritic(input_size, hidden_size, num_actions)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the A2C agent
gamma = 0.99
a2c_agent = A2CAgent(model, optimizer, gamma)

# Train the agent
a2c_agent.train(env, num_episodes=1000)


# import torch.optim as optim
# from model import RecurrentActorCritic
# from a2c_agent import A2CAgent
# from environment import GridMazeWithCorridors

# # Initialize environment
# env = GridMazeWithCorridors()

# # Define the model
# state_size = env.observation_space.shape[0]
# input_size = state_size + 2  # State size + previous action + previous reward
# hidden_size = 128
# num_actions = env.action_space.n

# model = RecurrentActorCritic(input_size, hidden_size, num_actions)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Initialize the A2C agent
# gamma = 0.99
# a2c_agent = A2CAgent(model, optimizer, gamma)

# # Train the agent
# a2c_agent.train(env, num_episodes=1000)

# import torch.optim as optim
# from model import RecurrentActorCritic
# from a2c_agent import A2CAgent
# from environment import GridMazeWithCorridorsEnv
# import gym

# # Initialize environment
# env = GridMazeWithCorridorsEnv()

# # Define the model
# input_size = 4  #
# hidden_size = 128
# num_actions = env.action_space.n

# model = RecurrentActorCritic(input_size, hidden_size, num_actions)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Initialize the A2C agent
# gamma = 0.99
# a2c_agent = A2CAgent(model, optimizer, gamma)

# # Train the agent
# a2c_agent.train(env, num_episodes=1000)


# from environment import GridMazeWithCorridorsEnv
# from ppo_agent import PPOAgent
# from model import RecurrentActorCritic
# import torch.optim as optim
# import gym  # or the custom environment library you are using

# # Define the environment and PPO parameters
# env = GridMazeWithCorridorsEnv() # Replace with your environment
# input_size = 4  # Example: 2 for current location, 1 for previous action, 1 for previous reward
# hidden_size = 128
# num_actions = env.action_space.n  # Number of possible actions

# # Initialize model, optimizer, and PPO agent
# model = RecurrentActorCritic(input_size, hidden_size, num_actions)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# gamma = 0.99
# lambda_gae = 0.95
# eps_clip = 0.2

# ppo_agent = PPOAgent(model, optimizer, gamma, lambda_gae, eps_clip)

# # Train the model
# ppo_agent.train(env, num_episodes=1000, rollout_length=100)


# from environment import GridMazeWithCorridorsEnv
# from ppo_agent import PPOAgent
# from model import RecurrentActorCritic
# import torch.optim as optim

# # Hyperparameters
# input_size = 4  # (x, y, previous_action, previous_reward)
# hidden_size = 128  # Number of GRU units
# num_actions = 4  # Up, Right, Down, Left
# learning_rate = 3e-4
# gamma = 0.99  # Discount factor
# lambda_gae = 0.95  # GAE lambda
# eps_clip = 0.2  # PPO epsilon

# if __name__ == "__main__":
#     # Instantiate the environment
#     env = GridMazeWithCorridorsEnv()

#     # Instantiate the model and optimizer
#     model = RecurrentActorCritic(input_size, hidden_size, num_actions)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     # Instantiate the PPO agent
#     ppo_agent = PPOAgent(model, optimizer, gamma, lambda_gae, eps_clip)

#     # Train the agent
#     ppo_agent.train(env)


# from environment import GridMazeWithCorridorsEnv
# from ppo_agent import PPOAgent, model, optimizer, gamma, lambda_gae, eps_clip

# if __name__ == "__main__":
#     # Instantiate the environment
#     env = GridMazeWithCorridorsEnv()
    
#     ppo_agent = PPOAgent(model, optimizer, gamma, lambda_gae, eps_clip)
#     # Train the agent
#     ppo_agent.train(env)
