# ABCD Recurrent Actor-Critic Network

This repository contains an implementation of a reinforcement learning (RL) agent that navigates a grid maze with corridors, receiving rewards in a specific sequence. The agent is trained using a Proximal Policy Optimization (PPO) algorithm with a Recurrent Actor-Critic network that includes a GRU layer.

## Project Structure

├── README.md # Project documentation
├── main.py # Entry point for training and running the model
├── model.py # Recurrent Actor-Critic model definition
├── environment.py # Custom environment definition
└── ppo_agent.py # PPO agent and training loop

