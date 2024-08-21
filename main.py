from environment import GridMazeWithCorridorsEnv
from ppo_agent import ppo_agent

if __name__ == "__main__":
    # Instantiate the environment
    env = GridMazeWithCorridorsEnv()

    # Train the agent
    ppo_agent.train(env)
