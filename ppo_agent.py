import torch
import torch.optim as optim
from torch.distributions import Categorical
from model import RecurrentActorCritic

class PPOAgent:
    def __init__(self, model, optimizer, gamma, lambda_gae, eps_clip):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.eps_clip = eps_clip

    def select_action(self, state, hidden_state):
        # Combine current location, previous action, and previous reward into one input tensor
        state = torch.FloatTensor(state).unsqueeze(0)  # Expand to [1, state_dim] for batch processing
        policy_logits, value, hidden_state = self.model(state, hidden_state)
        probs = Categorical(logits=policy_logits)
        action = probs.sample()
        return action.item(), value, probs.log_prob(action), probs.entropy(), hidden_state

    def compute_gae(self, rewards, values, masks, next_value):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lambda_gae * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, memory):
        states = torch.FloatTensor(memory['states'])
        actions = torch.LongTensor(memory['actions'])
        returns = torch.FloatTensor(memory['returns'])
        log_probs_old = torch.FloatTensor(memory['log_probs'])
        advantages = returns - torch.FloatTensor(memory['values'])

        hidden_state = self.model.init_hidden_state()

        for _ in range(4):  # PPO epochs
            policy_logits, values, _ = self.model(states, hidden_state)
            probs = Categorical(logits=policy_logits)
            log_probs = probs.log_prob(actions)
            entropy = probs.entropy()

            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.nn.functional.mse_loss(values, returns)
            loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, num_episodes=1000, rollout_length=100):
        for episode in range(num_episodes):
            state = env.reset()
            hidden_state = self.model.init_hidden_state()
            previous_action = 0  # Initialize with a default action
            previous_reward = 0  # Initialize with no reward
            memory = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': [], 'masks': [], 'returns': []}

            for _ in range(rollout_length):
                combined_state = self.get_combined_state(state, previous_action, previous_reward)
                action, value, log_prob, entropy, hidden_state = self.select_action(combined_state, hidden_state)
                next_state, reward, done, _ = env.step(action)

                memory['states'].append(combined_state)
                memory['actions'].append(action)
                memory['rewards'].append(reward)
                memory['values'].append(value.item())
                memory['log_probs'].append(log_prob.item())
                memory['masks'].append(1 - done)

                previous_action = action
                previous_reward = reward
                state = next_state

                if done:
                    break

            next_value = self.model(torch.FloatTensor(self.get_combined_state(state, previous_action, previous_reward)).unsqueeze(0), hidden_state)[1].item()
            memory['returns'] = self.compute_gae(memory['rewards'], memory['values'], memory['masks'], next_value)
            self.update(memory)

            if episode % 10 == 0:
                print(f'Episode {episode}: Total Reward: {sum(memory["rewards"])}')

    def get_combined_state(self, state, previous_action, previous_reward):
        # Concatenate the current state (location), previous action, and previous reward
        return torch.tensor([*state, previous_action, previous_reward])

# Hyperparameters
input_size = 4  # (x, y, previous_action, previous_reward)
hidden_size = 128  # Number of GRU units
num_actions = 4  # Up, Right, Down, Left
learning_rate = 3e-4
gamma = 0.99  # Discount factor
lambda_gae = 0.95  # GAE lambda
eps_clip = 0.2  # PPO epsilon

# Instantiate the network and the optimizer
model = RecurrentActorCritic(input_size, hidden_size, num_actions)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

ppo_agent = PPOAgent(model, optimizer, gamma, lambda_gae, eps_clip)
