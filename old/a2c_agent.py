import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class A2CAgent:
    def __init__(self, model, optimizer, gamma=0.99):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(self, state, hidden_state):
        policy_logits, value, hidden_state = self.model(state, hidden_state)
        policy_dist = F.softmax(policy_logits, dim=-1)
        action = Categorical(policy_dist).sample()
        log_prob = torch.log(policy_dist[action])
        return action.item(), log_prob, value, hidden_state

    def compute_returns(self, rewards, values, done, gamma):
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def train(self, env, num_episodes=1000):
        for episode in range(num_episodes):
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)
            hidden_state = self.model.init_hidden_state()
            log_probs = []
            values = []
            rewards = []
            done = False

            while not done:
                action, log_prob, value, hidden_state = self.select_action(state, hidden_state)
                next_state, reward, done, _ = env.step(action)
                next_state = torch.FloatTensor(next_state).unsqueeze(0)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)

                state = next_state

            # Compute returns and advantages
            returns = self.compute_returns(rewards, values, done, self.gamma)

            # Loss function
            policy_loss = 0
            value_loss = 0
            for log_prob, value, R in zip(log_probs, values, returns):
                advantage = R - value.item()
                policy_loss -= log_prob * advantage
                value_loss += F.mse_loss(value, torch.tensor([[R]]))

            loss = policy_loss + value_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward: {sum(rewards)}")

# import torch # type: ignore
# import torch.nn.functional as F
# from collections import deque

# class A2CAgent:
#     def __init__(self, model, optimizer, gamma):
#         self.model = model
#         self.optimizer = optimizer
#         self.gamma = gamma

#     def select_action(self, state, hidden_state):
#         policy_logits, value, hidden_state = self.model(state, hidden_state)
#         policy_dist = F.softmax(policy_logits, dim=-1)
#         action = torch.multinomial(policy_dist, 1).item()
#         log_prob = torch.log(policy_dist[action])
#         return action, log_prob, value, hidden_state

#     def update(self, rewards, log_probs, values, next_value):
#         returns = deque()
#         R = next_value
#         for r in reversed(rewards):
#             R = r + self.gamma * R
#             returns.appendleft(R)
        
#         returns = torch.tensor(returns).float()
#         log_probs = torch.stack(log_probs)
#         values = torch.stack(values)
        
#         advantages = returns - values
        
#         actor_loss = -(log_probs * advantages.detach()).mean()
#         critic_loss = F.mse_loss(values, returns)
#         loss = actor_loss + critic_loss

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#     def train(self, env, num_episodes):
#         for episode in range(num_episodes):
#             state = env.reset()
#             state = torch.FloatTensor(state).unsqueeze(0)
#             hidden_state = self.model.init_hidden_state()

#             log_probs = []
#             values = []
#             rewards = []
#             total_reward = 0

#             done = False
#             while not done:
#                 action, log_prob, value, hidden_state = self.select_action(state, hidden_state)
#                 next_state, reward, done, _ = env.step(action)
                
#                 next_state = torch.FloatTensor(next_state).unsqueeze(0)
#                 log_probs.append(log_prob)
#                 values.append(value)
#                 rewards.append(reward)
#                 total_reward += reward
                
#                 state = next_state
            
#             _, next_value, _ = self.model(state, hidden_state)
#             self.update(rewards, log_probs, values, next_value.item())
#             print(f"Episode {episode}: Total Reward: {total_reward}")
