import torch
import torch.nn as nn

class RecurrentActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(RecurrentActorCritic, self).__init__()

        # GRU Layer
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        # Actor layer
        self.actor = nn.Linear(hidden_size, num_actions)
        
        # Critic layer
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden_state):
        # Pass through the GRU layer
        x, hidden_state = self.gru(x.unsqueeze(1), hidden_state)

        # Actor and Critic outputs
        policy_logits = self.actor(x.squeeze(1))
        state_value = self.critic(x.squeeze(1))

        return policy_logits, state_value, hidden_state

    def init_hidden_state(self, batch_size=1):
        # Initialize hidden state with zeros
        return torch.zeros(1, batch_size, self.gru.hidden_size)
