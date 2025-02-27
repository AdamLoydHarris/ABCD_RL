
import torch
import torch.nn as nn

class RecurrentActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(RecurrentActorCritic, self).__init__()
        self.hidden_size = hidden_size

        # Correct input size to include state + prev_action + prev_reward
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden_state):
        # Ensure input size matches
        x, hidden_state = self.gru(x.unsqueeze(1), hidden_state)
        x = x.squeeze(1)
        x = F.ReLU(self.hidden(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value, hidden_state

    def init_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class RecurrentActorCritic(nn.Module):
#     def __init__(self, input_size, hidden_size, num_actions):
#         super(RecurrentActorCritic, self).__init__()
#         self.hidden_size = hidden_size

#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.actor = nn.Linear(hidden_size, num_actions)
#         self.critic = nn.Linear(hidden_size, 1)

#     def forward(self, x, hidden_state):
#         x, hidden_state = self.gru(x.unsqueeze(1), hidden_state)
#         x = x.squeeze(1)
#         policy_logits = self.actor(x)
#         value = self.critic(x)
#         return policy_logits, value, hidden_state

#     def init_hidden_state(self):
#         return torch.zeros(1, 1, self.hidden_size)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class RecurrentActorCritic(nn.Module):
#     def __init__(self, input_size, hidden_size, num_actions):
#         super(RecurrentActorCritic, self).__init__()
#         self.hidden_size = hidden_size

#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.actor = nn.Linear(hidden_size, num_actions)
#         self.critic = nn.Linear(hidden_size, 1)

#     def forward(self, x, hidden_state):
#         x, hidden_state = self.gru(x.unsqueeze(1), hidden_state)
#         x = x.squeeze(1)
#         policy_logits = self.actor(x)
#         value = self.critic(x)
#         return policy_logits, value, hidden_state

#     def init_hidden_state(self):
#         return torch.zeros(1, 1, self.hidden_size)


# import torch
# import torch.nn as nn

# class RecurrentActorCritic(nn.Module):
#     def __init__(self, input_size, hidden_size, num_actions):
#         super(RecurrentActorCritic, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.actor = nn.Linear(hidden_size, num_actions)
#         self.critic = nn.Linear(hidden_size, 1)
#         self.hidden_size = hidden_size
#         self.num_layers = 1  # Number of GRU layers

#     def forward(self, x, hidden_state):
#         # Handle cases where input is 2D (batch_size, input_size) or 3D (batch_size, seq_len, input_size)
#         if x.dim() == 2:
#             x = x.unsqueeze(1)  # Add sequence length dimension for unbatched input
#         x, hidden_state = self.gru(x, hidden_state)
#         x = x[:, -1, :]  # Take the output from the last time step
#         policy_logits = self.actor(x)
#         value = self.critic(x)
#         return policy_logits, value, hidden_state

#     def init_hidden_state(self, batch_size=1):
#         return torch.zeros(self.num_layers, batch_size, self.hidden_size)


# import torch
# import torch.nn as nn

# class RecurrentActorCritic(nn.Module):
#     def __init__(self, input_size, hidden_size, num_actions):
#         super(RecurrentActorCritic, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.actor = nn.Linear(hidden_size, num_actions)
#         self.critic = nn.Linear(hidden_size, 1)
#         self.hidden_size = hidden_size
#         self.num_layers = 1  # Number of GRU layers

#     def forward(self, x, hidden_state):
#         x, hidden_state = self.gru(x, hidden_state)
#         x = x[:, -1, :]  # Take the output from the last time step
#         policy_logits = self.actor(x)
#         value = self.critic(x)
#         return policy_logits, value, hidden_state

#     def init_hidden_state(self, batch_size=1):
#         return torch.zeros(self.num_layers, batch_size, self.hidden_size)


# import torch
# import torch.nn as nn

# class RecurrentActorCritic(nn.Module):
#     def __init__(self, input_size, hidden_size, num_actions):
#         super(RecurrentActorCritic, self).__init__()

#         # GRU Layer
#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

#         # Actor layer
#         self.actor = nn.Linear(hidden_size, num_actions)
        
#         # Critic layer
#         self.critic = nn.Linear(hidden_size, 1)

#     def forward(self, x, hidden_state):
#         # Pass through the GRU layer
#         x, hidden_state = self.gru(x.unsqueeze(1), hidden_state)

#         # Actor and Critic outputs
#         policy_logits = self.actor(x.squeeze(1))
#         state_value = self.critic(x.squeeze(1))

#         return policy_logits, state_value, hidden_state

#     def init_hidden_state(self, batch_size=1):
#         # Initialize hidden state with zeros
#         return torch.zeros(1, batch_size, self.gru.hidden_size)
