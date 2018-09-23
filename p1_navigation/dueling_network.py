import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        feature_size = 256
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, feature_size),
            nn.ReLU())
        
        value_size = 128
        self.value_layer = nn.Sequential(
            nn.Linear(feature_size, value_size),
            nn.ReLU(),
            nn.Linear(value_size, 1))
        
        advantage_size = 128
        self.advantage_layer = nn.Sequential(
            nn.Linear(feature_size, advantage_size),
            nn.ReLU(),
            nn.Linear(advantage_size, action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        feature = self.feature_layer(x)
        action_value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        
        q_value = action_value + (advantage - advantage.mean())
        return q_value