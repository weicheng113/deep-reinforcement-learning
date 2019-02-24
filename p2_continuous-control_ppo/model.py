import torch.nn.functional as f
import torch.nn as nn
import torch
import torch.distributions as distributions


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_units, seed):
        super(Actor, self).__init__()

        torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(in_features=hidden_units, out_features=action_size)
        self.std = nn.Parameter(torch.zeros(action_size))

        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.output)

    def forward(self, state, action=None):
        x = state
        x = f.elu(self.fc1(x))

        x = f.elu(self.fc2(x))

        x = self.output(x)
        action_mean = torch.tanh(x)

        dist = distributions.Normal(action_mean, f.softplus(self.std))
        if action is None:
            action = dist.sample()

        prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        # entropy1 = dist.entropy()
        # entropy = dist.entropy().mean(-1, keepdim=True)
        return action, prob, entropy


class Critic(nn.Module):
    def __init__(self, state_size, hidden_units, seed):
        super(Critic, self).__init__()

        torch.manual_seed(seed)

        self.fc1 = nn.Linear(in_features=state_size, out_features=hidden_units)
        self.fc2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.output = nn.Linear(in_features=hidden_units, out_features=1)
        self.bn1 = nn.BatchNorm1d(num_features=state_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_units)

        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.output)

    def forward(self, state):
        x = self.bn1(state)
        x = f.elu(self.fc1(x))
        x = self.bn2(x)
        x = f.elu(self.fc2(x))

        return self.output(x)
