import torch.nn as nn
import torch.distributions as distributions


class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()

        in_channels = 2
        conv_channels = [4, 16]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=3),
            nn.ReLU())
        self.conv_out_size = 16 * 8 * 8

        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim))

    def forward(self, state, action=None):
        x = self.conv_layers(state)

        x = x.view(-1, self.conv_out_size)

        logits = self.fc_layers(x)

        dist = distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        action.unsqueeze_(1)
        prob = dist.probs.gather(1, action)
        return action, prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        in_channels = 2
        conv_channels = [4, 16]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=5, stride=3),
            nn.ReLU())
        self.conv_out_size = 16 * 8 * 8

        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, state):
        x = self.conv_layers(state)
        x = x.view(-1, self.conv_out_size)
        # print(x.size())
        return self.fc_layers(x)
