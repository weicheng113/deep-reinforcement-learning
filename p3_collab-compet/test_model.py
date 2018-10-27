import unittest
from model import Actor, Critic
import numpy as np
import torch


class TestActor(unittest.TestCase):
    def setUp(self):
        self.state_dim = 24
        self.action_dim = 2

        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim, fc1_units=64, fc2_units=64, seed=0)

    def test_forward(self):
        n = 2
        states = torch.tensor(np.random.random_sample((n, self.state_dim)), dtype=torch.float)

        actions = self.actor.forward(states)
        self.assertEqual((n, self.action_dim), actions.size())


class TestCritic(unittest.TestCase):
    def setUp(self):
        self.state_dim = 24 * 2
        self.action_dim = 2 * 2

        self.critic = Critic(state_dim=self.state_dim, action_dim=self.action_dim, fc1_units=64, fc2_units=64, seed=0)

    def test_forward(self):
        n = 2
        states = torch.tensor(np.random.random_sample((n, self.state_dim)), dtype=torch.float)
        actions = torch.tensor(np.random.random_sample((n, self.action_dim)), dtype=torch.float)

        values = self.critic.forward(states, actions)
        self.assertEqual((n, 1), values.size())
