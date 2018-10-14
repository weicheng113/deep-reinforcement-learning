import unittest
from model import Actor, Critic
import numpy as np
import torch


class TestActor(unittest.TestCase):
    def setUp(self):
        self.state_dim = (2, 80, 80)

        self.actor = Actor(action_dim=3)

    def test_forward(self):
        n = 2
        batch = torch.tensor(np.random.random_sample((n, ) + self.state_dim), dtype=torch.float)

        actions, probs = self.actor.forward(batch)
        self.assertEqual((n, 1), actions.size())
        self.assertEqual((n, 1), probs.size())


class TestCritic(unittest.TestCase):
    def setUp(self):
        self.state_dim = (2, 80, 80)

        self.critic = Critic()

    def test_forward(self):
        n = 2
        batch = torch.tensor(np.random.random_sample((n, ) + self.state_dim), dtype=torch.float)

        values = self.critic.forward(batch)
        self.assertEqual((n, 1), values.size())
