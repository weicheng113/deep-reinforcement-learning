import unittest
from model import Actor
import numpy as np
import torch


class TestActor(unittest.TestCase):
    def setUp(self):
        self.state_dim = 10

        self.actor = Actor(action_dim=3, state_dim=self.state_dim, fc1_units=10, fc2_units=10, seed=0)

    def test_forward(self):
        n = 2
        batch = torch.tensor(np.random.random_sample((n, self.state_dim)), dtype=torch.float)

        actions_1, probs_1, _ = self.actor.forward(batch)
        actions_2, probs_2, _ = self.actor.forward(batch, actions_1.view(-1))
        np.testing.assert_array_equal(probs_1.cpu().detach().numpy(), probs_2.cpu().detach().numpy())
