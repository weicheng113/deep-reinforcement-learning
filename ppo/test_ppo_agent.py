import unittest
from model import Actor, Critic
from ppo_agent import Agent
import numpy as np
import torch


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.state_dim = (2, 80, 80)
        self.action_dim = 3

        def create_actor():
            return Actor(action_dim=self.action_dim)

        def create_critic():
            return Critic()

        self.agent = Agent(create_actor=create_actor, create_critic=create_critic)

    def test_act(self):
        n = 2
        states = np.random.random_sample((n, ) + self.state_dim)

        actions, probs = self.agent.act(states)
        self.assertEqual((n, ), actions.shape)
        self.assertEqual((n, ), probs.shape)

    def test_discounted_returns(self):
        self.agent.discount = 0.9
        returns = self.agent.discounted_returns(np.array([1, 0, 2], dtype=np.float))
        np.testing.assert_equal(
            returns,
            np.array([1*1 + 0*0.9 + 2*0.81, 0*1 + 2*0.9, 2*1], dtype=np.float))

    def test_learn_policy(self):
        n_states = 3
        probs = np.random.random_sample(n_states)
        advantages = np.random.random_sample(n_states)
        states = np.random.random_sample((n_states, ) + self.state_dim)
        actions = np.random.randint(self.action_dim, size=n_states)

        self.agent.learn_policy(
            sampled_probs=probs,
            sampled_advantages=advantages,
            sampled_states=states,
            sampled_actions=actions)
        self.assertTrue(True)

    def test_learn_value(self):
        n_states = 3
        returns = np.random.random_sample(n_states)
        states = np.random.random_sample((n_states, ) + self.state_dim)

        self.agent.learn_value(states=states, sampled_returns=returns)
        self.assertTrue(True)

    def test_prob_entropy(self):
        n_states = 3
        old_probs = torch.tensor(np.random.random_sample(n_states))
        new_probs = torch.tensor(np.random.random_sample(n_states))

        entropy = self.agent.prob_entropy(old_probs=old_probs, new_probs=new_probs)
        self.assertEqual((n_states, ), entropy.shape)

    def test_learn(self):
        n_trajectories = 2
        n_states = 3

        probs = np.random.random_sample((n_trajectories, n_states))
        states = np.random.random_sample((n_trajectories, n_states) + self.state_dim)
        actions = np.random.randint(self.action_dim, size=(n_trajectories, n_states))
        rewards = np.random.random_sample((n_trajectories, n_states))

        self.agent.learn(
            action_probs=probs,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=None,
            dones=None)
        self.assertTrue(True)

    def test_reshape(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_equal(arr.reshape(-1), np.array([1, 2, 3, 4, 5, 6]))
