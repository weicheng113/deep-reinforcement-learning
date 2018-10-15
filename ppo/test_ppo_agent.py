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
        rewards = np.array([1, 0, 2, 1], dtype=np.float)
        dones = np.array([False, False, False, True])
        returns = self.agent.discounted_returns(rewards, dones)
        np.testing.assert_equal(
            returns,
            np.array([(1*1 + 0*0.9 + 2*0.81 + 0), (0*1 + 2*0.9 + 0), (2*1 + 0), 0], dtype=np.float))

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
        dones = np.random.choice(a=[False, True], size=(n_trajectories, n_states), p=[0.9, 0.1])

        self.agent.learn(
            action_probs=probs,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=None,
            dones=dones)
        self.assertTrue(True)

    def test_reshape(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_equal(arr.reshape(-1), np.array([1, 2, 3, 4, 5, 6]))

    def test_learn_policy_progress(self):
        n_states = 64
        probs = np.random.random_sample(n_states)
        advantages = np.random.random_sample(n_states)
        returns = np.random.random_sample(n_states)
        states = np.random.random_sample((n_states, ) + self.state_dim)
        actions = np.random.randint(self.action_dim, size=n_states)

        policy_losses = []
        value_losses = []
        for _ in range(10):
            policy_loss, value_loss = self.agent.learn_policy(
                sampled_probs=probs,
                sampled_advantages=advantages,
                sampled_states=states,
                sampled_actions=actions,
                sampled_returns=returns)
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        self.assertEqual(policy_losses, sorted(policy_losses, reverse=True))
        self.assertEqual(value_losses, sorted(value_losses, reverse=True))

    def test_learn_progress(self):
        n_trajectories = 4
        n_states = 64

        probs = np.random.random_sample((n_trajectories, n_states))
        states = np.random.random_sample((n_trajectories, n_states) + self.state_dim)
        actions = np.random.randint(self.action_dim, size=(n_trajectories, n_states))
        rewards = np.random.random_sample((n_trajectories, n_states))
        dones = np.random.choice(a=[False, True], size=(n_trajectories, n_states), p=[0.9, 0.1])

        policy_losses, value_losses = self.agent.learn(
            action_probs=probs,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=None,
            dones=dones)

        self.assertEqual(policy_losses, sorted(policy_losses, reverse=True))
        self.assertEqual(value_losses, sorted(value_losses, reverse=True))
