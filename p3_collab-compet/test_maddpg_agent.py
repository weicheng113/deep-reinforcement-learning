import unittest
from model import Actor, Critic
from maddpg_agent import MultiAgent
import numpy as np
from replay_buffer import ReplayBuffer
from noise import OUNoise
import torch
from maddpg_agent_other import MADDPG_Agent
from collections import namedtuple


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.state_dim = 24
        self.action_dim = 2
        self.num_agents = 2
        seed = 2

        def create_actor():
            return Actor(state_dim=self.state_dim, action_dim=self.action_dim, fc1_units=64, fc2_units=64, seed=seed)

        def create_critic():
            return Critic(
                state_dim=self.state_dim * self.num_agents,
                action_dim=self.action_dim * self.num_agents,
                fc1_units=64,
                fc2_units=64,
                seed=seed)

        def create_noise():
            return OUNoise(size=self.action_dim, seed=seed)

        self.multi_agent = MultiAgent(
            num_agents=self.num_agents,
            create_actor=create_actor,
            create_critic=create_critic,
            replay_buffer=None,
            create_noise=create_noise,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            episodes_before_train=100,
            seed=seed)
        self.multi_agent_other = MADDPG_Agent(
            n_agents=self.num_agents,
            dim_obs=self.state_dim,
            dim_act=self.action_dim,
            batch_size=10,
            capacity=int(1e5),
            eps_b_train=100)

    def test_act(self):
        states = np.random.random_sample((self.num_agents, self.state_dim))

        actions = self.multi_agent.act(states)
        actions_other = self.multi_agent_other.act(torch.tensor(states).float()).data.numpy()
        np.testing.assert_array_equal(actions, actions_other)

    def test_learn(self):
        samples = self.n_samples(10)

        tensor_samples = self.multi_agent.to_tensor(samples=samples)
        actor_loss, critic_loss = self.multi_agent.learn(agent_i=0, samples=tensor_samples)

        critic_loss1, actor_loss1,  = self.multi_agent_other.learn_impl(transitions=self.to_experiences(tensor_samples), agent=0)

        self.assertEqual(critic_loss, critic_loss1.detach().item())
        self.assertEqual(actor_loss, actor_loss1.detach().item())

    def n_samples(self, n):
        states = []
        full_states = []
        actions = []
        rewards = []
        next_states = []
        full_next_states = []
        dones = []
        for _ in range(n):
            state = np.random.random_sample((self.num_agents, self.state_dim))
            full_state = state.reshape(-1)
            action = np.random.random_sample((self.num_agents, self.action_dim))
            reward = np.random.random_sample(self.num_agents)
            next_state = np.random.random_sample((self.num_agents, self.state_dim))
            full_next_state = next_state.reshape(-1)
            done = np.random.choice(a=[False, True], size=self.num_agents, p=[0.95, 0.05])

            states.append(state)
            full_states.append(full_state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            full_next_states.append(full_next_state)
            dones.append(done)
        return (np.array(states), np.array(full_states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(full_next_states), np.array(dones))

    @staticmethod
    def to_experiences(samples):
        states, full_states, actions, rewards, next_states, full_next_states, dones = samples
        experiences = [(s, a, ns, r) for s, a, r, ns in zip(states, actions, rewards, next_states)]
        return experiences

    def test_target_and_local_act(self):
        n_samples = 10
        states = np.random.random_sample((n_samples, self.num_agents, self.state_dim))
        states = torch.from_numpy(states).float()

        actions = self.multi_agent.target_act(states)
        self.assertEqual((n_samples, self.num_agents, self.action_dim), actions.shape)

        actions2 = self.multi_agent.local_act(states)
        self.assertEqual((n_samples, self.num_agents, self.action_dim), actions2.shape)
