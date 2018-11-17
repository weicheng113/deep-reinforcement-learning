import unittest
import numpy as np
from trajectory import ParallelTrajectory


class TestParallelTrajectory(unittest.TestCase):
    def setUp(self):
        self.trajectory = ParallelTrajectory(2)

    def test_discounted_returns(self):
        self.add_rewards(step=1, parallel_rewards=[1, 1], parallel_dones=[False, False])
        self.add_rewards(step=2, parallel_rewards=[1, 1], parallel_dones=[False, False])
        self.add_rewards(step=3, parallel_rewards=[1, 1], parallel_dones=[True, False])
        self.add_rewards(step=4, parallel_rewards=[1, 1], parallel_dones=[False, True])
        self.add_rewards(step=5, parallel_rewards=[1, 1], parallel_dones=[False, False])
        self.add_rewards(step=6, parallel_rewards=[1, 1], parallel_dones=[True, False])

        actual_returns = self.trajectory.discounted_returns(discount=0.9)
        expected_returns = [[2.71, 3.439],
                            [1.9, 2.71],
                            [1., 1.9],
                            [2.71, 1.],
                            [1.9, 1.9],
                            [1., 1.]]
        np.testing.assert_array_equal(actual_returns, np.array(expected_returns))

    def add_rewards(self, step, parallel_rewards, parallel_dones):
        i = step
        self.trajectory.add(
            parallel_states=np.array([i, i]),
            parallel_actions=np.array([i, i]),
            parallel_action_probs=np.array([0.5, 0.5]),
            parallel_rewards=np.array(parallel_rewards),
            parallel_next_states=np.array([i+1, i+1]),
            parallel_dones=np.array(parallel_dones))

    def test_rewards(self):
        a = [1, 2, 3, 4]
        b = a[2:] + a[:2]
        c = (np.array(a) + np.array(b))/2.0
        np.testing.assert_array_equal(c, np.array([2., 3., 2., 3.]))

    def test_action_probs(self):
        self.add_action_probs(step=1, parallel_states=[10, 20], parallel_action_probs=[0.10, 0.20])
        self.add_action_probs(step=2, parallel_states=[11, 21], parallel_action_probs=[0.11, 0.21])
        self.add_action_probs(step=3, parallel_states=[12, 22], parallel_action_probs=[0.12, 0.22])
        self.add_action_probs(step=4, parallel_states=[13, 23], parallel_action_probs=[0.13, 0.23])
        self.add_action_probs(step=5, parallel_states=[14, 24], parallel_action_probs=[0.14, 0.24])
        self.add_action_probs(step=6, parallel_states=[15, 25], parallel_action_probs=[0.15, 0.25])

        states, full_states, actions, action_probs, rewards, next_states, dones = self.trajectory.numpy()

        np.testing.assert_array_equal(states[:, 0], np.array([10, 11, 12, 13, 14, 15]))
        np.testing.assert_array_equal(action_probs[:, 0], np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15]))

        np.testing.assert_array_equal(states[:, 1], np.array([20, 21, 22, 23, 24, 25]))
        np.testing.assert_array_equal(action_probs[:, 1], np.array([0.20, 0.21, 0.22, 0.23, 0.24, 0.25]))

    def add_action_probs(self, step, parallel_states, parallel_action_probs):
        i = step
        self.trajectory.add(
            parallel_states=np.array(parallel_states),
            parallel_actions=np.array([i, i]),
            parallel_action_probs=np.array(parallel_action_probs),
            parallel_rewards=np.array([i, i]),
            parallel_next_states=np.array([i+1, i+1]),
            parallel_dones=np.array([False, False]))
