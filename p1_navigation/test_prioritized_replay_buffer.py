import unittest
import numpy as np
from prioritized_replay_buffer import PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = PrioritizedReplayBuffer(prioritized_memory=self.memory, buffer_size=4, seed=0, device="cpu")
        self.memory = self.buffer.memory

    def test_add(self):
        self.buffer.add(state=[1, 2, 3], action=0, reward=0, next_state=[4, 5, 6], done=False)
        priority_one = self.buffer._calculate_priority(1)
        np.testing.assert_array_equal([priority_one, priority_one, 0, priority_one, 0, 0, 0], self.memory.tree)

    def test_sample(self):
        self.buffer.add(state=[1, 2, 3], action=0, reward=0, next_state=[4, 5, 6], done=False)
        self.buffer.add(state=[4, 5, 6], action=0, reward=0, next_state=[7, 8, 9], done=False)
        self.buffer.add(state=[7, 8, 9], action=0, reward=0, next_state=[10, 11, 12], done=False)

        sample = self.buffer.sample(2)
        print(sample)

