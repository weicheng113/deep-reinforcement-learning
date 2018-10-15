import unittest
from pong_wrapper import PongDeterministic
from pong_helper import PongHelper


class TestPongHelper(unittest.TestCase):
    def setUp(self):
        self.env = PongDeterministic()
        self.env.reset()

    def test_preprocess(self):
        [s] = self.some_observations(1)

        result = PongHelper.preprocess(s)

        self.assertEqual(result.shape, (80, 80))

    def some_observations(self, n):
        return [self.env.step(self.env.action_space.sample())[0] for _ in range(n)]

    def test_stack_frames(self):
        [s1, s2] = self.some_observations(2)

        result = PongHelper.stack_frames([s1, s2])

        self.assertEqual(result.shape, (2, 80, 80))
