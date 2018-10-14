import gym
from gym.spaces import Discrete


class PongDeterministic:
    def __init__(self):
        self.env = gym.make('PongDeterministic-v4')
        self.action_space = Discrete(3)
        self.observation_space = self.env.observation_space
        self.action_map = {0: 0, 1: 4, 2: 5}

    @staticmethod
    def get_action_meanings():
        return ['NOOP', 'RIGHTFIRE', 'LEFTFIRE']

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(self.action_map[action])

    def render(self):
        self.render()

    def close(self):
        self.env.close()


