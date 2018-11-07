import gym
from gym.spaces import Discrete
from gym.spaces import Box
import numpy as np


class PongDeterministic:
    default_background_color = np.array([144, 72, 17])

    def __init__(self, skip_num_frames=1):
        self.env = gym.make('PongDeterministic-v4')
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=255, shape=(2, 80, 80), dtype=np.uint8)
        self.action_map = {0: 0, 1: 4, 2: 5}
        self.last_frame = np.zeros((80, 80))
        self.skip_num_frames = skip_num_frames

    @staticmethod
    def get_action_meanings():
        return ['NOOP', 'RIGHTFIRE', 'LEFTFIRE']

    def reset(self):
        raw_frame = self.env.reset()
        frame = PongDeterministic.preprocess(raw_frame)
        state = PongDeterministic.stack_frame(np.zeros((80, 80)), frame)

        self.last_frame = frame
        return state, frame

    @staticmethod
    def stack_frame(f1, f2):
        return np.vstack((f1[np.newaxis, :], f2[np.newaxis, :]))

    @staticmethod
    def preprocess(frame, background_color=default_background_color):
        # image of shape: (210, 160, 3)
        # horizontally (210 - 50) / 2 = 80, vertically 160 / 2 = 80
        processed = np.mean(frame[34:-16:2, ::2] - background_color, axis=-1) / 255.
        # image of shape: (80, 80)
        return processed

    def step(self, action):
        raw_frame, reward, done, info = self.env.step(self.action_map[action])
        if not done:
            reward += self.skip_frames()
        frame = PongDeterministic.preprocess(raw_frame)
        state = PongDeterministic.stack_frame(self.last_frame, frame)

        self.last_frame = frame
        return state, reward, done, info, raw_frame

    def skip_frames(self):
        total_reward = 0
        for _ in range(self.skip_num_frames):
            _, reward, done, _ = self.env.step(0)
            total_reward += reward
            if done:
                break
        return total_reward

    def render(self):
        self.render()

    def close(self):
        self.env.close()


