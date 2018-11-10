import numpy as np


class SoccerEnvWrapper:
    def __init__(self, env, train_mode=False):
        self.env = env
        self.brain_names = env.brain_names
        self.train_mode = train_mode
        self.goalie_action_size = 4
        self.num_goalies = 2
        self.striker_action_size = 6
        self.num_strikers = 2
        self.state_size = 336
        self.num_agents = self.num_goalies + self.num_strikers

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)
        states = []
        for brain_name in self.env.brain_names:
            states.extend(env_info[brain_name].vector_observations)
        return np.array(states, np.float)

    def step(self, actions):
        actions_per_brain = int(len(actions)/len(self.env.brain_names))
        brain_actions = dict()
        for i in range(len(self.env.brain_names)):
            start = i * actions_per_brain
            brain_actions[self.env.brain_names[i]] = actions[start: start+actions_per_brain]

        env_info = self.env.step(brain_actions)

        next_states = []
        rewards = []
        dones = []
        for brain_name in self.env.brain_names:
            next_states.extend(env_info[brain_name].vector_observations)
            rewards.extend(env_info[brain_name].rewards)
            dones.extend(env_info[brain_name].local_done)

        return np.array(next_states, np.float), np.array(rewards, np.float), np.array(dones, np.bool)
