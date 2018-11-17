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
        states = np.zeros((4, self.state_size), dtype=np.float)
        states[:2, :] = env_info[self.env.brain_names[0]].vector_observations
        states[2:, :] = env_info[self.env.brain_names[1]].vector_observations
        return np.array(states, np.float)

    def step(self, actions):
        actions_per_brain = int(len(actions)/len(self.env.brain_names))
        brain_actions = dict()
        for i in range(len(self.env.brain_names)):
            start = i * actions_per_brain
            brain_actions[self.env.brain_names[i]] = actions[start: start+actions_per_brain]

        env_info = self.env.step(brain_actions)

        next_states = np.zeros((4, self.state_size), dtype=np.float)
        rewards = np.zeros(4, dtype=np.float)
        dones = np.zeros(4, dtype=np.bool)

        next_states[:2, :] = env_info[self.env.brain_names[0]].vector_observations
        next_states[2:, :] = env_info[self.env.brain_names[1]].vector_observations
        rewards[:2] = env_info[self.env.brain_names[0]].rewards
        rewards[2:] = env_info[self.env.brain_names[1]].rewards
        dones[:2] = env_info[self.env.brain_names[0]].local_done
        dones[2:] = env_info[self.env.brain_names[1]].local_done
        # for brain_name in self.env.brain_names:
        #     next_states.extend(env_info[brain_name].vector_observations)
        #     rewards.extend()
        #     dones.extend(env_info[brain_name].local_done)

        # peer_rewards = rewards[2:] + rewards[:2]
        # team_avg_rewards = (np.array(rewards) + np.array(peer_rewards))/2.0
        # return np.array(next_states, np.float), np.array(team_avg_rewards, np.float), np.array(dones, np.bool)
        return next_states, rewards, dones

