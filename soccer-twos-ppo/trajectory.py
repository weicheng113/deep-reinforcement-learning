import numpy as np


class ParallelTrajectory:
    def __init__(self, n):
        self.n_parallels = n
        self.states = []
        self.full_states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, parallel_states, parallel_actions, parallel_action_probs,
            parallel_rewards, parallel_next_states, parallel_dones):
        full_state = parallel_states.reshape(-1)
        self.states.append(parallel_states)
        self.full_states.append(full_state)
        self.actions.append(parallel_actions)
        self.action_probs.append(parallel_action_probs)
        self.rewards.append(parallel_rewards)
        self.next_states.append(parallel_next_states)
        self.dones.append(parallel_dones)

    def discounted_returns(self, discount, last_return=None):
        running_return = np.zeros(self.n_parallels, dtype=np.float)
        if last_return:
            running_return = last_return

        n_rewards = len(self.rewards)
        returns = np.zeros((n_rewards, self.n_parallels), dtype=np.float)
        # for i, dones in enumerate(self.dones):
        #     if np.any(dones):
        #         print(f"step {i}, dones: {dones}")
        for i in reversed(range(n_rewards)):
            rewards = np.array(self.rewards[i])
            dones = np.array(self.dones[i]).astype(np.uint8)
            running_return = rewards + discount * (1.0-dones) * running_return
            returns[i, :] = running_return
        return returns

    def numpy(self):
        return (np.array(self.states, dtype=np.float),
                np.array(self.full_states, dtype=np.float),
                np.array(self.actions, dtype=np.long),
                np.array(self.action_probs, dtype=np.float),
                np.array(self.rewards, dtype=np.float),
                np.array(self.next_states, dtype=np.float),
                np.array(self.dones, dtype=np.bool))

    def clear(self):
        self.states = []
        self.full_states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
