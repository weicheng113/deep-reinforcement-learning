import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.i_episode = 1
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.gamma = 0.03
        self.discount = 1.0
        
    def next_episode(self):
        self.i_episode += 1
        self.epsilon = self.epsilon_decay**(self.i_episode-1)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = self.get_action_probs(state)
        return np.random.choice(self.nA, p=probs)
    
    def get_action_probs(self, state):
        action_values = self.Q[state]
        probs = np.full(self.nA, self.epsilon/self.nA)
        probs[np.argmax(action_values)] += 1 - self.epsilon
        return probs

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.sarsaMax(state, action, reward, next_state, done)
#         self.expectedSarsa(state, action, reward, next_state, done)
        if done:
            self.next_episode()
        
    def expectedSarsa(self, state, action, reward, next_state, done):
        if done:
            expected_next_state_value = 0
        else:
            action_probs = self.get_action_probs(state)
            expected_next_state_value = np.dot(action_probs, self.Q[next_state])
        
        target = reward + self.discount * expected_next_state_value
        action_value = self.Q[state][action]
        self.Q[state][action] = action_value + self.gamma*(target - action_value)
        
    def sarsaMax(self, state, action, reward, next_state, done):
        if done:
            max_next_action_value = 0
        else:
            max_next_action_value = max(self.Q[next_state])
        
        target = reward + self.discount * max_next_action_value
        action_value = self.Q[state][action]
        self.Q[state][action] = action_value + self.gamma*(target - action_value)