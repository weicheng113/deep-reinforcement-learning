from collections import namedtuple, deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
        """
        random.seed(seed)
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "full_state", "action", "full_action", "reward", "next_state", "full_next_state", "done"])

    def add(self, state, full_state, action, full_action, reward, next_state, full_next_state, done):
        """Add a new experience to memory."""

        e = self.experience(state, full_state, action, full_action, reward, next_state, full_next_state, done)
        self.memory.append(e)

    def sample(self, n):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=n)

        states = np.array([e.state for e in experiences if e is not None])
        full_states = np.array([e.full_state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        full_actions = np.array([e.full_action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        full_next_states = np.array([e.full_next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None])

        return states, full_states, actions, full_actions, rewards, next_states, full_next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def size(self):
        return len(self.memory)
