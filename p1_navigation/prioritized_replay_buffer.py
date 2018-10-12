import numpy as np
from collections import namedtuple
import random
from sum_tree import SumTree


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            seed (int): random seed
        """
        self.memory = SumTree(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        # epsilon: small amount to avoid zero priority
        # alpha: [0~1] determines how much prioritization is used. with 0, we would get the uniform case
        # beta: Controls importance-sampling compensation. fully compensates for the non-uniform probabilities
        #   when beta=1. The unbiased nature of the updates is most important near convergence at the end of
        #   training, so we define a schedule on the exponent beta that starts from initial value and reaches 1
        #   only at the end of learning.

        self.epsilon = 0.01
        self.alpha = 0.6
        
        beta_start = 0.4
        self.beta_end = 1.0
        self.beta = beta_start
        beta_increments = 200
        self.beta_increment = (self.beta_end - beta_start)/beta_increments

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = self.experience(state, action, reward, next_state, done)
        p = self.memory.max_p()
        if p == 0:
            p = 1.0
        self.memory.add(p=p, data=experience)

    def sample(self, n):
        """Randomly sample a batch of experiences from memory."""
        experiences = []
        indices = []
        priorities = []
        segment = self.memory.total_p() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, experience) = self.memory.get(s)
            experiences.append(experience)
            indices.append(idx)
            priorities.append(p)
        priorities = np.array(priorities, dtype=np.float64)
        indices = np.array(indices, dtype=np.int32)

        # print(f"priorities: {priorities}")
        probs = priorities / self.memory.total_p()
        # print(f"probs: {probs}")
        # importance-sampling (IS) weights
        w_is = (self.memory.capacity * probs) ** (-self.beta)
        # print(f"w_IS: {w_IS}")
        w_is_normalized = w_is/w_is.max()
        # print(f"w_IS_normalized: {w_IS_normalized}")
        # w_is_normalized = torch.from_numpy(w_is_normalized).float().to(self.device)
        
        return experiences, indices, w_is_normalized

    def update_errors(self, indices, errors):
        priorities = [self._to_priority(e) for e in errors]
        for (idx, p) in zip(indices, priorities):
            self.memory.update(idx, p)

    def _to_priority(self, error):
        return (error + self.epsilon) ** self.alpha
    
    def increase_beta(self):
        if self.beta < self.beta_end:
            self.beta = min(self.beta_end, self.beta + self.beta_increment)

    def __len__(self):
        return len(self.memory)
