import numpy as np
from collections import namedtuple
import random
from sum_tree import SumTree
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.tree = SumTree(buffer_size)

        # Small amount to avoid zero priority
        self.priority_epsilon = 0.01
        # [0~1] determines how much prioritization is used. with 0, we would get the uniform case.
        self.priority_alpha = 0.6
        # Controls importance-sampling compensation. fully compensates for the non-uniform probabilities when beta=1
        # The unbiased nature of the updates is most important near convergence at the end of training,
        # so we define a schedule on the exponent beta that starts from initial value and reaches 1 only at the end
        # of learning.
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = self.experience(state, action, reward, next_state, done)
        p = self.tree.max_p()
        if p == 0:
            p = self._calculate_priority(error=0)
        self.tree.add(p=p, data=experience)

    def _calculate_priority(self, error):
        return (error + self.priority_epsilon) ** self.priority_alpha

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = []
        tree_indices = []
        priorities = []
        segment = self.tree.total_p() / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (tree_idx, p, experience) = self.tree.get(s)
            experiences.append(experience)
            tree_indices.append(tree_idx)
            priorities.append(p)
        priorities = np.array(priorities, dtype=np.float64)
        tree_indices = np.array(tree_indices, dtype=np.int32)

        probs = priorities / self.tree.total_p()
        #importance-sampling (IS) weights
        w_IS = (self.tree.capacity * probs) ** (-self.beta)
        w_IS_normalized = w_IS/np.max(w_IS)
        w_IS_normalized = torch.from_numpy(w_IS_normalized).float().to(device)

        self.beta = min(1., self.beta + self.beta_increment_per_sampling)
        return self._vstack_experiences(experiences) + (tree_indices, w_IS_normalized)

    def _vstack_experiences(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def update_errors(self, indices, errors):
        priorities = [self._calculate_priority(e.data) for e in errors]
        for (tree_idx, p) in zip(indices, priorities):
            self.tree.update(tree_idx, p)
    def __len__(self):
        return len(self.tree)
