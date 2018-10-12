import numpy as np


class SumTree:
    """
    https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay
    Modified version of: https://github.com/jaara/AI-blog/blob/master/SumTree.py
    """
    data_idx = 0

    def __init__(self, capacity):
        self.capacity = capacity  # data capacity
        # size of parent nodes: capacity - 1; size of priority leaves: capacity
        self.data = np.zeros(capacity, dtype=object)
        self.tree = np.zeros(2*capacity - 1)

    def add(self, p, data):
        """
        Add data.

        p - priority of data.
        data - data.
        """
        tree_idx = self.data_idx + self.capacity - 1

        self.data[self.data_idx] = data
        self.update(tree_idx, p)

        self.data_idx += 1
        if self.data_idx >= self.capacity:
            self.data_idx = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx, change):
        while tree_idx != 0:
            parent = (tree_idx - 1) // 2
            self.tree[parent] += change
            tree_idx = parent

    def get(self, s):
        """
        get tree node based on given sample sum value.
        Tree:
                   42
                /     \
              29      13
             /  \     / \
           13   16   3  10
          / \   / \ / \ / \
         3  10 12 4 1 2 8 2

        Get 24: 42-->29-->16(11=24-13)-->12
        Result: index of 9, value of 12
        Reason: s of 24 fall in the sum up till 25 = 3+10+12
        """
        tree_idx = self._traverse(0, s)
        data_idx = tree_idx - self.capacity + 1

        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    def _traverse(self, tree_idx, s):
        if tree_idx >= (self.capacity - 1): # is leaf node.
            return tree_idx

        left = 2 * tree_idx + 1
        right = left + 1

        if s <= self.tree[left]:
            return self._traverse(left, s)
        else:
            return self._traverse(right, s-self.tree[left])

    def total_p(self):
        """
        Sum of priorities, which is equal to root.
        """
        return self.tree[0]

    def max_p(self):
        if len(self) == 0:
            return 0.0
        else:
            return np.max(self.tree[-self.capacity:])

    def __len__(self):
        return np.count_nonzero(self.data)
