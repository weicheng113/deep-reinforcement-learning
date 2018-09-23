import unittest
import numpy as np
from p1_navigation.sum_tree import SumTree


class TestSumTree(unittest.TestCase):
    def test_add(self):
        instance = SumTree(4)

        instance.add(p=1, data=1)
        np.testing.assert_array_equal([1, 1, 0, 1, 0, 0, 0], instance.tree)

        instance.add(p=2, data=2)
        np.testing.assert_array_equal([3, 3, 0, 1, 2, 0, 0], instance.tree)

    def test_sum(self):
        instance = TestSumTree.create_tree([1, 2, 3, 4])
        np.testing.assert_array_equal([10, 3, 7, 1, 2, 3, 4], instance.tree)

    @staticmethod
    def create_tree(sample):
        tree = SumTree(len(sample))
        for e in sample:
            tree.add(p=e, data=e)

        return tree

    def test_get(self):
        instance = TestSumTree.create_tree([3, 10, 12, 4, 1, 2, 8, 2])

        self.assertEqual((9, 12, 12), instance.get(24), 12)
        self.assertEqual((11, 1, 1), instance.get(30))

    def test_update(self):
        instance = TestSumTree.create_tree([1, 2, 3, 4])

        instance.update(tree_idx=4, p=2+2)
        np.testing.assert_array_equal([12, 5, 7, 1, 4, 3, 4], instance.tree)

        instance.update(tree_idx=5, p=3-1)
        np.testing.assert_array_equal([11, 5, 6, 1, 4, 2, 4], instance.tree)
