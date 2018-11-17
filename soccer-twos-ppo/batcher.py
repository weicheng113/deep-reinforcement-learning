import numpy as np


class Batcher:
    def __init__(self, num_data_points, batch_size):
        self.data_indices = np.arange(num_data_points, dtype=np.long)
        self.batch_size = batch_size

    def batches(self):
        next_batch_start = 0
        data_size = self.data_indices.size
        while next_batch_start + self.batch_size < data_size:
            batch_start = next_batch_start
            next_batch_start = next_batch_start + self.batch_size
            yield self.data_indices[batch_start: next_batch_start]

    def shuffle(self):
        np.random.shuffle(self.data_indices)
