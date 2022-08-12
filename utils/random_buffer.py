import copy
import torch
import numpy as np

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

class Buffer:
    def __init__(self, buffer_size, num_features, device):
        self.buffer_size = buffer_size
        self.num_features = num_features
        self.device = device
        self.position = 0
        self.num_examples = 0
        self.examples = torch.zeros((self.buffer_size, self.num_features), dtype=torch.float32, device=self.device)
        self.labels = torch.zeros((self.buffer_size, 1), dtype=torch.int64, device=self.device)

    def __len__(self):
        return min(self.num_examples, self.buffer_size)

    def add_data(self, examples, labels=None):
        
        for i in range(examples.shape[0]):
            index = reservoir(self.num_examples, self.buffer_size)
            self.num_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                self.labels[index] = labels[i].to(self.device)

    def get_data(self, size, return_index=False):
        # Randomly get a sample from the buffer
        if size > min(self.num_examples, self.examples.shape[0]):
            size = min(self.num_examples, self.examples.shape[0])
        
        choice = np.random.choice(min(self.num_examples, self.examples.shape[0]), size=size, replace=False)

        return self.examples[choice].to(self.device), self.labels[choice].to(self.device)

    def get_all_data(self):
        return self.examples.to(self.device), self.labels.to(self.device)

    def is_empty(self):
        if self.num_examples == 0:
            return True
        else:
            return False

    def empty(self):
        self.examples = torch.zeros((self.buffer_size, self.num_features), dtype=torch.float32, device=self.device)
        self.labels = torch.zeros((self.buffer_size, 1), dtype=torch.int64, device=self.device)
        self.num_examples = 0
        self.position = 0



