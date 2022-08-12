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
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.position = 0
        self.num_examples = 0
        self.attributes = ['examples', 'labels', 'logits']

    def __len__(self):
        return min(self.num_examples, self.buffer_size)

    def init_tensors(self, examples, labels, logits):
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str=='labels' else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits)
        
        for i in range(examples.shape[0]):
            index = reservoir(self.num_examples, self.buffer_size)
            self.num_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)

    def get_data(self, size, return_index=False):
        # Randomly get a sample from the buffer
        if size > min(self.num_examples, self.examples.shape[0]):
            size = min(self.num_examples, self.examples.shape[0])
        
        choice = np.random.choice(min(self.num_examples, self.examples.shape[0]), size=size, replace=False)

        ret_tuple = (self.examples[choice].to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice].to(self.device),)

        return ret_tuple

    def get_all_data(self):
        if hasattr(self, 'labels'):
            return self.examples.to(self.device), self.labels.to(self.device)
        if hasattr(self, 'logits'):
            return self.examples.to(self.device), self.logits.to(self.device)

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



