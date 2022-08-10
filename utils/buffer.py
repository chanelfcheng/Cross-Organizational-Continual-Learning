import copy
import torch
import numpy as np
from typing import Tuple
from torchvision import transforms

def reservoir(model, current_input: torch.tensor):
    outputs = model(current_input.float())
    probabilities = torch.nn.functional.softmax(outputs, dim=0)
    max_prob, _ = torch.max(probabilities, 0)

    if max_prob < 0.7:
        return True
    else:
        return False

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

    def add_data(self, model, examples, labels=None):
        
        for i in range(examples.shape[0]):
            uncertain = reservoir(model, examples[i])
            if uncertain:
                self.examples[self.position] = examples[i].to(self.device)
                self.labels[self.position] = labels[i].to(self.device)
                self.position += 1

                if self.position >= self.buffer_size:
                    self.position = 0
                
                if self.num_examples < self.buffer_size:
                    self.num_examples += 1

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



