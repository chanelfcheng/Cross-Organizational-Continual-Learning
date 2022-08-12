import copy
import torch
import numpy as np

def reservoir(model, current_input: torch.tensor):
    outputs = model(current_input.float())
    probabilities = torch.nn.functional.softmax(outputs, dim=0)
    max_prob, _ = torch.max(probabilities, 0)

    if max_prob < 0.7:
        return True
    else:
        return False

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

    def add_data(self, model, examples, labels=None, logits=None):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits)
        
        for i in range(examples.shape[0]):
            uncertain = reservoir(model, examples[i])
            index = np.random.randint(0, self.num_examples) if self.num_examples > 0 else 0
            if uncertain:
                self.examples[index] = examples[i].to(self.device)
                                
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                
                if self.num_examples < self.buffer_size:
                    self.num_examples += 1

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
        ret_tuple = (self.examples.to(self.device),)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        
        return ret_tuple

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



