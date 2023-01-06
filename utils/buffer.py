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

def uncertainty_reservoir(model, sampling_threshold, current_input: torch.tensor):
    outputs = model(current_input.float())
    probabilities = torch.nn.functional.softmax(outputs, dim=0) # [0.1 0.2 0.2 0.1]
    max_prob, _ = torch.max(probabilities, 0)

    if max_prob < sampling_threshold: # sampling threshold (slightly uncertain)
        return 1
    elif max_prob < 0.3: # minimum threshold (highly uncertain)
        return 2
    else:
        return 0 # wow i am so confident that i don't need to sample this


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

class ModifiedBuffer:
    def __init__(self, buffer_size, sampling_threshold, device):
        self.buffer_size = buffer_size
        self.sampling_threshold = sampling_threshold
        self.device = device
        self.position = 0
        self.num_examples = 0
        self.attributes = ['examples', 'labels', 'logits']
        self.buffer_content = {}
        self.batch_count = 1

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
            uncertain = uncertainty_reservoir(model, self.sampling_threshold, examples[i])
            index = np.random.randint(0, self.num_examples) if self.num_examples > 0 else 0
            if uncertain != 0:
                if self.num_examples < self.buffer_size:
                    self.num_examples += 1
                # else:
                #     if self.labels[index].item() in self.buffer_content:
                #         if self.buffer_content[self.labels[index].item()] > 0:
                #             self.buffer_content[self.labels[index].item()] -= 1
                #         # else:
                #         #     self.buffer_content.pop(self.labels[index].item(), None)

                self.examples[index] = examples[i].to(self.device)
                        
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device) if uncertain == 1 else model.net.num_classes-1
                    if self.labels[index].item() not in self.buffer_content:
                        self.buffer_content[self.labels[index].item()] = 1
                    else:
                        # if sum(self.buffer_content.values()) < self.buffer_size:
                        #     self.buffer_content[self.labels[index].item()] += 1
                        self.buffer_content[self.labels[index].item()] += 1

                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
        
        self.batch_count += 1

        return self.buffer_content

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



