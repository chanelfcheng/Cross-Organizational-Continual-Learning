import os
import time
from datetime import timedelta
from argparse import Namespace

import numpy as np

import torch.nn as nn
import torch
from torch.nn import functional as F

from sklearn.metrics import classification_report

from utils.buffer import Buffer, ModifiedBuffer
from utils.progress import progress_bar

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, architecture: nn.Module, criterion: nn.Module, optimizer: nn.Module,
                args: Namespace) -> None:
        super(ContinualModel, self).__init__()

        self.net = architecture
        self.loss = criterion
        self.args = args
        self.opt = optimizer
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :return: the value of the loss function
        """
        pass

class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, architecture, criterion, optimizer, args):
        super(Er, self).__init__(architecture, criterion, optimizer, args)
        self.buffer = ModifiedBuffer(self.args.buffer_size, self.args.sampling_threshold, self.device) if self.args.buffer_strategy == 'uncertainty' \
            else Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels):
        self.buffer.add_data(self.net, examples=inputs,labels=labels) if self.args.buffer_strategy == 'uncertainty' \
        else self.buffer.add_data(examples=inputs, labels=labels)

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size)

            # support = {}
            # for label in np.array(buf_labels.detach().cpu()):
            #     if label not in support:
            #         support[label] = 1
            #     else:
            #         support[label] += 1
            # print(support)

            # inputs = torch.cat((inputs, buf_inputs))
            # labels = torch.cat((labels, buf_labels))

        outputs = self.net(buf_inputs)

        loss = self.loss(outputs, buf_labels)
        loss.backward()
        self.opt.step()

        return loss.item()
    

class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, architecture, criterion, optimizer, args):
        super(Der, self).__init__(architecture, criterion, optimizer, args)
        self.buffer = ModifiedBuffer(self.args.buffer_size, self.args.sampling_threshold, self.device) if self.args.buffer_strategy == 'uncertainty' \
            else Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(self.net, examples=inputs, logits=outputs.data) if self.args.buffer_strategy == 'uncertainty' \
            else self.buffer.add_data(examples=inputs, labels=labels)

        return loss.item()

def train_continual(model, dataset, out_path, counter, args):
    print('\nTraining phase')
    print('Current malicious class:',  list(dataset.label_mapping.keys())[dataset.train_classes[1]])
    model.net.to(model.device)

    model.net.train()
    i = 0
    start = time.time()
    while not dataset.train_over:
        if i == 0:
            max_progress = sum(dataset.active_remaining_training_items) // args.batch_size
        
        if not (i + 1) % (max_progress + 1):
            eval_continual(model, dataset, out_path, counter)

            for key in model.buffer.buffer_content:
                model.buffer.buffer_content[key] = (model.buffer.buffer_content[key] // (model.buffer.batch_count)) / args.batch_size


            print("samples saved to buffer:", model.buffer.buffer_content)
            # print("number of batches:", model.buffer.batch_count)
            
            model.buffer.batch_count = 0

            for key in model.buffer.buffer_content:
                model.buffer.buffer_content[key] = 0

            model.buffer.sample_count = 0
            model.buffer.max_probability_sum = 0
            model.buffer.confidence_sum = 0

            i = 0
        else:
            i += 1
        
        inputs, labels = dataset.get_train_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        loss = model.observe(inputs.float(), labels)

        progress_bar(i, max_progress, dataset.completed_rounds + 1, 'C', loss)
    
    time_elapsed = time.time() - start
    print(f'\nTraining complete in {timedelta(seconds=round(time_elapsed))}')

    with open(os.path.join(out_path, f'log_{counter}.txt'), 'a') as file:
        file.write(f'\nTraining complete in {timedelta(seconds=round(time_elapsed))}\n')

    torch.save(model.state_dict(), os.path.join(out_path, f'model_{counter}.pt'))

    eval_continual(model, dataset, out_path, counter)

def eval_continual(model, dataset, out_path, counter):
    print('\nEvaluation phase')
    model.net.eval()
    dataset.test_over = False
    dataset.test_class = 0
    start_test = True
    
    while not dataset.test_over:
        inputs, labels = dataset.get_test_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs.float())
        _, preds = torch.max(outputs.data, 1)

        if start_test:
            all_preds = preds.float().cpu()
            all_labels = labels.float()
            start_test = False
        else:
            all_preds = torch.cat((all_preds, preds.float().cpu()), 0)
            all_labels = torch.cat((all_labels, labels.float()), 0)
    
    all_labels = all_labels.detach().cpu().numpy()
    all_preds = all_preds.detach().cpu().numpy()

    report = classification_report(all_labels, all_preds, target_names=dataset.classes, digits=4)
    print(report)

    with open(os.path.join(out_path, f'log_{counter}.txt'), 'a') as file:
        file.write(f'\n{report}\n')
    
    return report