import os
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
import torch
from torch.nn import functional as F
from argparse import Namespace

from architectures import ARCHITECTURES
from architectures.mlp import MLP
from datasets import CIC_2018, USB_2021, CIC_CLASSES, USB_CLASSES, UNKNOWN, CIC_PATH, USB_PATH, get_support
from datasets.continual_dataset import ContinualDataset
from utils.buffer import Buffer, ModifiedBuffer
from utils.focal_loss import FocalLoss
from utils.train_eval import train_continual
from utils import create_if_not_exists

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

        # outputs = self.net(inputs)
        # loss = self.loss(outputs, labels)
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
    
def train_mlp(args):
    name = args.arch + '-' + args.exp
    include_categorical = args.categorical

    if args.exp == 'continual-cic':
        dataset_names = [CIC_2018]
        classes = CIC_CLASSES + UNKNOWN
        dataset_paths = [CIC_PATH]
    if args.exp == 'continual-usb':
        dataset_names = [USB_2021]
        classes = USB_CLASSES + UNKNOWN
        dataset_paths = [USB_PATH]
    if args.exp == 'continual-cic-usb':
        dataset_names = [CIC_2018, USB_2021]
        classes = list(dict.fromkeys(CIC_CLASSES + USB_CLASSES + UNKNOWN))
        dataset_paths = [CIC_PATH, USB_PATH]
    
    dataset = ContinualDataset(dataset_names, classes, dataset_paths, args)
    print('\nContinual dataset:', dataset_names)
    print('Classes:', classes)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Get class weights
    train_support = get_support(dataset.train_dataset)
    train_support = dict(sorted(train_support.items()))
    weights = 1 / np.array( list( train_support.values() ) )
    weights = weights / np.sum(weights) * dataset.num_classes
    weights = torch.Tensor(weights).to(device)

    # Initialize model
    architecture = MLP(88 if include_categorical else 76, dataset.num_classes)
    criterion = FocalLoss(beta=weights, gamma=args.gamma)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(architecture.parameters(), lr=args.lr)
    # optimizer = SGD(architecture.parameters(), lr=args.lr)
    if args.alpha > 0:
        model = Der(architecture, criterion, optimizer, args)
    else:
        model = Er(architecture, criterion, optimizer, args)
        for key in range(dataset.num_classes):
            model.buffer.buffer_content[key] = 0

    out_path = os.path.join('./out/', name)
    create_if_not_exists(out_path)

    counter = 0
    while os.path.exists(os.path.join(out_path, f'log_{counter}.txt')):
        counter += 1
    
    with open(os.path.join(out_path, f'log_{counter}.txt'), 'w') as file:
        file.write('Config for run: %s\n' % name)
        file.write('CATEGORICAL: %s\n' % args.categorical)
        file.write('NUM_ROUNDS: %d\n' % args.num_rounds)
        file.write('NUM_EPOCHS: %d\n' % args.n_epochs)
        file.write('BATCH_SIZE: %d\n' % args.batch_size)
        file.write('MINIBATCH_SIZE: %d\n' % args.minibatch_size)
        file.write('BUFFER_SIZE: %d\n' % args.buffer_size)
        file.write('BUFFER_STRATEGY: %s\n' % args.buffer_strategy)
        if args.alpha > 0: file.write('ALPHA: %f\n' % args.alpha)
        file.write('GAMMA: %f\n' % args.gamma)
        file.write('LR: %e\n' % args.lr)
    
    train_continual(model, dataset, out_path, counter, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=ARCHITECTURES, help='The model architecture')
    parser.add_argument('--exp', type=str, required=True, choices=['continual-cic', 'continual-usb', 'continual-cic-usb'], help='The experimental setup for continual learning')
    parser.add_argument('--categorical', default=True, help='Option to include or not include categorical features in the model')
    parser.add_argument('--num-rounds', type=int, default=3)
    parser.add_argument('--n-epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Number of samples per batch')
    parser.add_argument('--minibatch-size', type=int, default=256, help='Number of samples per minibatch')
    parser.add_argument('--buffer-size', type=int, default=1000, help='Maximum number of samples the buffer can hold')
    parser.add_argument('--buffer-strategy', type=str, default='uncertainty', help='Strategy to use for sampling to the buffer')
    parser.add_argument('--alpha', type=float, default=-1, help='Balance parameter for balancing trade-off between past and current samples')
    parser.add_argument('--gamma', type=float, default=2, help='Focus parameter for focal loss')
    parser.add_argument('--sampling-threshold', type=float, default=0.8, help='Probability threshold for sampling to the buffer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate during training')

    args = parser.parse_args()

    if args.arch == 'mlp':
        train_mlp(args)

if __name__ == '__main__':
    main()