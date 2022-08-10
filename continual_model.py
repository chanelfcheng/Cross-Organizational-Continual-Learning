import os
import argparse

import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
import torch
import torchvision
from torch.nn import functional as F
from argparse import Namespace

from architectures import ARCHITECTURES
from architectures.mlp import MLP
from datasets import CIC_2018, USB_2021, CIC_CLASSES, USB_CLASSES, CIC_PATH, USB_PATH
from datasets.continual_dataset import ContinualDataset
from utils.buffer import Buffer
from utils.focal_loss import FocalLoss
from utils.train_eval import train_continual

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, architecture: nn.Module, criterion: nn.Module,
                args: Namespace) -> None:
        super(ContinualModel, self).__init__()

        self.net = architecture
        self.loss = criterion
        self.args = args
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
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

    def __init__(self, architecture, criterion, args):
        super(Er, self).__init__(architecture, criterion, args)
        self.buffer = Buffer(self.args.buffer_size, self.net.num_in_features, self.device)

    def observe(self, inputs, labels):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels.flatten()))

        outputs = self.net(inputs.float())
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(self.net, examples=inputs,
                             labels=labels)

        return loss.item()
    

class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, architecture, criterion, args):
        super(Der, self).__init__(architecture, criterion, args)
        self.buffer = Buffer(self.args.buffer_size, self.device)

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
        self.buffer.add_data(examples=inputs, logits=outputs.data)

        return loss.item()
    
def train_mlp(args):
    name = args.arch + '-' + args.exp
    include_categorical = args.categorical

    if args.exp == 'continual-cic':
        dataset_names = [CIC_2018]
        classes = list(set(CIC_CLASSES))
        dataset_paths = [CIC_PATH]
    if args.exp == 'continual-usb':
        dataset_names = [USB_2021]
        classes = list(set(USB_CLASSES))
        dataset_paths = [USB_PATH]
    if args.exp == 'continual-cic-usb':
        dataset_names = [CIC_2018, USB_2021]
        classes = list(set(CIC_CLASSES + USB_CLASSES))
        dataset_paths = [CIC_PATH, USB_PATH]
    
    dataset = ContinualDataset(dataset_names, classes, dataset_paths, args)
    train = 'train'
    test = 'test'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize model
    architecture = MLP(88 if include_categorical else 76, dataset.num_classes)
    criterion = FocalLoss(gamma=2)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(architecture.parameters(), lr=args.lr)
    model = Er(architecture, criterion, args)

    out_dir = os.path.join('./out/', name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, 'config.txt'), 'w') as file:
        file.write('Config for run: %s\n' % name)
        file.write('NUM_EPOCHS: %d\n' % args.n_epochs)
        # file.write('WARMUP_EPOCHS: %d\n' % 2)
        file.write('LR: %e\n' % args.lr)
        # file.write('MIN_LR: %e\n' % 1e-6)
        # file.write('WARMUP_LR: %e\n' % args.warmup_lr)
        file.write('BATCH_SIZE: %d\n' % args.batch_size)
    
    train_continual(model, dataset, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=ARCHITECTURES, help='The model architecture')
    parser.add_argument('--exp', type=str, required=True, choices=['continual-cic', 'continual-usb', 'continual-cic-usb'], help='The experimental setup for continual learning')
    parser.add_argument('--categorical', default=True, help='Option to include or not include categorical features in the model')
    parser.add_argument('--num-rounds', type=int, default=3)
    parser.add_argument('--n-epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of samples per batch')
    parser.add_argument('--minibatch-size', type=int, default=16, help='Number of samples per minibatch')
    parser.add_argument('--buffer-size', type=int, default=1000, help='Maximum number of samples the buffer can hold')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha term for balancing trade-off between past and current loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate during training')

    args = parser.parse_args()

    if args.arch == 'mlp':
        train_mlp(args)

if __name__ == '__main__':
    main()