import os

import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
import torch
import torchvision
from torch.nn import functional as F
from argparse import Namespace

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

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
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

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args):
        super(Er, self).__init__(backbone, loss, args)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
    

class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args):
        super(Der, self).__init__(backbone, loss, args)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

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
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
    
def train_mlp(args):
    name = args.arch + '-' + args.exp
    include_categorical = args.categorical

    if args.exp == 'cic-only':
        data_sets = [CIC_2018]
        classes = list(set(CIC_CLASSES))
        data_paths = [CIC_PATH]
    if args.exp == 'usb-only':
        data_sets = [USB_2021]
        classes = list(set(USB_CLASSES))
        data_paths = [USB_PATH]
    if args.exp == 'cic-usb':
        data_sets = [CIC_2018, USB_2021]
        classes = list(set(CIC_CLASSES + USB_CLASSES))
        data_paths = [CIC_PATH, USB_PATH]
    
    cd = ContinualDataset(data_sets, classes, data_paths, args)
    train = 'train'
    test = 'test'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize model
    model = MLP(88 if include_categorical else 76, cd.num_classes)

    model = model.to(device)

    criterion = FocalLoss(gamma=2)
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience)

    # Could make this a command line argument
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
    
    trained_model = train_continual(model)