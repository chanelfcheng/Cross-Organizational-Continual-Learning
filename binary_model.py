import argparse
import os

import numpy as np
import torch
from timm.scheduler import CosineLRScheduler
from torch import nn, optim
from torch.utils.data import RandomSampler, SequentialSampler

from architectures import ARCHITECTURES
from architectures.mlp import MLP
from datasets import CIC_2018, CIC_CLASSES, USB_2021, CIC_PATH, USB_CLASSES, USB_PATH
from datasets.binary_dataset import BinaryDataset
from utils.train_eval import train

def train_mlp(args):
    """
    Sets up PyTorch objects and trains the MLP model.
    :param args: The command line arguments
    :return: None
    """
    name = args.arch + '-' + args.exp
    include_categorical = args.categorical

    if args.exp == 'binary-cic-usb':
        dataset_names = [CIC_2018, USB_2021]
        classes = list(dict.fromkeys(CIC_CLASSES + USB_CLASSES))
        dataset_paths = [CIC_PATH, USB_PATH]
    
    bd = BinaryDataset(dataset_names, classes, dataset_paths, args)
    train_idx = 'train'
    test_idx = 'test'

    # Load dataset
    print('\nLoading dataset...')
    datasets = {train_idx: bd.train_dataset, test_idx: bd.test_dataset}
    print(f'Dataset classes: {np.unique(bd.labels_train)}\n')

    samplers = {}
    samplers[train_idx] = RandomSampler(datasets[train_idx])
    samplers[test_idx] = SequentialSampler(datasets[test_idx])

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                  batch_size=args.batch_size if x == train_idx else 1028,
                                                  sampler=samplers[x],
                                                  num_workers=20)
                   for x in [train_idx, test_idx]}
    dataset_sizes = {x: len(datasets[x]) for x in [train_idx, test_idx]}
    class_names = datasets[train_idx].classes.copy()
    num_classes = 2

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize model
    model = MLP(88 if include_categorical else 76, num_classes)

    # Layer freezing
    print(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience)

    # Could make this a command line argument
    eval_batch_freq = len(dataloaders[train_idx]) // 10
    print(f'Evaluation will be performed every {eval_batch_freq} batches.\n')

    out_path = os.path.join('./out/', name)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    with open(os.path.join(out_path, 'config.txt'), 'w') as file:
        file.write('Config for run: %s\n' % name)
        file.write('NUM_EPOCHS: %d\n' % args.n_epochs)
        file.write('LR: %e\n' % args.lr)
        file.write('BATCH_SIZE: %d\n' % args.batch_size)

    train(model, criterion, optimizer,
                           lr_scheduler, args.lr_patience, dataloaders, device, eval_batch_freq, out_path, train_idx, test_idx,
                           n_epochs=args.n_epochs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=ARCHITECTURES, help='The model architecture')
    parser.add_argument('--exp', type=str, required=True, choices=['binary-cic-usb'], help='The experimental setup for transfer learning')
    parser.add_argument('--categorical', default=True, help='Option to include or not include categorical features in the model')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Number of samples per training batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate during training')
    parser.add_argument('--lr-patience', type=int, default=3, help='Determines patience tolerance for reducing learning rate when learning stagnates. Higher means waiting longer before learning rate is reduced')

    args = parser.parse_args()

    if args.arch == 'mlp':
        train_mlp(args)

if __name__ == '__main__':
    main()