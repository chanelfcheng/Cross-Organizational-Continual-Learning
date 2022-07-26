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
from datasets.train_test_dataset import TrainTestDataset
from utils.model import train_model

def train_mlp(args):
    """
    Sets up PyTorch objects and trains the MLP model.
    :param args: The command line arguments
    :return: None
    """
    name = args.arch + '-' + args.exp
    include_categorical = args.categorical

    if args.exp == 'train-test-cic-usb':
        train_set = CIC_2018
        train_classes = CIC_CLASSES
        train_path = CIC_PATH
        test_set = USB_2021
        test_classes = USB_CLASSES
        test_path = USB_PATH
    if args.exp == 'train-test-usb-cic':
        train_set = USB_2021
        train_classes = USB_CLASSES
        train_path = USB_PATH
        test_set = CIC_2018
        test_classes = CIC_CLASSES
        test_path = CIC_PATH
    
    ttd = TrainTestDataset(train_set, train_classes, train_path, test_set, test_classes, test_path, include_categorical)
    train = 'train'
    test = 'test'

    # Load dataset
    print('\nLoading dataset...')
    dataset_train, dataset_test = ttd.get_pytorch_dataset(arch=args.arch)
    datasets = {train: dataset_train, test: dataset_test}
    print(f'Dataset classes: {np.unique(ttd.labels_train)}\n')

    samplers = {}
    samplers[train] = RandomSampler(datasets[train])
    samplers[test] = SequentialSampler(datasets[test])

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                  batch_size=args.batch_size if x == train else 1028,
                                                  sampler=samplers[x],
                                                  num_workers=20)
                   for x in [train, test]}
    dataset_sizes = {x: len(datasets[x]) for x in [train, test]}
    class_names = datasets[train].classes.copy()
    num_classes = len(class_names)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize model
    model = MLP(88 if include_categorical else 76, num_classes)
    print(model)

    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)

    # n_iter_per_epoch = len(dataloaders[train])
    # num_steps = int(args.n_epochs * n_iter_per_epoch)
    # warmup_steps = int(2 * n_iter_per_epoch)
    # lr_scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=num_steps,
    #     lr_min=1e-6,
    #     warmup_lr_init=args.warmup_lr,
    #     warmup_t=warmup_steps,
    #     cycle_limit=1,
    #     t_in_epochs=False,
    # )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience)

    # Could make this a command line argument
    eval_batch_freq = len(dataloaders[train]) // 5
    print(f'Evaluation will be performed every {eval_batch_freq} batches.\n')

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

    trained_model = train_model(model, criterion, optimizer,
                           lr_scheduler, args.lr_patience, dataloaders, device, eval_batch_freq, out_dir, train, test,
                           n_epochs=args.n_epochs)
    
    return trained_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=ARCHITECTURES, help='The model architecture')
    parser.add_argument('--exp', type=str, required=True, choices=['train-test-cic-usb', 'train-test-usb-cic'], help='The experimental setup for transfer learning')
    parser.add_argument('--categorical', default=True, help='Option to include or not include categorical features in the model')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Number of samples per training batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate during training')
    parser.add_argument('--lr-patience', type=int, default=3, help='Determines patience tolerance for reducing learning rate when learning stagnates. Higher means waiting longer before learning rate is reduced')
    parser.add_argument('--pretrained-path', type=str, default='', help='Path to the pretrained model to transfer weights from')

    args = parser.parse_args()

    if args.arch == 'mlp':
        train_mlp(args)

if __name__ == '__main__':
    main()