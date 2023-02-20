import os
import argparse

from collections import namedtuple
import numpy as np
import torch.optim as optim
import torch

from architectures import ARCHITECTURES
from architectures.mlp import MLP
from datasets import get_support
from datasets.continual_dataset import ContinualDataset
from utils.focal_loss import FocalLoss
from utils import create_if_not_exists
from models.continual_model import train_continual, Er, Der

"""
usage: continual.py [-h] 
--exp_name EXP_NAME
--dataset_names DATASET_NAMES
--dataset_paths DATASET_PATHS
--dataset_classes CLASSES
--rename_labels LABELS 
--arch {mlp}
"""

def run_continual(args):
    name = args.exp_name
    dataset_names = args.dataset_names.split(',')
    dataset_classes = args.dataset_classes.split(',')
    include_categorical = args.categorical
    
    dataset = ContinualDataset(args)
    print('\nContinual dataset:', dataset_names)
    print('Classes:', dataset_classes)

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
    # Main configuration options
    parser.add_argument('--exp-name', type=str, default='test', help='Name of the experiment')
    parser.add_argument('--dataset-names', type=str, default='cic-2018,usb-2021', help='Names of the dataset to include in the experiment')
    parser.add_argument('--dataset-paths', type=str, default='../data/CIC-IDS2018,../data/USB-IDS2021', help='Paths of the dataset to include in the experiment')
    parser.add_argument('--dataset-classes', type=str, default='benign,hulk,slowloris,slowhttp,goldeneye,tcpflood', help='Classes from the dataset to include in the experiment')
    parser.add_argument('--rename-labels', type=str, default='Benign,DoS-Hulk,DoS-Slowloris,DoS-SlowHttpTest,DoS-GoldenEye,DoS-TCPFlood', help='Labels to which to rename the classes')
    parser.add_argument('--arch', type=str, default='mlp', choices=ARCHITECTURES, help='The model architecture')

    # Experiment parameters
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
        run_continual(args)

if __name__ == '__main__':
    main()