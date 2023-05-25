import os
import argparse

import numpy as np
import torch.optim as optim
import torch

from architectures import ARCHITECTURES
from architectures.mlp import MLP
from architectures.vae import VAE
from datasets import get_class_weights
from datasets.continual_hierarchical_dataset import ContinualHierarchicalDataset
from utils.focal_loss import FocalLoss
from utils import create_if_not_exists
from models.continual_hierarchical_model import train, eval_check, \
    Er

"""
usage: continual.py [-h] 
--exp_name EXP_NAME
--dataset_names DATASET_NAMES
--dataset_paths DATASET_PATHS
--dataset_classes CLASSES
--rename_labels LABELS 
--arch {mlp}
"""

def run_continual_hierarchical(args):
    dataset_names = args.dataset_names.split(',')
    dataset_classes = args.dataset_classes.split(',')
    
    dataset = ContinualHierarchicalDataset(args)
    print('\nContinual dataset:', dataset_names)
    print('Classes:', dataset_classes)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize model for intrusion detection
    binary_arch = MLP(88 if args.categorical else 76, 2)
    super_arch = MLP(88 if args.categorical else 76, dataset.num_super)
    sub_archs = []
    for i in range(len(dataset.sub)):
        sub_archs += [MLP(88 if args.categorical else 76, dataset.num_sub[i])]
    archs = [binary_arch, super_arch, sub_archs]

    # TODO: fix weights to match number of classes for binary (2), super (2),
    # and sub (5, 2)

    # Get class weights
    binary_weights = get_class_weights(dataset, device, target_col=-3)
    super_weights = get_class_weights(dataset, device, target_col=-2)
    sub_weights = []
    for i in range(len(dataset.sub)):
        sub_weights += [get_class_weights(dataset, device, target_col=-1)]

    # Define loss criterions
    binary_criterion = FocalLoss(beta=binary_weights, gamma=args.gamma)
    super_criterion = FocalLoss(beta=super_weights, gamma=args.gamma)
    sub_criterions = []
    for i in range(len(dataset.sub)):
        sub_criterions += [FocalLoss(beta=sub_weights[i], gamma=args.gamma)]
    criterions = [binary_criterion, super_criterion, sub_criterions]
    
    # Define optimizers
    binary_optimizer = optim.RAdam(binary_arch.parameters(), lr=args.lr)
    super_optimizer = optim.RAdam(super_arch.parameters(), lr=args.lr)
    sub_optimizers = []
    for i in range(len(dataset.sub)):
        sub_optimizers += [optim.RAdam(sub_archs[i].parameters(), lr=args.lr)]
    optimizers = [binary_optimizer, super_optimizer, sub_optimizers]

    hierarchical_model = Er(archs, criterions, optimizers, args)

    out_path = os.path.join('./out/', args.exp_name)
    create_if_not_exists(out_path)

    counter = 0
    while os.path.exists(os.path.join(out_path, f'model_{counter}.pt')):
        counter += 1

    return hierarchical_model, dataset, out_path, counter

def main():
    parser = argparse.ArgumentParser()
    # Main configuration options
    parser.add_argument('--exp-name', type=str, default='test', help='Name of the experiment')
    parser.add_argument('--dataset-names', type=str, default='cic-2018,usb-2021', help='Names of the dataset to include in the experiment')
    parser.add_argument('--dataset-paths', type=str, default='../data/CIC-IDS2018,../data/USB-IDS2021', help='Paths of the dataset to include in the experiment')
    parser.add_argument('--dataset-classes', type=str, default='benign,hulk,slowloris,slowhttp,goldeneye,tcpflood', help='Classes from the dataset to include in the experiment')
    parser.add_argument('--rename-binary', type=str, default='Benign,DoS-Hulk,DoS-Slowloris,DoS-SlowHttpTest,DoS-GoldenEye,DoS-TCPFlood', help='Labels to which to rename the binary classes')
    parser.add_argument('--rename-super', type=str, default='Benign,DoS-Hulk,DoS-Slowloris,DoS-SlowHttpTest,DoS-GoldenEye,DoS-TCPFlood', help='Labels to which to rename the super classes')
    parser.add_argument('--rename-sub', type=str, default='Benign,DoS-Hulk,DoS-Slowloris,DoS-SlowHttpTest,DoS-GoldenEye,DoS-TCPFlood', help='Labels to which to rename the sub classes')
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
    parser.add_argument('--sampling-threshold', type=float, default=0.85, help='Probability threshold for sampling to the buffer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate during training')

    args = parser.parse_args()
    user_in = input("Train or evaluate the model? (train/eval) ")

    while user_in not in ['train', 'eval']:
        user_in = input("Train or evaluate the model? (train/eval) ")
    
    hierarchical_model, dataset, out_path, counter = run_continual_hierarchical(args)

    if user_in == 'train':
        train(hierarchical_model, dataset, out_path, counter, args)
    if user_in == 'eval':
        counter -= 1
        eval_check(hierarchical_model, dataset, out_path, counter)

if __name__ == '__main__':
    main()