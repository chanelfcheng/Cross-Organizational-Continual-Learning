import argparse
import copy
import os
import sys
import time
import csv

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from timm.scheduler import CosineLRScheduler
from torch import nn, optim
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architectures import ARCHITECTURES
from architectures.mlp import MLP
from eval_model import evaluate
from datasets import CIC_2018, CIC_CLASSES, USB_2021, CIC_PATH, USB_CLASSES, USB_PATH
from datasets.train_test_dataset import TrainTestDataset

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
    dataset_train, dataset_test = ttd.get_pytorch_dataset(arch=args.arch)
    datasets = {train: dataset_train, test: dataset_test}

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

    n_iter_per_epoch = len(dataloaders[train])
    num_steps = int(args.n_epochs * n_iter_per_epoch)
    warmup_steps = int(2 * n_iter_per_epoch)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=1e-6,
        warmup_lr_init=args.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    # Could make this a command line argument
    eval_batch_freq = len(dataloaders[train]) // 5

    out_dir = os.path.join('./out/', name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, 'config.txt'), 'w') as file:
        file.write('Config for run: %s\n' % name)
        file.write('NUM_EPOCHS: %d\n' % args.n_epochs)
        file.write('WARMUP_EPOCHS: %d\n' % 2)
        file.write('LR: %e\n' % args.lr)
        file.write('MIN_LR: %e\n' % 1e-6)
        file.write('WARMUP_LR: %e\n' % args.warmup_lr)
        file.write('BATCH_SIZE: %d\n' % args.batch_size)

    model_args = (88 if include_categorical else 76, num_classes)
    
    model = train_model(model, model_args, criterion, optimizer,
                           lr_scheduler, dataloaders, device, eval_batch_freq, out_dir, train, test,
                           n_epochs=args.n_epochs)
    return model

def train_model(model, model_args, criterion, optimizer, scheduler, dataloaders, device, eval_batch_freq, out_dir, train, test,
                n_epochs=25):
    """
    Helper function to perform the model training
    :param model: The MLP model to train
    :param criterion: The loss function
    :param optimizer: The optimizer object
    :param scheduler: The learning rate scheduler object
    :param dataloaders: Dictionary containing the training and testing dataset
    :param device: String for the device to perform training on
    :param eval_batch_freq: Number of iterations to perform between evaluation of model.
    :param out_dir: The output directory to save
    :param train: String denoting train key in dataloaders
    :param test: String denoting test key in dataloaders
    :param n_epochs: The number of epochs to train over
    :return: The trained model
    """
    # Model setup
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tensorboard_logs'))

    since = time.time()

    best_model_wts = model.state_dict()
    best_f1 = 0.0
    best_acc = 0.0
    eval_num = 1

    validation_accuracies = []

    # Training and testing phases
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [train, test]:
            if phase == train:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            start_test = True

            running_loss = 0.0

            # Iterate over data.
            iterator = tqdm(dataloaders[phase], file=sys.stdout)
            for idx, (inputs, labels) in enumerate(iterator):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == train):
                    outputs = model(inputs.float())
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == train:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if start_test:
                    all_preds = preds.float().cpu()
                    all_labels = labels.float()
                    start_test = False
                else:
                    all_preds = torch.cat((all_preds, preds.float().cpu()), 0)
                    all_labels = torch.cat((all_labels, labels.float()), 0)

                if phase == train:
                    num_steps = len(dataloaders[train])
                    scheduler.step_update(epoch * num_steps + idx)

                if phase == train and eval_batch_freq > 0:
                    if (idx + 1) % eval_batch_freq == 0:
                        # Evaluate the model every set number of batches
                        print('Evaluating model...')
                        model_f1, model_acc = evaluate(model, dataloaders[test], device, out_path=out_dir)
                        print('Deep copying model weights...')
                        validation_accuracies.append(model_acc)
                        if model_f1 > best_f1:
                            best_f1 = model_f1
                            best_model_wts = model.state_dict()
                        if model_acc > best_acc:
                            best_acc = model_acc
                        eval_num += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            all_labels = all_labels.detach().cpu().numpy()
            all_preds = all_preds.detach().cpu().numpy()
            top1_acc = accuracy_score(all_labels, all_preds)
            ave_f1_score = f1_score(all_labels, all_preds,
                                    average='macro')

            if phase == test:
                validation_accuracies.append(top1_acc)

                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning Rate', scalar_value=lr, global_step=epoch)
                writer.add_scalar('Training Loss', scalar_value=epoch_loss, global_step=epoch)
                writer.add_scalar('Validation Top-1 Acc', scalar_value=top1_acc, global_step=epoch)
                writer.add_scalar('Validation F1 Score', scalar_value=ave_f1_score, global_step=epoch)

            # Print out epoch results
            print('{} Loss: {:.4f} Top-1 Acc: {:.4f} F1 Score: {:.4f}'.format(phase, epoch_loss, top1_acc, ave_f1_score))

            if phase == test:
                if ave_f1_score > best_f1:
                    best_f1 = ave_f1_score
                if top1_acc > best_acc:
                    best_acc = top1_acc

        # save the model
        torch.save(model.state_dict(), os.path.join(out_dir, 'model_epoch_%d.pt' % epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))
    print('Best Accuracy: {:4f}'.format(best_acc))

    cr_dict = classification_report(all_labels, all_preds, target_names=dataloaders[test].dataset.classes, digits=4, output_dict=True)
    pd.DataFrame(cr_dict).transpose().to_csv(os.path.join(out_dir, 'report.csv'))
    
    with open(os.path.join(out_dir, 'report.csv'), 'a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['training time (min)'] + [str(time_elapsed/60)])

    # load best model weights
    model_copy = type(model)(*model_args)
    model_copy.load_state_dict(best_model_wts)
    return model_copy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=ARCHITECTURES, help='The model architecture')
    parser.add_argument('--exp', type=str, required=True, choices=['train-test-cic-usb', 'train-test-usb-cic'], help='The experimental setup for transfer learning')
    parser.add_argument('--categorical', default=True, help='Option to include or not include categorical features in the model')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Number of samples per training batch')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate during training')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, help='Learning rate during warmup')

    args = parser.parse_args()

    if args.arch == 'mlp':
        train_mlp(args)

if __name__ == '__main__':
    main()