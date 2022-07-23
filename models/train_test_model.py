import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from timm.scheduler import CosineLRScheduler
from torch import nn, optim
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mlp import MLP
from eval_mlp import eval_model
from datasets import CIC_2018, USB_2021, CIC_PATH, USB_PATH
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
        train_path = CIC_PATH
        test_set = USB_2021
        test_path = USB_PATH
        transfer_learn = 'freeze-feature'
    if args.exp == 'train-test-usb-cic':
        train_set = USB_2021
        train_path = USB_PATH
        test_set = CIC_2018
        test_path = CIC_PATH
        transfer_learn = 'freeze-feature'
    
    ttd = TrainTestDataset(train_set, train_path, test_set, test_path, include_categorical)
    train = 'train'
    test = 'test'

    # Load dataset
    dataset_train, dataset_test = ttd.get_pytorch_dataset(model=args.arch)
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
    num_steps = int(args.num_epochs * n_iter_per_epoch)
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
        file.write('NUM_EPOCHS: %d\n' % args.num_epochs)
        file.write('WARMUP_EPOCHS: %d\n' % 2)
        file.write('LR: %e\n' % args.lr)
        file.write('MIN_LR: %e\n' % 1e-6)
        file.write('WARMUP_LR: %e\n' % args.warmup_lr)
        file.write('BATCH_SIZE: %d\n' % args.batch_size)

    model_ft = train_model(name, model, criterion, optimizer,
                           lr_scheduler, dataloaders, device, eval_batch_freq, out_dir, train, test,
                           num_epochs=args.num_epochs)

def train_model(name, model, criterion, optimizer, scheduler, dataloaders, device, eval_batch_freq, out_dir, train, test,
                num_epochs=25):
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
    :param num_epochs: The number of epochs to train over
    :return: The trained model
    """
    # Model setup
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tensorboard_logs'))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_acc = 0.0
    eval_num = 1

    validation_accuracies = []

    # Training and testing phases
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                        model_f1, model_acc = eval_model(name, model, dataloaders[test], device, out_path=out_dir)
                        validation_accuracies.append(model_acc)
                        if model_f1 > best_f1:
                            best_f1 = model_f1
                            best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), os.path.join(out_dir, 'model_eval_%d.pt' % eval_num))
                        if model_acc > best_acc:
                            best_acc = model_acc
                        eval_num += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            all_labels = all_labels.detach().cpu().numpy()
            all_preds = all_preds.detach().cpu().numpy()
            top1_acc = accuracy_score(all_labels, all_preds)
            val_f1_score = f1_score(all_labels, all_preds,
                                    average='macro')

            if phase == test:
                validation_accuracies.append(top1_acc)

                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning Rate', scalar_value=lr, global_step=epoch)
                writer.add_scalar('Training Loss', scalar_value=epoch_loss, global_step=epoch)
                writer.add_scalar('Validation Top-1 Acc', scalar_value=top1_acc, global_step=epoch)
                writer.add_scalar('Validation F1 Score', scalar_value=val_f1_score, global_step=epoch)

            print('{} Loss: {:.4f} Top-1 Acc: {:.4f} F1 Score: {:.4f}'.format(
                phase, epoch_loss, top1_acc, val_f1_score))

            # deep copy the model
            if phase == test:
                if val_f1_score > best_f1:
                    best_f1 = val_f1_score
                if top1_acc > best_acc:
                    best_acc = top1_acc

                if len(validation_accuracies) < 10:
                    end = len(validation_accuracies)
                else:
                    end = 10
                top_10 = np.flip(np.argsort(np.asarray(validation_accuracies)))[:end]
                for i in range(top_10.size):
                    epoch_num = top_10[i]
                    epoch_accuracy = validation_accuracies[epoch_num]
                    print('Rank %d: Eval Num: %d, Acc1: %.3f%%' % (i + 1, epoch_num + 1, epoch_accuracy))

                torch.save(model.state_dict(), os.path.join(out_dir, 'model_eval_%d.pt' % eval_num))
                eval_num += 1

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))
    print('Best Accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True,
    choices=['mlp', 'cnn'], help='The model architecture')
    parser.add_argument('--exp', type=str, required=True,
    choices=['train-test-cic-usb', 'train-test-usb-cic'], help='The experimental setup for transfer learning')
    parser.add_argument('--categorical', action='store_true', help='Option to include or not include categorical features in the model')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Number of samples per training batch')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate during training')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, help='Learning rate during warmup')

    args = parser.parse_args()

    train_mlp(args)

if __name__ == '__main__':
    main()