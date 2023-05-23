import os
import time
from datetime import timedelta
from argparse import Namespace

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.nn import functional as F

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

from utils.buffer import Buffer, ModifiedBuffer
from utils.progress import progress_bar

class ContinualHierarchicalModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, architectures: nn.Module, criterion: nn.Module, optimizer: nn.Module,
                args: Namespace) -> None:
        super(ContinualHierarchicalModel, self).__init__()

        self.nets = architectures
        self.loss = criterion
        self.args = args
        self.opt = optimizer
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x: torch.Tensor, net_idx: int) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.nets[net_idx](x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :return: the value of the loss function
        """
        pass

class Er(ContinualHierarchicalModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, architectures, criterion, optimizer, args):
        super(Er, self).__init__(architectures, criterion, optimizer, args)
        self.buffer = ModifiedBuffer(self.args.buffer_size, self.args.sampling_threshold, self.device) if self.args.buffer_strategy == 'uncertainty' \
            else Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, net_idx, drift=None):
        self.buffer.add_data(self.nets, examples=inputs, labels=labels, drift=drift) if self.args.buffer_strategy == 'uncertainty' \
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

        outputs = self.nets[net_idx](buf_inputs)

        loss = self.loss(outputs, buf_labels)
        loss.backward()
        self.opt.step()

        return loss.item()
    

class Der(ContinualHierarchicalModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, architecture, criterion, optimizer, args):
        super(Der, self).__init__(architecture, criterion, optimizer, args)
        self.buffer = ModifiedBuffer(self.args.buffer_size, self.args.sampling_threshold, self.device) if self.args.buffer_strategy == 'uncertainty' \
            else Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels):

        self.opt.zero_grad()

        outputs = self.nets(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size)
            buf_outputs = self.nets(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(self.nets, examples=inputs, logits=outputs.data) if self.args.buffer_strategy == 'uncertainty' \
            else self.buffer.add_data(examples=inputs, labels=labels)

        return loss.item()

class Ddm():
    def __init__(self, autoencoder, optimizer):
        self.optimizer = optimizer
        self.autoencoder = autoencoder
    
    def observe(self, inputs):
        # Normalize the data to be between 0 and 1 (VAE requires this)
        mean = inputs.mean()
        std = inputs.std()
        inputs = (inputs - mean) / std
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())

        r_inputs, mu, logvar = self.autoencoder(inputs)
        loss = self.autoencoder.encoder_loss(r_inputs, inputs, mu, logvar)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        

def train(hierarchical_model, dataset, out_path, counter, args):
    init_log(out_path, counter, args)

    print('\nTraining phase')
    print('Current malicious class:',  list(dataset.label_mapping.keys())[dataset.train_classes[1]])
    hierarchical_model[0].net.to(hierarchical_model[0].device)
    hierarchical_model[1].net.to(hierarchical_model[1].device)
    for i in range(dataset.num_sub):
        hierarchical_model[2][i].net.to(hierarchical_model[2][i].device)

    hierarchical_model[0].net.train()
    hierarchical_model[1].net.train()
    for i in range(dataset.num_sub):
        hierarchical_model[2][i].net.train()

    i = 0
    start = time.time()
    while not dataset.train_over:
        if i == 0:
            max_progress = sum(dataset.active_remaining_training_items) // args.batch_size
        
        if not (i + 1) % (max_progress + 1):
            eval_check(hierarchical_model, dataset, out_path, counter)

            # for key in model.buffer.buffer_content:
            #     model.buffer.buffer_content[key] = (model.buffer.buffer_content[key] // (model.buffer.batch_count)) / args.batch_size


            # print("samples saved to buffer:", model.buffer.buffer_content)
            # # print("number of batches:", model.buffer.batch_count)
            
            # model.buffer.batch_count = 0

            # for key in model.buffer.buffer_content:
            #     model.buffer.buffer_content[key] = 0

            # model.buffer.sample_count = 0

            i = 0
        else:
            i += 1
        
        inputs, labels = dataset.get_train_data()   # TODO: fix this to get the hierarchical labels
        print("train:", inputs.shape, labels.shape)

        inputs, labels = inputs.to(hierarchical_model.device), labels.to(hierarchical_model.device)
        loss = hierarchical_model.observe(inputs.float(), labels, drift=None)

        progress_bar(i, max_progress, dataset.completed_rounds + 1, 'C', loss)
    
    time_elapsed = time.time() - start
    print(f'\nTraining complete in {timedelta(seconds=round(time_elapsed))}')

    with open(os.path.join(out_path, f'log_{counter}.txt'), 'a') as file:
        file.write(f'\nTraining complete in {timedelta(seconds=round(time_elapsed))}\n')

    torch.save(hierarchical_model[0].state_dict(), os.path.join(out_path, f'binary_model_{counter}.pt'))
    torch.save(hierarchical_model[1].state_dict(), os.path.join(out_path, f'super_model_{counter}.pt'))
    for i in range(len(hierarchical_model[2])):
        torch.save(hierarchical_model[2][i].state_dict(), os.path.join(out_path, f'sub_model_{i}_{counter}.pt'))

    eval_check(hierarchical_model, dataset, out_path, counter)

def eval_check(hierarchical_model, dataset, classes, out_path, counter, save_embedding=False, target_col=-1):
    print('\nEvaluation phase')
    hierarchical_model[0].net.to(hierarchical_model[0].device)
    hierarchical_model[1].net.to(hierarchical_model[1].device)
    for i in range(dataset.num_sub):
        hierarchical_model[2][i].net.to(hierarchical_model[2][i].device)

    hierarchical_model[0].net.eval()
    hierarchical_model[1].net.eval()
    for i in range(dataset.num_sub):
        hierarchical_model[2][i].net.eval()

    dataset.test_over = False
    dataset.test_class = 0
    start_test = True
    
    while not dataset.test_over:
        inputs, labels = dataset.get_test_data()    # TODO: fix this to get the hierarchical labels
        print("test:", inputs.shape, labels.shape)
        inputs, labels = inputs.to(hierarchical_model.device), labels.to(hierarchical_model.device)
        outputs = hierarchical_model.net(inputs.float(), return_embedding=False)

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
    all_embeddings = all_embeddings.detach().cpu().numpy()

    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print(report)

    log_results(report, out_path, counter)
    
    return all_labels, all_preds, all_embeddings

def init_log(out_path, counter, args):
    with open(os.path.join(out_path, f'log_{counter}.txt'), 'w') as file:
        file.write('Config for run: %s\n' % args.exp_name)
        file.write('CATEGORICAL: %s\n' % args.categorical)
        file.write('NUM_ROUNDS: %d\n' % args.num_rounds)
        file.write('NUM_EPOCHS: %d\n' % args.n_epochs)
        file.write('BATCH_SIZE: %d\n' % args.batch_size)
        file.write('MINIBATCH_SIZE: %d\n' % args.minibatch_size)
        file.write('BUFFER_SIZE: %d\n' % args.buffer_size)
        file.write('BUFFER_STRATEGY: %s, %f\n' % (args.buffer_strategy, args.sampling_threshold))
        if args.alpha > 0: file.write('ALPHA: %f\n' % args.alpha)
        file.write('GAMMA: %f\n' % args.gamma)
        file.write('LR: %e\n' % args.lr)

def log_results(report, out_path, counter):
    filename = f'log_{counter}.txt'
    with open(os.path.join(out_path, filename), 'a') as file:
        file.write(f'\n{report}\n')

def save_confusion_matrix(labels, preds, classes, out_path, counter):
    matrix = confusion_matrix(labels, preds, normalize='true')
    filename = f'matrix_{counter}.png'

    df_matrix = pd.DataFrame(matrix, index=classes, columns=classes)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # plt.figure(figsize=(15,10))
    plot_matrix = sn.heatmap(df_matrix, annot=True, ax=ax)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('True', fontsize=15)
    plot_matrix.figure.savefig(os.path.join(out_path, filename), bbox_inches='tight')