import os
import sys
import time
import math
import copy
from argparse import Namespace
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.manifold import TSNE


class BaselineModel(nn.Module):
    """
    Baseline learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, architecture: nn.Module, criterion: nn.Module, optimizer: nn.Module,
                args: Namespace) -> None:
        super(BaselineModel, self).__init__()

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


def train(model, criterion, optimizer, scheduler, patience, dataloaders, device, eval_batch_freq, out_path, train, test,
                n_epochs=25):
    """
    Helper function to perform the model training
    :param model: The MLP model to train
    :param criterion: The loss function
    :param optimizer: The optimizer object
    :param scheduler: The learning rate scheduler object
    :param dataloaders: Dictionary containing the training and testing dataset
    :param device: String for the device to perform training on
    :param out_dir: The output directory to save
    :param train: String denoting train key in dataloaders
    :param test: String denoting test key in dataloaders
    :param n_epochs: The number of epochs to train over
    :return: The trained model
    """
    # Model setup
    writer = SummaryWriter(log_dir=os.path.join(out_path, 'tensorboard_logs'))

    since = time.time()

    best_model_wts = copy.deepcopy(model.net.state_dict())
    best_f1 = 0.0
    best_acc = 0.0
    best_loss = 1_000_000.
    loss_counter = 0

    # Iterate over each epoch
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [train, test]:
            if phase == train:
                print('Training model...')
                model.net.train()  # Set model to training mode
            else:
                print('Evaluating model...')
                model.net.eval()  # Set model to evaluate mode
            start = True

            running_loss = 0.0

            # Iterate over each batch of data.
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

                if start:
                    all_preds = preds.float().cpu()
                    all_labels = labels.float()
                    start = False
                else:
                    all_preds = torch.cat((all_preds, preds.float().cpu()), 0)
                    all_labels = torch.cat((all_labels, labels.float()), 0)

                # statistics
                running_loss += loss.item() * inputs.size(0)

                if phase == train:
                    # # Update learning rate
                    # num_steps = len(dataloaders[train])
                    # scheduler.step_update(epoch * num_steps + idx)

                    # Evaluate the model every set number of batches
                    if (idx + 1) % eval_batch_freq == 0 and eval_batch_freq > 0:
                        eval_f1, eval_acc, report = eval(model, dataloaders[test], device, out_path=out_path)
                        
                        # Save results
                        with open(os.path.join(out_path, 'report1.txt'), 'a') as file:
                            file.write(f'\n{report}')
                        
                        # Update best f1 score
                        if eval_f1 > best_f1:
                            best_f1 = eval_f1

                            # Deep copy model weights with best f1 score
                            best_model_wts = copy.deepcopy(model.net.state_dict())

                            # Save best model
                            torch.save(model.net.state_dict(), os.path.join(out_path, 'model_epoch_%d.pt' % epoch))
                        
                        # Update best accuracy
                        if eval_acc > best_acc:
                            best_acc = eval_acc
                        
                        # Get current loss and update learning rate if needed
                        eval_loss = running_loss / len(dataloaders[test].dataset)
                        scheduler.step(eval_loss)

                        # Update best loss
                        if eval_loss < best_loss:
                            best_loss = eval_loss

                        # Early stopping
                        if eval_loss > best_loss:
                            loss_counter += 1                            
                            if loss_counter > patience * 5:

                                # Get time elapsed
                                time_elapsed = time.time() - since
                                print('Training complete in {:.0f}m {:.0f}s'.format(
                                    time_elapsed // 60, time_elapsed % 60))
                                print('Best val F1: {:4f}'.format(best_f1))
                                print('Best Accuracy: {:4f}'.format(best_acc))

                                # Save time elapsed
                                with open(os.path.join(out_path, 'report.txt'), 'a') as file:
                                    file.write('\n')
                                    file.write('training time (min): ' + str(time_elapsed / 60))

                                # Load best model
                                model.net.load_state_dict(best_model_wts)
                                return model
                        
            # Calculate metrics
            phase_loss = running_loss / len(dataloaders[phase].dataset)
            all_labels = all_labels.detach().cpu().numpy()
            all_preds = all_preds.detach().cpu().numpy()
            phase_acc = accuracy_score(all_labels, all_preds)
            phase_f1_score = f1_score(all_labels, all_preds,
                                    average='macro')
            
            # Print out results from this phase
            print('{} Loss: {:.4f} Top-1 Acc: {:.4f} F1 Score: {:.4f}'.format(phase, phase_loss, phase_acc, phase_f1_score))

            if phase == test:
                # Log data 
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning Rate', scalar_value=lr, global_step=epoch)
                writer.add_scalar('Training Loss', scalar_value=phase_loss, global_step=epoch)
                writer.add_scalar('Validation Acc', scalar_value=phase_acc, global_step=epoch)
                writer.add_scalar('Validation F1 Score', scalar_value=phase_f1_score, global_step=epoch)
            
                if phase_f1_score > best_f1:
                    best_f1 = phase_f1_score                  
                if phase_acc > best_acc:
                    best_acc = phase_acc

        print()
    
    # Get time elapsed
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))
    print('Best Accuracy: {:4f}'.format(best_acc))

    # Load best model weights
    model.net.load_state_dict(best_model_wts)
    return model

def eval(model, dataloader, device, out_path=None, tsne=False, tsne_percent=0.01):
    """
    Evaluate the given model
    :param model: The MLP model
    :param dataloader: Dataloader for the evaluation data
    :param device: string for the specified device to perform computation
    :param out_path: Path to the output dir to save information
    :param tsne: Boolean flag on whether to perform TSNE visualization
    :param tsne_percent: The percentage of evaluation data to plot for TSNE
    :return: The f1-score and accuracy
    """
    model.net.eval()  # Set model to evaluate mode
    start_test = True

    # Iterate over data.
    if tsne:
        max_iter = math.floor(len(dataloader) * tsne_percent)
    else:
        max_iter = len(dataloader) + 5
    
    iterator = tqdm(dataloader, file=sys.stdout)
    for idx, (inputs, labels) in enumerate(iterator):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if tsne:
            outputs, feat_embeddings = model(inputs.float())
        else:
            outputs = model(inputs.float())
        _, preds = torch.max(outputs, 1)

        # statistics
        if start_test:
            all_preds = preds.float().cpu()
            all_labels = labels.float()
            if tsne:
                embeddings = feat_embeddings.float().cpu().detach().numpy()
            start_test = False
        else:
            all_preds = torch.cat((all_preds, preds.float().cpu()), 0)
            all_labels = torch.cat((all_labels, labels.float()), 0)
            if tsne:
                embeddings = np.concatenate([embeddings, feat_embeddings.detach().cpu().numpy()], axis=0)

        if idx > max_iter:
            break

    all_labels = all_labels.detach().cpu().numpy()
    all_preds = all_preds.detach().cpu().numpy()
    top1_acc = accuracy_score(all_labels, all_preds)
    ave_f1_score = f1_score(all_labels, all_preds,
                            average='macro')

    if out_path is not None:
        plt.clf()
        plt.figure(figsize=(10,10))
        cf_matrix = confusion_matrix(all_labels, all_preds)
        cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
        acc = cf_matrix.diagonal() / cf_matrix.sum(axis=1) * 100
        disp = ConfusionMatrixDisplay.from_predictions(all_labels, all_preds,
                                                       display_labels=dataloader.dataset.classes, values_format='0.2f',
                                                       normalize='true', xticks_rotation='vertical')
        disp.plot(values_format='0.2f', xticks_rotation='vertical')
        plt.title('CF acc=%.2f%%' % top1_acc)
        plt.rc('font', size=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'cf.png'))
        plt.close('all')

    if tsne:
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(embeddings)

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 8))
        num_categories = len(dataloader.dataset.classes)
        colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange']
        for lab in range(num_categories):
            indices = all_labels == lab
            ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], label=dataloader.dataset.classes[lab], c=colors[lab],
                       alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.title('TSNE acc=%.2f%%' % acc.mean())
        plt.savefig(os.path.join(out_path, 'tsne.png'))
        plt.close('all')
    else:
        report = classification_report(all_labels, all_preds, target_names=dataloader.dataset.classes, digits=4)
        print('\n', report)
        return ave_f1_score, top1_acc, report