import argparse
import math
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import RandomSampler
from tqdm import tqdm

import mlp
from load_data import load_pytorch_dataset, CIC_2018, USB_2021
from utils.save_figures import save_classification_report

def eval(pretrained_path, include_categorical, args):
    """
    Setup the data objects to evaluate the specified MLP model
    :param pretrained_path: Path to the pretrained model
    :param args: The command line arguments
    :return: None
    """
    batch_size = args.batch_size

    # Load dataset
    _, eval_dataset = load_pytorch_dataset(dset=args.dset, data_path=args.data_path, pkl_path=args.pkl_path, include_categorical=True, model='mlp')
    sampler = RandomSampler(eval_dataset)  # RandomSample for more balance for t-SNE

    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, sampler=sampler,
                                             num_workers=20)
    class_names = eval_dataset.classes.copy()
    num_classes = len(class_names)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize Model
    model = mlp.MLP(88 if include_categorical else 76, num_classes, embeddings=args.tsne)

    model.load_state_dict(torch.load(pretrained_path))

    model = model.to(device)

    out_path = os.path.join('out', args.name)
    eval_model(model, dataloader, device, out_path, tsne=args.tsne, tsne_percent=args.tsne_percent)


def eval_model(name, model, dataloader, device, out_path=None, tsne=False, tsne_percent=0.01):
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
    model.eval()  # Set model to evaluate mode
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
    val_f1_score = f1_score(all_labels, all_preds,
                            average='macro')

    if out_path is not None:
        plt.clf()
        cf_matrix = confusion_matrix(all_labels, all_preds)
        cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
        acc = cf_matrix.diagonal() / cf_matrix.sum(axis=1) * 100
        disp = ConfusionMatrixDisplay.from_predictions(all_labels, all_preds,
                                                       display_labels=dataloader.dataset.classes, values_format='0.2f',
                                                       normalize='true', xticks_rotation='vertical')
        disp.plot(values_format='0.2f', xticks_rotation='vertical')
        plt.title('CF acc=%.2f%%' % top1_acc)
        # plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'cf.png'))
        plt.clf()

    if tsne:
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(embeddings)

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 8))
        num_categories = len(dataloader.dataset.classes)
        for lab in range(num_categories):
            indices = all_labels == lab
            ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], label=dataloader.dataset.classes[lab],
                       alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.title('TSNE acc=%.2f%%' % acc.mean())
        plt.savefig(os.path.join(out_path, 'tsne.png'))
        plt.clf()

    print('Top-1 Acc: {:.4f} F1 Score: {:.4f}'.format(top1_acc, val_f1_score))
    log_str = classification_report(all_labels, all_preds, target_names=dataloader.dataset.classes, digits=4)
    print(log_str)
    return val_f1_score, top1_acc, log_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='debug', help='Unique name used for saving output files')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pretrained weights')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset files')
    parser.add_argument('--dset', required=True, choices=[CIC_2018, USB_2021], help='Specify which dataset to use for'
                                                                                    'evaluation')
    parser.add_argument('--pkl-path', type=str, help='Path to store pickle files.  Saves time by storing preprocessed '
                                                     'data')
    parser.add_argument('--batch-size', type=int, required=True, help='The batch size to use for evaluation')
    parser.add_argument('--tsne', action='store_true', help='If set generates TSNE plots using subset of data.'
                                                            'Other metrics are not valid')
    parser.add_argument('--tsne-percent', default=0.01, help='To speed up TSNE, only run on a small portion of the '
                                                             'dataset')
    args = parser.parse_args()

    path = args.model_path
    if not os.path.exists(path):
        print('Path is invalid', file=sys.stderr)
        exit(1)

    eval(path, True, args)
    print('Done')


if __name__ == '__main__':
    main()