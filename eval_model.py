import argparse
import os
import sys

import torch
import torch
from torch.utils.data import RandomSampler

from architectures.mlp import MLP
from datasets import get_pytorch_dataset
from utils.model import eval_model

def eval_mlp(args):
    """
    Setup the data objects to evaluate the specified MLP model
    :param args: The command line arguments
    :return: None
    """
    batch_size = args.batch_size
    include_categorical = args.categorical

    # Load dataset
    _, eval_dataset = get_pytorch_dataset(dset=args.dset, model='mlp')
    sampler = RandomSampler(eval_dataset)  # RandomSample for more balance for t-SNE

    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, sampler=sampler,
                                             num_workers=20)
    class_names = eval_dataset.classes.copy()
    num_classes = len(class_names)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize Model
    model = MLP(88 if include_categorical else 76, num_classes, embeddings=args.tsne)

    model.load_state_dict(torch.load(args.pretrained_path))

    model = model.to(device)

    out_path = os.path.abspath(args.pretrained_path)
    eval_model(model, dataloader, device, out_path, tsne=args.tsne, tsne_percent=args.tsne_percent)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, help='The model architecture')
    parser.add_argument('--pretrained-path', type=str, required=True, help='Path to the pretrained weights')
    parser.add_argument('--categorical', default=True, help='Option to include or not include categorical features in the model')
    parser.add_argument('--batch-size', type=int, default=1028, help='The batch size to use for evaluation')
    parser.add_argument('--tsne', action='store_true', help='If set generates TSNE plots using subset of data.'
                                                            'Other metrics are not valid')
    parser.add_argument('--tsne-percent', default=0.01, help='To speed up TSNE, only run on a small portion of the '
                                                             'dataset')
    args = parser.parse_args()

    path = args.pretrained_path
    if not os.path.exists(path):
        print('Pretrained path is invalid.', file=sys.stderr)
        exit(1)

    if args.arch == 'mlp':
        eval_mlp(args)
    
    print('Done')


if __name__ == '__main__':
    main()
