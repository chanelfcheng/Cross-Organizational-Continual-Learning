import argparse
import os
import sys

import torch
import torch
from torch.utils.data import RandomSampler

from architectures.mlp import MLP
from datasets import get_pytorch_dataset
from datasets import CIC_2018, USB_2021, CIC_CLASSES, USB_CLASSES, CIC_PATH, USB_PATH
from datasets.train_test_dataset import TrainTestDataset
from datasets.transfer_dataset import TransferDataset
from utils.train_eval import eval

def eval_mlp(args):
    """
    Setup the data objects to evaluate the specified MLP model
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
        dataset = TrainTestDataset(train_set, train_classes, train_path, test_set, test_classes, test_path, include_categorical)
        _, eval_dataset = dataset.get_pytorch_dataset(arch=args.arch)
    if args.exp == 'train-test-usb-cic':
        train_set = USB_2021
        train_classes = USB_CLASSES
        train_path = USB_PATH
        test_set = CIC_2018
        test_classes = CIC_CLASSES
        test_path = CIC_PATH
        dataset = TrainTestDataset(train_set, train_classes, train_path, test_set, test_classes, test_path, include_categorical)
        _, eval_dataset = dataset.get_pytorch_dataset(arch=args.arch)

    if args.exp == 'train-cic':
        a_set = CIC_2018
        a_path = CIC_PATH
        b_set = ''
        b_path = ''
        transfer_learn = 'none'
        source_classes = -1
        dataset = TransferDataset(a_set, a_path, b_set, b_path, include_categorical)
        _, eval_dataset = dataset.get_pytorch_dataset_a(arch=args.arch)
    if args.exp == 'train-usb':
        a_set = USB_2021
        a_path = USB_PATH
        b_set = ''
        b_path = ''
        transfer_learn = 'none'
        source_classes = -1
        dataset = TransferDataset(a_set, a_path, b_set, b_path, include_categorical)
        _, eval_dataset = dataset.get_pytorch_dataset_a(arch=args.arch)

    if args.exp == 'transfer-cic-usb':
        a_set = CIC_2018
        a_path = CIC_PATH
        b_set = USB_2021
        b_path = USB_PATH
        transfer_learn = 'freeze-feature'
        source_classes = 5
        dataset = TransferDataset(a_set, a_path, b_set, b_path, include_categorical)
        _, eval_dataset = dataset.get_pytorch_dataset_b(arch=args.arch)
    if args.exp == 'transfer-usb-cic':
        a_set = USB_2021
        a_path = USB_PATH
        b_set = CIC_2018
        b_path = CIC_PATH
        transfer_learn = 'freeze-feature'
        source_classes = 5
        dataset = TransferDataset(a_set, a_path, b_set, b_path, include_categorical)
        _, eval_dataset = dataset.get_pytorch_dataset_b(arch=args.arch)

    # Get Dataloader
    sampler = RandomSampler(eval_dataset)  # RandomSample for more balance for t-SNE

    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, sampler=sampler,
                                             num_workers=20)
    class_names = eval_dataset.classes.copy()
    num_classes = len(class_names)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize Model
    model = MLP(88 if include_categorical else 76, num_classes, embeddings=args.tsne)

    model.load_state_dict(torch.load(args.pretrained_path))

    model = model.to(device)

    out_path = os.path.abspath(os.path.join('./out/', name))
    eval(model, dataloader, device, out_path, tsne=args.tsne, tsne_percent=args.tsne_percent)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, help='The model architecture')
    parser.add_argument('--exp', type=str, required=True, choices=['train-cic', 'train-usb', 'train-test-cic-usb', 'train-test-usb-cic', 'transfer-cic-usb', 'transfer-usb-cic'], help='The experimental setup for transfer learning')
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
