import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.stats.mstats import winsorize
from torch.utils.data import TensorDataset
from utils.preprocessing import process_features, remove_invalid, resample_data

# Datasets
CIC_2018 = 'cic-2018'
USB_2021 = 'usb-2021'

# Classes (benign is always first)
CIC_CLASSES = ['Benign', 'DoS-Hulk', 'DoS-Slowloris', 'DoS-SlowHttpTest', 'DoS-GoldenEye']
USB_CLASSES = ['Benign', 'DoS-Hulk', 'DoS-Slowloris', 'DoS-SlowHttpTest', 'DoS-TCPFlood']

# Data paths
# TODO: Change this to reflect your local path to the data files
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data/'))
CIC_PATH = os.path.join(data_path, 'CIC-IDS2018/DoS')
USB_PATH = os.path.join(data_path, 'USB-IDS2021')

# Pickle paths
PKL_PATH = os.path.abspath(os.path.join(os.pardir, os.getcwd(), 'pickle/'))

# Get number of samples for each class in a dataset
def get_support(dataset):
    support = {}
    for label in np.array(dataset.tensors[1]):
        if label not in support:
            support[label] = 1
        else:
            support[label] += 1

    return support