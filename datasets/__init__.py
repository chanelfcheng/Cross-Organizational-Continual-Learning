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

# Classes
CIC_CLASSES = ['Benign', 'DoS-Hulk', 'DoS-Slowloris', 'DoS-SlowHttpTest', 'DoS-GoldenEye']
USB_CLASSES = ['Benign', 'DoS-Hulk', 'DoS-Slowloris', 'DoS-SlowHttpTest', 'DoS-TCPFlood']

# Data paths
# TODO: Change this to reflect your local path to the data files
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data/'))
CIC_PATH = os.path.join(data_path, 'CIC-IDS2018/DoS')
USB_PATH = os.path.join(data_path, 'USB-IDS2021')

# Pickle paths
PKL_PATH = os.path.abspath(os.path.join(os.pardir, os.getcwd(), 'pickle/'))

# Setting agnostic method for getting already pre-processed datasets saved as pickle files
def get_pytorch_dataset(dset, model='mlp'):
    
    # Check if pre-processed pickle file exists
    if not os.path.exists(PKL_PATH + dset + '.pkl'):
        print('Pickle file not found.')
        features_train, features_test, labels_train, labels_test = np.array([]), np.array([]), np.array([]), np.array([])
    else:
        with open(PKL_PATH + dset + '.pkl', 'rb') as file:
            features_train, features_test, labels_train, labels_test = pickle.load(file)  # Load data from pickle file

    # Normalize train and test data
    scale = RobustScaler(quantile_range=(5,95)).fit(features_train)
    features_train = scale.transform(features_train)
    features_test = scale.transform(features_test)

    # Create pytorch datasets for data only
    features_train = torch.tensor(features_train)
    features_test = torch.tensor(features_test)

    # Reshape input features for CNN
    if model == 'cnn':
        features_train = features_train.reshape(len(features_train), features_train.shape[1], 1)
        features_test = features_test.reshape(len(features_test), features_test.shape[1], 1)
        features_train.shape, features_test.shape

    # Label encoding
    label_encoding = {}
    value = 0
    for label in labels_test:
        if label not in label_encoding:
            label_encoding[label] = value
            value += 1
    
    labels_idx_train = []
    for i in range(len(labels_train)):
        label = labels_train[i]
        value = label_encoding[label]
        labels_idx_train.append(value)

    labels_idx_test = []
    for i in range(len(labels_test)):
        label = labels_test[i]
        value = label_encoding[label]
        labels_idx_test.append(value)

    labels_train = torch.tensor(labels_idx_train)
    labels_test = torch.tensor(labels_idx_test)
    classes = list(label_encoding.keys())

    # Create pytorch datasets with labels
    dataset_train = TensorDataset(features_train, labels_train)
    dataset_test = TensorDataset(features_test, labels_test)

    # Define attack classes
    dataset_train.classes = classes
    dataset_test.classes = classes

    return dataset_train, dataset_test