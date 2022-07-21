import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from utils.data_preprocessing import process_features, remove_invalid, resample_data
from sklearn.preprocessing import RobustScaler

class TrainTestDataset():
    """
    Dataset for train test setting used to evaluate model training and testing
    w/o transfer or continual learning
    """
    def __init__(self, train_set, test_set, train_path, test_path, train_pkl_path, test_pkl_path, include_categorical):
        # Load in train and test sets
        self.features_train, self.labels_train = load_data(train_set + '-train', train_path, train_pkl_path, include_categorical, resample=True)
        self.features_test, self.labels_test = load_data(test_set + '-test', test_path, test_pkl_path, include_categorical, resample=False)

        # Remove classes uncommon between train and test sets
        _, train_idx, test_idx = np.intersect1d(self.labels_train, self.labels_test, return_indices=True)
        self.features_train = np.take(self.features_train, train_idx)
        self.labels_train = np.take(self.labels_train, train_idx)
        self.features_test = np.take(self.features_test, test_idx)
        self.labels_test = np.take(self.labels_test, test_idx)
        
        # Save to pickle files
        if not os.path.exists(train_pkl_path):
            with open(train_pkl_path, 'wb') as file:
                pickle.dump((self.features_train, self.labels_train), file)

        if not os.path.exists(test_pkl_path):
            with open(test_pkl_path, 'wb') as file:
                pickle.dump((self.features_test, self.labels_test), file)

    def get_dataset(self, model='mlp'):
        # Normalize train and test data
        scale = RobustScaler(quantile_range=(5,95)).fit(self.features_train)
        features_train = scale.transform(self.features_train)
        features_test = scale.transform(self.features_test)

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

        # Define classes
        dataset_train.classes = classes
        dataset_test.classes = classes

        return dataset_train, dataset_test

def load_data(dset, data_path, pkl_path, include_categorical=True, resample=True):
    """
    Loads in dataset from a folder containing all the data files. Processes
    features, replaces invalid values, and concatenates all data files into a
    single dataset. Splits dataset into train (.80) and test (.20) sets
    :param dset: name of the dataset
    :param data_path: path to the folder containing the data files
    :param pkl_path: path to pickle file for saving pre-processed data
    :param include_categorical: option to include or exclude categorical features
    :return: the training features, training labels, test features, and test labels
    """
    # Define variables to store all features, labels, and invalid count after concatenation
    all_features = None
    all_labels = []
    all_invalid = 0

    # Check if pre-processed pickle file exists
    if os.path.exists(pkl_path): 
        with open(pkl_path, 'rb') as file:
            all_features, all_labels = pickle.load(file)  # Load data from pickle file
    else:
        for file in list(glob.glob(f'{data_path}/*.csv')):
            print('Loading ', file, '...')
            reader = pd.read_csv(file, dtype=str, chunksize=10**6, skipinitialspace=True)  # Read in data from csv file

            for df in reader:
                # Process the features and labels
                features, labels = process_features(dset, df, include_categorical)

                # Convert dataframe to numpy array for processing
                data_np = np.array(features.to_numpy(), dtype=float)
                labels_lst = labels.tolist()

                data_np, labels_lst, num_invalid = remove_invalid(data_np, labels_lst)  # Clean data of invalid values

                # Combine all data, labels, and number of invalid values
                if all_features is None:
                    all_features = data_np  # If no data yet, set all data to current data
                else:
                    all_features = np.concatenate((all_features, data_np))  # Else, concatenate data
                all_labels += labels_lst
                all_invalid += num_invalid

        # Print total number of invalid values dropped, total number of data
        # values, and total percentage of invalid data
        print('Total Number of invalid values: %d' % all_invalid)
        print('Total Data values: %d' % len(all_labels))
        print('Invalid data: %.2f%%' % (all_invalid / float(all_features.size) * 100))

        # Save histogram of cleaned data
        axs = pd.DataFrame(all_features, columns=features.columns.values.tolist()).hist(figsize=(30,30))
        plt.tight_layout()
        plt.savefig('./figures/hist_' + dset + '.png')

        # Resample data
        if resample:
            all_features, all_labels = resample_data(dset, all_features, all_labels)
        
    return all_features, all_labels