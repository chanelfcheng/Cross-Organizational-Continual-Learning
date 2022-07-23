import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from utils.data_preprocessing import process_features, remove_invalid, resample_data

class TransferDataset():
    """
    Dataset for transfer learning setting used to evaluate feature freezing from
    one dataset to another
    """
    def __init__(self, dset, data_path, pkl_path, include_categorical):
        self.a_features_train, \
            self.a_features_test, \
                self.a_labels_train, \
                    self.a_labels_test = load_data(dset + '-transfer', data_path, pkl_path, include_categorical)

    def get_dataset(self, model='mlp'):
        # Normalize train and test data
        scale = RobustScaler(quantile_range=(5,95)).fit(self.a_features_train)
        features_train = scale.transform(self.a_features_train)
        features_test = scale.transform(self.a_features_test)

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

def load_data(dset, data_path, pkl_path, include_categorical):
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
            features_train, features_test, labels_train, labels_test = pickle.load(file)  # Load data from pickle file
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

        # Perform train/test split of 80-20
        features_train, features_test, labels_train, labels_test = train_test_split(all_features, all_labels, test_size=0.2)

        # Resample training data to reduce class imbalance
        features_train, labels_train = resample_data(dset, features_train, labels_train)
        
        # Save to pickle file
        with open(pkl_path, 'wb') as file:
            pickle.dump((features_train, features_test, labels_train, labels_test), file)
        
    return features_train, features_test, labels_train, labels_test