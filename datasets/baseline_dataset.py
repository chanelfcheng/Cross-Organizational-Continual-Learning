import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import RobustScaler, LabelEncoder

from utils.preprocessing import process_features, remove_invalid, resample_data
from datasets import PKL_PATH

class BaselineDataset:
    """
    Dataset for train test setting used to evaluate model training and testing
    on two datasets w/o transfer or continual learning
    """
    def __init__(self, train_set, train_classes, train_path, test_set, test_classes, test_path, include_categorical=True):
        # Load in train and test sets
        self.features_train, self.labels_train = load_data(train_set + '-train-test', train_path, train_classes, test_classes, include_categorical, training=True)
        self.features_test, self.labels_test = load_data(test_set + '-train-test', test_path, train_classes, test_classes, include_categorical, training=False)

    def get_pytorch_dataset(self, arch='mlp'):
        # Fit scaler to train features and scale the train and test features
        scale = RobustScaler(quantile_range=(5,95)).fit(self.features_train)
        features_train = scale.transform(self.features_train)
        features_test = scale.transform(self.features_test)

        # Create pytorch tensors containing features only
        features_train = torch.tensor(features_train)
        features_test = torch.tensor(features_test)

        # Reshape input features for CNN
        if arch == 'cnn':
            features_train = features_train.reshape(len(features_train), features_train.shape[1], 1)
            features_test = features_test.reshape(len(features_test), features_test.shape[1], 1)
            features_train.shape, features_test.shape

        # Label encoding
        le = LabelEncoder()
        le.fit(self.labels_train)
        le_train = le.transform(self.labels_train)
        le_test = le.transform(self.labels_test)
        label_mapping = dict( zip( le.classes_, range( 0, len(le.classes_) ) ) )

        # Create pytorch tensors containing labels only
        labels_train = torch.tensor(le_train)
        labels_test = torch.tensor(le_test)
        classes = list(label_mapping.keys())

        # Create pytorch datasets with labels
        dataset_train = TensorDataset(features_train, labels_train)
        dataset_test = TensorDataset(features_test, labels_test)

        # Define dataset classes
        dataset_train.classes = classes
        dataset_test.classes = classes

        return dataset_train, dataset_test

def load_data(dset, data_path, train_classes, test_classes, include_categorical=True, training=True):
    """
    Loads in dataset from a folder containing all the data files. Processes
    features, replaces invalid values, and concatenates all data files into a
    single dataset. Splits dataset into train (.80) and test (.20) sets
    :param dset: name of the dataset
    :param data_path: path to the folder containing the data files
    :param include_categorical: option to include categorical features
    :param training: whether the dataset will be used for training or not
    :return: the training features, training labels, test features, and test labels
    """
    # Define variables to store all features, labels, and invalid count after concatenation
    all_features = np.array([])
    all_labels = []
    all_invalid = 0

    # Check if pre-processed pickle file exists
    if os.path.exists(os.path.join(PKL_PATH, f'{dset}.pkl')): 
        with open(os.path.join(PKL_PATH, f'{dset}.pkl'), 'rb') as file:
            all_features, all_labels = pickle.load(file)  # Load data from pickle file
    else:
        for file in list(glob.glob(os.path.join(f'{data_path}', '*.csv'))):
            print('\nLoading', file, '...')
            reader = pd.read_csv(file, dtype=str, chunksize=10**6, skipinitialspace=True)  # Read in data from csv file

            for df in reader:
                
                # Randomly sample 80% of the train set or 20% of the test set
                if training:
                    df = df.sample(frac=0.8)
                else:
                    df = df.sample(frac=0.2)

                # Process the features and labels
                features, labels = process_features(dset, df, include_categorical)

                # Only keep samples with classes found in both train and test sets
                features = features.loc[ labels.isin( list( set( train_classes ) & set( test_classes ) ) ) ]
                labels = labels.loc[ labels.isin( list( set( train_classes ) & set( test_classes ) ) ) ]

                # Convert dataframe to numpy array for processing
                data_np = np.array(features.to_numpy(), dtype=float)
                labels_lst = labels.tolist()

                data_np, labels_lst, num_invalid = remove_invalid(data_np, labels_lst)  # Clean data of invalid values

                # Combine all data, labels, and number of invalid values
                if all_features.size == 0:
                    all_features = data_np  # If no data yet, set all data to current data
                else:
                    all_features = np.concatenate((all_features, data_np))  # Else, concatenate data
                all_labels += labels_lst
                all_invalid += num_invalid

        # Print total number of invalid values dropped, total number of data
        # values, and total percentage of invalid data
        print('\nTotal Number of invalid values: %d' % all_invalid)
        print('Total Data values: %d' % len(all_labels))
        print('Invalid data: %.2f%%' % (all_invalid / float(all_features.size) * 100))

        # Save histogram of cleaned data
        axs = pd.DataFrame(all_features, columns=features.columns.values.tolist()).hist(figsize=(30,30))
        plt.tight_layout()
        plt.savefig(os.path.join('./figures/', f'hist_{dset}.png'))

        # Resample training data only
        if training:
            print('\nResampling training data...')
            all_features, all_labels = resample_data(dset, all_features, all_labels)

        # Save to pickle files
        with open(os.path.join(PKL_PATH, f'{dset}.pkl'), 'wb') as file:
            pickle.dump((all_features, all_labels), file)
        
    return all_features, all_labels