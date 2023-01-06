import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder

from utils.preprocessing import process_features, remove_invalid, resample_data
from datasets import PKL_PATH

class BinaryDataset:
    """
    Dataset for transfer learning setting used to evaluate feature freezing from
    one dataset to another
    """
    def __init__(self, dataset_names, classes, dataset_paths, args):
        # Set initial variables
        self.num_classes = len(classes)
        self.args = args

        # Get label mapping
        self.label_mapping = {}
        for i in range(self.num_classes):
            if i == 0:
                self.label_mapping[classes[i]] = 0
            else:
                self.label_mapping[classes[i]] = 1
        self.classes = ['Benign', 'Malicious']
        
        # Load the data
        self.features_train, self.features_test, self.labels_train, self.labels_test = load_data(dataset_names[0] + '-binary', dataset_paths[0], self.args.categorical)
        if len(dataset_names) > 1:
            for i in range(1, len(dataset_names)):
                features_train, features_test, labels_train, labels_test = load_data(dataset_names[i] + '-binary', dataset_paths[i], self.args.categorical)
                self.features_train = np.concatenate([self.features_train, features_train])
                self.features_test = np.concatenate([self.features_test, features_test])
                self.labels_train = np.concatenate([self.labels_train, labels_train]).tolist()
                self.labels_test = np.concatenate([self.labels_test, labels_test]).tolist()

        # Resample training data
        print('\nResampling training data...')
        self.features_train, self.labels_train = resample_data('continual', self.features_train, self.labels_train)

        # Get train and test datasets
        self.train_dataset, self.test_dataset = self.get_pytorch_datasets(self.args.arch)

    def get_pytorch_datasets(self, arch='mlp'):
        # print('Getting pytorch datasets...')
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
        le_train = []
        for label in self.labels_train:
            le_train.append(self.label_mapping[label])
        
        le_test = []
        for label in self.labels_test:
            le_test.append(self.label_mapping[label])

        # Create pytorch tensors containing labels only
        labels_train = torch.tensor(le_train)
        labels_test = torch.tensor(le_test)

        # Create pytorch datasets with labels
        train_dataset = TensorDataset(features_train, labels_train)
        test_dataset = TensorDataset(features_test, labels_test)

        # Define dataset classes
        train_dataset.classes = self.classes
        test_dataset.classes = self.classes

        return train_dataset, test_dataset

def load_data(dset, data_path, include_categorical=True):
    """
    Loads in dataset from a folder containing all the data files. Processes
    features, replaces invalid values, and concatenates all data files into a
    single dataset. Splits dataset into train (.80) and test (.20) sets
    :param dset: name of the dataset
    :param data_path: path to the folder containing the data files
    :param include_categorical: option to include categorical features
    :param resample: option to resample the data to reduce class imbalance
    :return: the training features, training labels, test features, and test labels
    """
    # Define variables to store all features, labels, and invalid count after concatenation
    all_features = np.array([])
    all_labels = []
    all_invalid = 0

    # Check if pre-processed pickle file exists
    if os.path.exists(os.path.join(PKL_PATH, f'{dset}.pkl')): 
        with open(os.path.join(PKL_PATH, f'{dset}.pkl'), 'rb') as file:
            features_train, features_test, labels_train, labels_test = pickle.load(file)  # Load data from pickle file
    else:
        for file in list(glob.glob(os.path.join(f'{data_path}', '*.csv'))):
            print('\nLoading', file, '...')
            reader = pd.read_csv(file, dtype=str, chunksize=10**6, skipinitialspace=True)  # Read in data from csv file

            for df in reader:
                # Process the features and labels
                features, labels = process_features(dset, df.sample(frac=0.1), include_categorical)

                # Convert dataframe to numpy array for processing
                data_np = np.array(features.to_numpy(), dtype=np.float32)
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
        plt.savefig(os.path.join('./out/', f'hist_{dset}.png'))

        # Perform train/test split -- Benign and Hulk/Benign and Other Attacks
        df = pd.DataFrame( np.hstack( ( all_features, np.array(all_labels).reshape(-1,1) ) ), columns=features.columns.values.tolist() + ['Label'] )
        df_train = df.loc[df['Label'].str.contains('benign', case=False)].sample(frac=0.5)
        df_train = pd.concat([df_train, df.loc[df['Label'].str.contains('hulk', case=False)]])
        df_test = df.drop(df_train.index)
        features_train = df_train.iloc[:, :-1].to_numpy()
        features_test = df_test.iloc[:, :-1].to_numpy()
        labels_train = df_train.iloc[:, -1].tolist()
        labels_test = df_test.iloc[:, -1].tolist()

        # # Resample training data
        # print('\nResampling training data...')
        # features_train, labels_train = resample_data(dset, features_train, labels_train)
        
        # Save to pickle file
        with open(os.path.join(PKL_PATH, f'{dset}.pkl'), 'wb') as file:
            pickle.dump((features_train, features_test, labels_train, labels_test), file)
        
    return features_train, features_test, labels_train, labels_test