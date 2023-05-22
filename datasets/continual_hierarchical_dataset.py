import os
import glob
import pickle
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from utils.preprocessing import process_features, remove_invalid_v2, resample_data
from utils import create_if_not_exists

PKL_PATH = './pickle/'
create_if_not_exists(PKL_PATH)

class ContinualHierarchicalDataset:
    """
    Dataset for continual learning setting used to evaluate learning from a continuous stream of data.
    """
    SETTING = 'General Continual Learning'
    NAME = 'IDS'

    def __init__(self, args):
        # Set initial variables
        self.dataset_names = args.dataset_names.split(',')
        self.dataset_paths = args.dataset_paths.split(',')
        self.dataset_classes = args.dataset_classes.split(',')
        self.rename_binary = args.rename_binary.split(',')
        self.rename_super = args.rename_super.split(',')
        self.rename_sub = args.rename_sub.split(',')

        self.binary = list(dict.fromkeys(self.rename_binary))

        self.super = list(dict.fromkeys(self.rename_super))
        self.num_super = len(self.super)-1 # All unique labels except benign

        self.sub = list(dict.fromkeys(self.rename_sub))
        self.num_sub = []
        for superlabel in self.super:
            self.num_sub += [len([label for label in self.sub if superlabel in label])]
        self.args = args

        self.train_over, self.test_over = False, False
        self.train_loaders, self.test_loaders = [], []
        self.remaining_training_items = []

        self.train_classes = [0, 1]
        self.completed_rounds, self.train_iteration, self.test_class, self.test_iteration = 0, 0, 0, 0

        # Get label mapping
        self.label_mapping = {}
        for i in range(self.num_classes):
            self.label_mapping[self.classes[i]] = i
        
        # Load the data
        self.features_train, self.features_test, self.labels_train, self.labels_test = self.__load_data(self.dataset_names[0] + '_' + self.args.exp_name, self.dataset_paths[0], self.args.categorical)
        if len(self.dataset_names) > 1:
            for i in range(1, len(self.dataset_names)):
                features_train, features_test, labels_train, labels_test = self.__load_data(self.dataset_names[i] + '_' + self.args.exp_name, self.dataset_paths[i], self.args.categorical)
                self.features_train = np.concatenate([self.features_train, features_train])
                self.features_test = np.concatenate([self.features_test, features_test])
                self.labels_train = np.concatenate([self.labels_train, labels_train]).tolist()
                self.labels_test = np.concatenate([self.labels_test, labels_test]).tolist()

        # Resample training data
        print('\nResampling training data...')
        self.features_train, self.labels_train = resample_data(self.args.exp_name, self.features_train, self.labels_train)

        # Get train and test datasets
        self.train_dataset, self.test_dataset = self.get_pytorch_datasets(self.args.arch) 

        # Initialize the data loaders
        self.init_data_loaders()

        # Set active data loaders
        self.active_train_loaders = [
            self.train_loaders[self.train_classes[0]][0], 
            self.train_loaders[self.train_classes[1]].pop()]

        self.active_remaining_training_items = [
            self.remaining_training_items[self.train_classes[0]][0],
            self.remaining_training_items[self.train_classes[1]].pop()]
        # print('Continual dataset ready.')

    def init_data_loaders(self):
        """
        Initializes the data loaders.
        """
        # print('Initializing data loaders...')
        # Fill the train loaders
        for k in range(self.args.num_rounds):
        # for j in range(self.num_classes):
            # for k in range(self.args.num_rounds): # num_rounds is how many
            # rounds to train in one epoch of dataset
            for j in range(self.num_classes):
                self.train_loaders.append([])
                self.remaining_training_items.append([])
                mask = np.isin(np.array(self.train_dataset.tensors[1]), [j])
                samples_per_iteration = 10000
                masked_dataset = TensorDataset(
                    self.train_dataset.tensors[0][mask][k * samples_per_iteration:(k+1) * samples_per_iteration], 
                    self.train_dataset.tensors[1][mask][k * samples_per_iteration:(k+1) * samples_per_iteration])
                # print(masked_dataset.tensors[1][:10])
                # print(masked_dataset.tensors[1].shape)
                if len(masked_dataset) != 0:
                    self.train_loaders[-1].append(DataLoader(
                        masked_dataset, batch_size=1, shuffle=True))
                    self.remaining_training_items[-1].append(
                        masked_dataset.tensors[0].shape[0])
        
        # Fill the test loaders
        for j in range(self.num_classes):
            mask = np.isin(np.array(self.test_dataset.tensors[1]), [j])
            masked_dataset = TensorDataset(
                    self.test_dataset.tensors[0][mask], 
                    self.test_dataset.tensors[1][mask])
            self.test_loaders.append(DataLoader(masked_dataset,
                            batch_size=self.args.batch_size, shuffle=True))

    def train_next_class(self):
        """
        Changes the current pair of training classes by shifting the second class only while leaving the first class the same. If all classes have been visited, recycle through again until all rounds have been completed.
        """
        self.train_classes[1] += 1
        if self.train_classes[1] % self.num_classes == 0:
            self.train_classes[1] += 1
            self.train_classes[0] += self.num_classes
            self.completed_rounds += 1
            if self.completed_rounds == self.args.num_rounds:
                self.train_over = True
                
        print('\nCurrent malicious class:', list(self.label_mapping.keys())[self.train_classes[1]%self.num_classes])

        # if self.train_classes[1] == 1:
        #     self.completed_rounds += 1
        #     if self.completed_rounds == self.args.num_rounds:
        #         self.train_over = True
        
        if not self.train_over:
            # print(len(self.train_loaders[self.train_classes[0]]))
            # quit()
            self.active_train_loaders = [
                self.train_loaders[self.train_classes[0]][self.train_iteration],
                self.train_loaders[self.train_classes[1]].pop()]

            self.active_remaining_training_items = [
                self.remaining_training_items[self.train_classes[0]][self.train_iteration],
                self.remaining_training_items[self.train_classes[1]].pop()]

    def get_train_data(self):
        assert not self.train_over

        k = 0

        batch_size_0 = min(int(round(self.active_remaining_training_items[0] /
                                     (self.active_remaining_training_items[0] +
                                      self.active_remaining_training_items[1]) *
                                     self.args.batch_size)),
                           self.active_remaining_training_items[0])
        batch_size_1 = min(self.args.batch_size - batch_size_0,
                           self.active_remaining_training_items[1])

        x_train, y_train = [], []
        for i in range(batch_size_0):
            x_i, y_i = next(iter(self.active_train_loaders[0]))
            x_train.append(x_i)
            y_train.append(y_i)
        for j in range(batch_size_1):
            x_j, y_j = next(iter(self.active_train_loaders[1]))
            x_train.append(x_j)
            y_train.append(y_j)

        x_train, y_train = torch.cat(x_train), torch.cat(y_train)

        self.active_remaining_training_items[0] -= batch_size_0
        self.active_remaining_training_items[1] -= batch_size_1

        # print(self.active_remaining_training_items[0], self.active_remaining_training_items[1])
        if self.active_remaining_training_items[0] <= 0 or self.active_remaining_training_items[1] <= 0:
            self.train_next_class()
        
        return x_train, y_train

    def get_test_data(self):
        assert not self.test_over

        x_test, y_test = next(iter(self.test_loaders[self.test_class]))
        residual_items = len(self.test_loaders[self.test_class].dataset) - \
                        self.test_iteration * self.args.batch_size - len(x_test)
        self.test_iteration += 1

        if residual_items <= 0:
            if residual_items < 0:
                x_test = x_test[:residual_items]
                y_test = y_test[:residual_items]
            
            self.test_iteration = 0
            self.test_class += 1

            if self.test_class == self.num_classes:
                self.test_over = True

        return x_test, y_test

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

    def __map_labels(self, df):
        """
        Maps label names to be consistent across datasets.
        :param df: The dataframe for which the labels will be renamed
        """
        df = df.loc[df['Label'].str.contains('|'.join(self.dataset_classes), case=False)].copy()

        for i in range(len(self.dataset_classes)):
            if df['Label'].str.contains(self.dataset_classes[i], case=False):
                df['Label0'] = self.rename_binary[i]
                df['Label1'] = self.rename_super[i]
                df['Label2'] = self.rename_sub[i]
        
        return df

    # TODO: fix this to work with hierarchical labels
    def __load_data(self, dset, data_path, include_categorical=True):
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
                    # Map the labels for consistency
                    df = self.__map_labels(df)

                    # Process the features and labels
                    features, labels = process_features(dset, df.sample(frac=0.1), include_categorical)

                    # Get hierarchical labels
                    binary_labels = df['Label0']
                    super_labels = df['Label1']
                    sub_labels = df['Label2']

                    # Convert dataframe to numpy array for processing
                    data_np = np.array(features.to_numpy(), dtype=np.float32)
                    binary_lst = binary_labels.to_list()
                    super_lst = super_labels.to_list()
                    sub_lst = sub_labels.to_list()

                    data_np, remove_idx = remove_invalid_v2(data_np)  # Clean data of invalid values
                    binary_lst = np.delete(binary_lst, remove_idx).tolist()
                    super_lst = np.delete(super_lst, remove_idx).tolist()
                    sub_lst = np.delete(sub_lst, remove_idx).tolist()

                    # Combine all data, labels, and number of invalid values
                    if all_features.size == 0:
                        all_features = data_np  # If no data yet, set all data to current data
                    else:
                        all_features = np.concatenate((all_features, data_np))  # Else, concatenate data
                    all_binary += binary_lst
                    all_super += super_lst
                    all_sub += sub_lst
                    all_invalid += len(remove_idx)

            # Print total number of invalid values dropped, total number of data
            # values, and total percentage of invalid data
            print('\nTotal Number of invalid values: %d' % all_invalid)
            print('Total Data values: %d' % len(all_binary))
            print('Invalid data: %.2f%%' % (all_invalid / float(all_features.size) * 100))

            # Save histogram of cleaned data
            axs = pd.DataFrame(all_features, columns=features.columns.values.tolist()).hist(figsize=(30,30))
            plt.tight_layout()
            plt.savefig(os.path.join('./out/', f'hist_{dset}.png'))

            # Create new dataframe by concatenating all_features with
            # all_binary, all_super, and all_sub
            hierarchical_df = pd.DataFrame(all_features, columns=features.columns.values.tolist())
            hierarchical_df['Label0'] = all_binary
            hierarchical_df['Label1'] = all_super
            hierarchical_df['Label2'] = all_sub

            # Perform train/test split of 80-20
            train_set, test_set = train_test_split(hierarchical_df, test_size=0.2)
            
            # Convert hiearchical_df to numpy by separating into features and
            # labels
            features_train = train_set.drop(['Label0', 'Label1', 'Label2'], axis=1)
            features_train = np.array(features_train.to_numpy(), dtype=np.float32)
            binary_train = train_set['Label0'].values.tolist()
            super_train = train_set['Label1'].values.tolist()
            sub_train = train_set['Label2'].values.tolist()

            features_test = test_set.drop(['Label0', 'Label1', 'Label2'], axis=1)
            features_test = np.array(features_test.to_numpy(), dtype=np.float32)
            binary_test = test_set['Label0'].values.tolist()
            super_test = test_set['Label1'].values.tolist()
            sub_test = test_set['Label2'].values.tolist()

            # # Resample training data
            # print('\nResampling training data...')
            # features_train, labels_train = resample_data(dset, features_train, labels_train)
            
            # Save to pickle file
            with open(os.path.join(PKL_PATH, f'{dset}.pkl'), 'wb') as file:
                pickle.dump((features_train, features_test, binary_train, binary_test, super_train, super_test, sub_train, sub_test), file)
            
        return features_train, features_test, binary_train, binary_test, super_train, super_test, sub_train, sub_test