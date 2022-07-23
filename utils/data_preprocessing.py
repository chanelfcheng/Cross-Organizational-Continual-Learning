import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

from utils.save_figures import save_feature_table, save_class_hist

# Encoder for protocol feature
ohe1 = OneHotEncoder(sparse=False)
ohe1.fit(np.array(['0', '6', '17']).reshape(-1,1))  # Most common protocols

# Encoder for destination port feature
ohe2 = OneHotEncoder(sparse=False)
ohe2.fit(np.array(['dns', 'http', 'https', 'wbt', 'smb', 'ftp', 'ssh',  'llmnr', 'other']).reshape(-1,1))  # Most common port services

def process_features(dset, df, include_categorical):
    """
    Processes features of a dataset and separates them from their class labels. Updates label names, removes redundant
    features, and encodes categorical features consistently across multiple datasets.
    :param dset: The name of the dataset
    :param df: The dataframe containing the dataset
    :param include_categorical: Option to include or exclude categorical features
    :return: The features and their corresponding labels 
    """
    print('Processing features...')
    rename_labels(df)

    attack = df.loc[df['Label'].str.contains('hulk|slowloris|slowhttptest|tcpflood|goldeneye', case=False)].copy()  # Get attack types
    benign = df.loc[df['Label'].str.contains('benign', case=False)].copy()  # Get all benign traffic

    features = pd.concat([attack, benign]).drop(['Label'], axis=1)  # Concatenate attack/benign and separate label from features
    labels = pd.concat([attack, benign])['Label']  # Save labels by themselves

    # Remove unused columns if present
    if 'Timestamp' in features:
        features = features.drop('Timestamp', axis=1)
    if 'Flow ID' in features:
        features = features.drop('Flow ID', axis=1)
    if 'Src IP' in features:
        features = features.drop('Src IP', axis=1)
    if 'Src Port' in features:
        features = features.drop('Src Port', axis=1)
    if 'Dst IP' in features:
        features = features.drop('Dst IP', axis=1)

    # Reset dataframe indexing
    features.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    # Categorical feature processing if included
    if include_categorical:
        # Protocol one-hot encoding
        print('protocol one-hot encoding...')
        ohe_protocol = ohe1.transform(features['Protocol'].values.reshape(-1,1))
        ohe_protocol = pd.DataFrame(ohe_protocol, columns=ohe1.get_feature_names_out(['Protocol']))
        
        features = features.drop('Protocol', axis=1)
        features = features.join(ohe_protocol)

        # Destination port mapping
        print('destination port mapping...')
        map_ports(features, 'Dst Port')

        # Destination port one-hot encoding
        print('destination port one-hot encoding...')
        ohe_dport = ohe2.transform(features['Dst Port'].values.reshape(-1,1))
        ohe_dport = pd.DataFrame(ohe_dport, columns=ohe2.get_feature_names_out(['Port']))

        features = features.drop('Dst Port', axis=1)
        features = features.join(ohe_dport)
    else:
        features = features.drop(['Protocol', 'Dst Port'], axis=1)

    # Save first three rows of table containing categorical features as a figure
    save_feature_table('encoded_categorical_' + dset, features, 0, 3, -12, None)

    return features, labels

def rename_labels(df):
    """
    Renames label names to be consistent across datasets.
    :param df: The dataframe for which the labels will be renamed
    """
    df.loc[df['Label'].str.contains('benign', case=False), 'Label'] = 'Benign'
    df.loc[df['Label'].str.contains('hulk', case=False), 'Label'] = 'DoS-Hulk' 
    df.loc[df['Label'].str.contains('slowloris', case=False), 'Label'] = 'DoS-Slowloris'
    df.loc[df['Label'].str.contains('slowhttptest', case=False), 'Label'] = 'DoS-SlowHttpTest'
    df.loc[df['Label'].str.contains('tcpflood', case=False), 'Label'] = 'DoS-TCPFlood'
    df.loc[df['Label'].str.contains('goldeneye', case=False), 'Label'] = 'DoS-GoldenEye'

def map_ports(features, feature_name):
    """
    Maps port numbers to their corresponding port service. The most common port
    services are denoted by their name and the rest are denoted as 'other'.
    :param features: The column containing the port number feature
    :param feature_name: The name of the feature column
    :return: None
    """
    dns = features[feature_name] == '53'
    http = (features[feature_name] == '80') | (features[feature_name] == '8080')
    https = (features[feature_name] == '443') | (features[feature_name] == '8443')
    wbt = features[feature_name] == '3389'
    smb = (features[feature_name] == '445') | (features[feature_name] == '139') | (features[feature_name] == '137')
    ftp = (features[feature_name] == '20') | (features[feature_name] == '21')
    ssh = features[feature_name] == '22'
    llmnr = features[feature_name] == '5535'
    other = ~(http | https | ftp | ssh | dns | smb | wbt | llmnr)

    features.loc[dns, feature_name] = 'dns'
    features.loc[http, feature_name] = 'http'
    features.loc[https, feature_name] = 'https'
    features.loc[wbt, feature_name] = 'wbt'
    features.loc[smb, feature_name] = 'smb'
    features.loc[ftp, feature_name] = 'ftp'
    features.loc[ssh, feature_name] = 'ssh'
    features.loc[llmnr, feature_name] = 'llmnr'
    features.loc[other, feature_name] = 'other'

def remove_invalid(features_np, labels_lst):
    """
    Cleans a numpy array by removing samples with invalid feature values.
    :param features: The numpy array of features
    :param labels: List of the labels for each sample from the data array
    :return: The processed numpy array
    """
    num_invalid = 0
    remove_idx = []

    for flow_idx in tqdm(range(features_np.shape[0]), file=sys.stdout, desc='Cleaning data array...'):
        for feature_idx in range(features_np.shape[1]):
            data_val = features_np[flow_idx, feature_idx]
            if np.isnan(data_val) or np.isinf(data_val):
                remove_idx.append(flow_idx)
                num_invalid += 1

    features_np = np.delete(features_np, remove_idx, axis=0)
    labels_lst = np.delete(labels_lst, remove_idx, axis=0).tolist()

    print('Removed %d invalid values' % num_invalid)

    return features_np, labels_lst, num_invalid

def resample_data(dset, features, labels):
    """
    Resamples the data to reduce class imbalance. The largest class is randomly undersampled to 2x greater than the
    next largest class. The minority classes are randomly oversampled to 20% of
    the largest class, after undersampling.
    :param dset: The name of the dataset
    :param features: The columns containing the features
    :param labels: The column containing the labels
    :return: the resampled features and their corresponding labels
    """
    class_samples = {}
    orig_samples = len(labels)
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'orig_dist_' + dset)

    # Undersample largest class to 2x greater than next largest class
    largest_num = max(class_samples.values())
    largest_class = max(class_samples, key=class_samples.get)
    largest_min_num = 0
    for class_name in class_samples.keys():
        if class_name != largest_class and class_samples[class_name] > largest_min_num:
            largest_min_num = class_samples[class_name]
    target_largest = 2 * largest_min_num
    if target_largest < largest_num:
        print('Reducing %s data from %d to %d samples' % (largest_class, largest_num, target_largest))
        undersampler = RandomUnderSampler(sampling_strategy={largest_class: target_largest})
        features, labels = undersampler.fit_resample(features, labels)
    else:
        print('Not reducing any classes')

    print('Finished Undersampling')
    class_samples = {}
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'after_undersampling_' + dset)

    # # Drop extreme minority classes
    # min_class_count = 0.01 * class_samples['Benign']
    # print('Dropping classes with < %d samples' % min_class_count)

    # classes_to_drop = []
    # for class_name in class_samples.keys():
    #     if class_samples[class_name] < min_class_count:
    #         classes_to_drop.append(class_name)

    # data, labels = drop_classes(data, labels, classes_to_drop)

    # class_samples = {}
    # for label in labels:
    #     if label not in class_samples:
    #         class_samples[label] = 1
    #     else:
    #         class_samples[label] += 1
    # save_class_hist(class_samples, 'after_dropping_' + name)

    # Oversample minority classes up to 20% of largest class (after undersampling)
    largest_num = max(class_samples.values())
    largest_class = max(class_samples, key=class_samples.get)
    target_dict = {}
    target_num = round(largest_num * 0.20)
    print('Targeting %d samples for each minority class' % target_num)
    for label in class_samples.keys():
        if label == largest_class or class_samples[label] > target_num:
            target_dict[label] = class_samples[label]
        else:
            target_dict[label] = target_num

    oversampler = RandomOverSampler(sampling_strategy=target_dict)
    start = time.time()
    features, labels = oversampler.fit_resample(features, labels)
    print('Finished Oversampling')
    class_samples = {}
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'after_oversampling_' + dset)
    print('Total Data values: %d' % orig_samples)

    return features, labels

def drop_classes(features, labels, classes_to_drop):
    """
    Drops the classes specified in the classes_to_drop list from the data and labels structures
    :param features: The entire numpy dataset of features
    :param labels: The labels list for the data array
    :param classes_to_drop: A list of the classes that should be dropped from the dataset
    :return: The updated data and labels objects
    """
    drop_row = np.full(len(labels), False)
    for i in range(len(labels)):
        if labels[i] in classes_to_drop:
            drop_row[i] = True

    new_labels = []
    for i in range(len(labels)):
        if labels[i] not in classes_to_drop:
            new_labels.append(labels[i])
    labels = new_labels

    features = features[~drop_row, :]

    print('Done dropping.  Shape of data %s -- Size of labels %d' % (str(features.shape), len(labels)))
    return features, labels