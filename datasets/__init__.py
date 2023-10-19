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

# Get number of samples for each class in a dataset
def get_support(dataset):
    support = {}
    for label in np.array(dataset.tensors[1]):
        if label not in support:
            support[label] = 1
        else:
            support[label] += 1

    return support