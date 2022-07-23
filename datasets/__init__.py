import os

from matplotlib import projections

# Datasets
CIC_2018 = 'cic-2018'
USB_2021 = 'usb-2021'

# Classes
CLASSES = ['Benign', 'DoS-Hulk', 'DoS-Slowloris', 'DoS-SlowHttpTest', 'DoS-TCPFlood', 'DoS-GoldenEye']

# Data paths
# TODO: Change this to reflect your local path to the data files
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data/'
CIC_PATH = data_path + 'CIC-IDS2018/DoS'
USB_PATH = data_path + 'USB-IDS2021'

# Pickle paths
PKL_PATH = os.path.abspath(os.path.join(os.pardir, os.getcwd())) + '/pickle/'
