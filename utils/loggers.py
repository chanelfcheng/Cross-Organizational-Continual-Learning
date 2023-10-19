import os
import csv
import sys
from typing import Dict, Any
from wsgiref.util import setup_testing_defaults
import numpy as np
import pandas as pd

from utils import create_if_not_exists

def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    mean_acc_class_il, mean_acc_task_il = mean_acc
    print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
            ' \t [Task-IL]: {} %\n'.format(task_number, round(
        mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)

class CsvLogger:
    def __init__(self, name):
        self.name = name
    
    def log(self, report_dict):
        report_df = pd.DataFrame(report_dict).transpose()
        self.report = report_df
    
    def write(self, args):
        columns = list(args.keys())

        create_if_not_exists(os.path.join('./out/', self.name))

        path = os.path.join('./out/', self.name, 'report.csv')
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            writer.writeheader()
            writer.writerow(args)

        with open(path, 'a') as tmp:
            writer = csv.writer(tmp)
            writer.writerow([])    
    
        self.report.to_csv(path, mode='a', index=False, header=False)

        with open(path, 'a') as tmp:
            writer = csv.writer(tmp)
            writer.writerow([])
            writer.writerow([])

class TensorboardLogger:
    def __init__(self):
        from torch.utils.tensorboard import SummaryWriter
