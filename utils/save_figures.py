import os
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

def save_class_hist(samples_dict: dict, name: str):
    """
    Saves the histogram for the specified distribution.  Useful for comparing class distribution during preprocessing
    :param samples_dict: The dictionary containing class name as the key and the number of samples as the value
    :param name: Unique name for savefile name.
    :return: None
    """
    classes = samples_dict.keys()
    samples = []
    for class_name in classes:
        samples.append(samples_dict[class_name])

    plt.clf()
    plt.bar(classes, samples)
    plt.title('Class Distribution')
    plt.ylabel('Num Samples')
    plt.xlabel('Class Name')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.rc('font', size=24)
    plt.savefig(os.path.join('./figures/', '%s.png' % name))  # TODO: Update to use specified output directory
    plt.clf()