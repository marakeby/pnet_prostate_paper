import sys
from os.path import dirname, realpath

current_dir = dirname(dirname(realpath(__file__)))
sys.path.insert(0, dirname(current_dir))

import pandas as pd
from matplotlib import pyplot as plt
from os.path import join
import numpy as np

import itertools
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

# set default params
from config_path import PROSTATE_LOG_PATH, PLOTS_PATH


def plot_confusion_matrix(ax, cm, classes, labels=None,
                          normalize=False,
                          # title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.set_title(title)
    # ax.colorbar()
    fig = plt.gcf()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=0.1)

    # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.outline.set_visible(False)
    tick_marks = np.arange(len(classes))

    if labels is None:
        fmt = '{:.2f}%' if normalize else '{:d}'
    else:
        fmt = '{}: {:.2f}%' if normalize else '{}: {:d}'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #         text= format(labels[i,j], cm[i, j], fmt)
        text = fmt.format(labels[i, j], cm[i, j])
        ax.text(j, i, text,
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=12)

    # ax.tight_layout()
    fontproperties = {'family': 'Arial', 'weight': 'bold', 'size': 14}

    ax.set_ylabel('True label', fontproperties)
    # ax.set_ylabel('True label', fontsize=12, fontweight = 'bold', fontproperties )
    ax.set_xlabel('Predicted label', fontproperties)
    # plt.gcf().subplots_adjust(bottom=0.25, left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    print tick_marks
    print classes

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    #
    # ax.set_yticks([t-0.25 for t in tick_marks])
    ax.set_yticks([t for t in tick_marks])
    ax.set_yticklabels(classes, rotation=0)

    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().subplots_adjust(left=0.25)

    # ax.set_xticks(tick_marks, classes)
    # ax.set_yticks(tick_marks, classes)

    # ax.set_xticks([])
    # ax.set_yticks([])
    # for minor ticks
    # ax.set_xticks([], minor=True)
    # ax.set_yticks([], minor=True)


#                     cmap = plt.cm.Greys)

def plot_confusion_matrix_all(ax):
    base_dir = join(PROSTATE_LOG_PATH, 'pnet')
    models_base_dir = join(base_dir, 'onsplit_average_reg_10_tanh_large_testing')
    filename = join(models_base_dir, 'P-net_ALL_testing.csv')
    df = pd.read_csv(filename, index_col=0)

    # df.pred = df.pred_scores > 0.5
    df.head()

    y_t = df.y

    y_pred_test = df.pred
    cnf_matrix = confusion_matrix(y_t, y_pred_test)
    print cnf_matrix

    cm = np.array(cnf_matrix)
    classes = ['Primary', 'Metastatic']
    labels = np.array([['TN', 'FP'], ['FN ', 'TP']])
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    plot_confusion_matrix(ax, cm, classes,
                          labels,
                          normalize=True,
                          # title='Confusion matrix',
                          # cmap=plt.cm.Blues)
                          cmap=plt.cm.Reds)


def run_matrix():
    saving_dir = join(PLOTS_PATH, 'figure1')
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5, 4), dpi=400)
    plot_confusion_matrix_all(ax)
    filename = join(saving_dir, 'confusion matrix.png')
    plt.savefig(filename, dpi=400)
    plt.close()


if __name__ == '__main__':
    run_matrix()
