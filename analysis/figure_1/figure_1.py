import sys
from os import makedirs
from os.path import join, dirname, realpath, exists

from config_path import PROSTATE_LOG_PATH, PLOTS_PATH

current_dir = dirname(dirname(realpath(__file__)))
sys.path.insert(0, dirname(current_dir))
import itertools
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import pandas as pd
from matplotlib import pyplot as plt

import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
import seaborn as sns


def plot_roc(ax, y_test, y_pred_score, save_dir, color, label=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, label=label + ' (%0.2f)' % roc_auc, linewidth=1, color=color)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.1)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontproperties)
    ax.set_ylabel('True Positive Rate', fontproperties)


def plot_prc(ax, y_test, y_pred_score, save_dir, color, label=''):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_score)
    roc_auc = average_precision_score(y_test, y_pred_score)
    ax.plot(recall, precision, label=label + ' (%0.2f)' % roc_auc, linewidth=1, color=color)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall', fontproperties, labelpad=1)
    ax.set_ylabel('Precision', fontproperties, labelpad=1)
    # We change the fontsize of minor ticks label
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=5)


def plot_prc_all(ax):
    colors = sns.color_palette(None, n)
    import collections

    # sort based on area under prc
    sorted_dict = {}
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        average_prc = average_precision_score(y_test, y_pred_score)
        sorted_dict[k] = average_prc

    sorted_dict = sorted(sorted_dict.items(), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_dict)
    print 'sorted_dict', sorted_dict

    for i, k in enumerate(sorted_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        print i, k
        plot_prc(ax, y_test, y_pred_score, None, label=k, color=colors[i])

    f_scores = np.linspace(0.2, 0.8, num=4)
    for i, f_score in enumerate(f_scores):
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.3, linewidth=0.5)
        if i == 0:
            tex = 'F1={0:.1f}'
            xy = (y[45] - 0.07, 1.02)
        else:
            tex = '{0:.1f}'
            xy = (y[45] - 0.03, 1.02)
        ax.annotate(tex.format(f_score), fontsize=fontsize, xy=xy, alpha=0.5)
    ax.legend(loc="lower left", fontsize=fontsize, framealpha=0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=0, width=0, grid_alpha=0.5)

    for tick in ax.get_yaxis().get_major_ticks():
        tick.set_pad(.7)
    for tick in ax.get_xaxis().get_major_ticks():
        tick.set_pad(.7)

    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)

    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=.1)


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
    fig = plt.gcf()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.tick_params(axis=u'both', which=u'both', length=0)
    cb.outline.set_visible(False)
    tick_marks = np.arange(len(classes))
    if labels is None:
        fmt = '{:.2f}%' if normalize else '{:d}'
    else:
        fmt = '{}: {:.2f}%' if normalize else '{}: {:d}'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = fmt.format(labels[i, j], cm[i, j])
        ax.text(j, i, text,
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    ax.set_ylabel('True label', fontproperties)
    ax.set_xlabel('Predicted label', fontproperties)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontproperties)
    ax.set_yticks([t - 0.25 for t in tick_marks])
    ax.set_yticklabels(classes, fontproperties, rotation=90)


def plot_confusion_matrix_all(ax):
    base_dir = join(PROSTATE_LOG_PATH, 'pnet')
    models_base_dir = join(base_dir, 'onsplit_average_reg_10_tanh_large_testing')
    filename = join(models_base_dir, 'P-net_ALL_testing.csv')
    df = pd.read_csv(filename, index_col=0)

    df.head()
    y_t = df.y
    y_pred_test = df.pred
    cnf_matrix = confusion_matrix(y_t, y_pred_test)
    print cnf_matrix

    cm = np.array(cnf_matrix)
    classes = ['Primary', 'Metastatic']
    labels = np.array([['TN', 'FP'], ['FN ', 'TP']])

    plot_confusion_matrix(ax, cm, classes,
                          labels,
                          normalize=True,
                          cmap=plt.cm.Reds)
    ax.tick_params(axis=u'both', which=u'both', length=0)


fontsize = 5
fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}

models = ['Linear Support Vector Machine ', 'RBF Support Vector Machine ', 'L2 Logistic Regression', 'Random Forest',
          'Adaptive Boosting', 'Decision Tree']
n = len(models) + 1

all_models_dict = {}

base_dir = PROSTATE_LOG_PATH
models_base_dir = join(base_dir, 'compare/onsplit_ML_test')
for i, m in enumerate(models):
    df = pd.read_csv(join(models_base_dir, m + '_data_0_testing.csv'), sep=',', index_col=0, header=[0, 1])
    all_models_dict[m] = df

pnet_base_dir = join(base_dir, 'pnet/onsplit_average_reg_10_tanh_large_testing')
df_pnet = pd.read_csv(join(pnet_base_dir, 'P-net_ALL_testing.csv'), sep=',', index_col=0, header=[0, 1])
all_models_dict['P-NET'] = df_pnet


def run():
    saving_dir = join(PLOTS_PATH, 'figure1')

    if not exists(saving_dir):
        makedirs(saving_dir)

    fig = plt.figure(constrained_layout=False, figsize=(3.5, 3.5))
    spec2 = gridspec.GridSpec(ncols=5, nrows=2, figure=fig, height_ratios=[5, 4])
    ax1 = fig.add_subplot(spec2[0, :])
    ax2 = fig.add_subplot(spec2[1, 0:3])
    ax3 = fig.add_subplot(spec2[1, 3:])

    fig.subplots_adjust(left=0.07, bottom=0.05, right=0.96, top=0.99, wspace=1.5, hspace=0.1)

    plot_prc_all(ax2)
    plot_confusion_matrix_all(ax3)

    ax1.set_xticks([])
    ax1.set_yticks([])
    # for minor ticks
    ax1.set_xticks([], minor=True)
    ax1.set_yticks([], minor=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    filename = join(saving_dir, 'figure1')
    plt.savefig(filename, dpi=400)
    plt.savefig(filename + '.pdf', transparent=True)
    # sns.set_style(None)


if __name__ == "__main__":
    run()
