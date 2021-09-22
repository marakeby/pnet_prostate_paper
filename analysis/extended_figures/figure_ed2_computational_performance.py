from matplotlib import gridspec
import collections
import itertools
from os import makedirs
from os.path import join, exists

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import metrics
from sklearn.metrics import confusion_matrix, average_precision_score

from config_path import PROSTATE_LOG_PATH, PLOTS_PATH


def plot_box_plot(df, axis):
    df.columns = df.columns.swaplevel(0, 1)
    metrics = df.columns.levels[0]
    print 'metrics', metrics
    for i, c in enumerate(metrics):
        dd = df[c].copy()
        dd.columns = [mapping_dict_cols[a] for a in dd.columns]
        avg = dd['P-NET'].median()
        sns.set_style("whitegrid")
        order = list(dd.median().sort_values().index)
        dd = dd.melt()
        # ax = sns.boxplot(ax=axis[i], x="variable", y="value", data=dd, whis=np.inf, order=order, palette=my_pal, linewidth=1)
        flierprops = dict(marker='o', markersize=1, alpha=0.7)

        ax = sns.boxplot(ax=axis[i], x="variable", y="value", data=dd, whis=1.5, order=order, palette=my_pal,
                         linewidth=1, flierprops=flierprops)
        ax.axhline(avg, ls='--', linewidth=1)
        ax.set_ylim([0.4, 1.05])
        ax.set_ylabel(mapping_dict[c], fontproperties)
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=fontsize)
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.minorticks_off()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


def plot_confusion_matrix_all(ax, adjust_threshold=False):
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

    base_dir = join(PROSTATE_LOG_PATH, 'pnet')
    models_base_dir = join(base_dir, 'onsplit_average_reg_10_tanh_large_testing')
    filename = join(models_base_dir, 'P-net_ALL_testing.csv')
    df = pd.read_csv(filename, index_col=0)

    if adjust_threshold:
        df.pred = df.pred_scores > 0.5
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


def plot_auc_all(ax):
    def plot_roc(ax, y_test, y_pred_score, save_dir, color, label=''):
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, label=label + ' (%0.2f)' % roc_auc, linewidth=1, color=color)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.1, linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontproperties)
        ax.set_ylabel('True Positive Rate', fontproperties)

    def get_prc_data():
        all_models_dict = {}

        base_dir = PROSTATE_LOG_PATH
        models_base_dir = join(base_dir, 'compare/onsplit_ML_test')
        models = ['Linear Support Vector Machine ', 'RBF Support Vector Machine ', 'L2 Logistic Regression',
                  'Random Forest',
                  'Adaptive Boosting', 'Decision Tree']

        for i, m in enumerate(models):
            df = pd.read_csv(join(models_base_dir, m + '_data_0_testing.csv'), sep=',', index_col=0, header=[0, 1])
            all_models_dict[m] = df

        pnet_base_dir = join(base_dir, 'pnet/onsplit_average_reg_10_tanh_large_testing')
        df_pnet = pd.read_csv(join(pnet_base_dir, 'P-net_ALL_testing.csv'), sep=',', index_col=0, header=[0, 1])
        all_models_dict['P-NET'] = df_pnet
        return all_models_dict

    # models = ['Linear Support Vector Machine ', 'RBF Support Vector Machine ', 'L2 Logistic Regression',
    #           'Random Forest',
    #           'Adaptive Boosting', 'Decision Tree']
    # base_dir = PROSTATE_LOG_PATH
    # models_base_dir = join(base_dir, 'compare/onsplit_ML_test')
    # all_models_dict = {}
    # for i, m in enumerate(models):
    #     df = pd.read_csv(join(models_base_dir, m + '_data_0_testing.csv'), sep=',', index_col=0, header=[0, 1])
    #     all_models_dict[m] = df
    all_models_dict = get_prc_data()
    n = len(all_models_dict.keys())
    # sort based on area under prc
    colors = sns.color_palette(None, n)
    sorted_dict = {}
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
        average_auc = metrics.auc(fpr, tpr)
        average_auc = average_precision_score(y_test, y_pred_score)
        sorted_dict[k] = average_auc

    sorted_dict = sorted(sorted_dict.items(), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_dict)

    for i, k in enumerate(sorted_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        plot_roc(ax, y_test, y_pred_score, None, color=colors[i], label=k)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    ax.legend(loc="lower right", bbox_to_anchor=(1.2, 0.0), fontsize=fontsize, framealpha=0.0)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # ax.spines['left'].set_visible(False)


def plot_crossvalidation_metrics(axis):
    df = pd.read_csv(join(models_base_dir, 'folds.csv'), sep=',', index_col=0, header=[0, 1])
    pnet_base_dir = join(base_dir, 'pnet/crossvalidation_average_reg_10_tanh')
    pnet_df = pd.read_csv(join(pnet_base_dir, 'folds.csv'), sep=',', index_col=0, header=[0, 1])
    df = pd.concat([pnet_df, df], axis=1)
    df = df.drop('dense_data_0', axis=1, level=0)
    df = df.drop('Logistic Regression_ALL', axis=1, level=0)
    plot_box_plot(df, axis)
    sns.set_style(None)


mapping_dict = {'accuracy': 'Accuracy', 'auc': 'Area Under Curve (AUC)', 'aupr': 'AUPRC', 'f1': 'F1',
                'precision': 'Precision', 'percision': 'Precision', 'recall': 'Recall'}

base_dir = PROSTATE_LOG_PATH
models_base_dir = join(base_dir, 'compare/crossvalidation_ML_test')

models = ['Decision Tree', 'L2 Logistic Regression', 'Random Forest', 'Ada. Boosting', 'Linear SVM', 'RBF SVM', 'P-NET']

mapping_dict_cols = {'Adaptive Boosting_data_0': 'Ada. Boosting',
                     'Decision Tree_data_0': 'Decision Tree',
                     'L2 Logistic Regression_data_0': 'L2 Logistic Regression',
                     'Linear Support Vector Machine _data_0': 'Linear SVM',
                     'Logistic Regression_ALL': 'Logistic Regression',
                     'P-net_ALL': 'P-NET',
                     'RBF Support Vector Machine _data_0': 'RBF SVM',
                     'Random Forest_data_0': 'Random Forest',
                     }
current_palette = sns.color_palette(None, len(models))

fontsize = 5  # legends, axis
fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}

my_pal = {}
for i, m in enumerate(models):
    print current_palette[i]
    my_pal[m] = current_palette[i]


def run():
    saving_dir = join(PLOTS_PATH, 'extended_data')

    if not exists(saving_dir):
        makedirs(saving_dir)

    # fig = plt.figure(constrained_layout=False, figsize=(8.3 , 11.7))
    fig = plt.figure(constrained_layout=False, figsize=(7.2, 9.72))

    spec2 = gridspec.GridSpec(ncols=2, nrows=5, figure=fig)
    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[0, 1])
    ax3 = fig.add_subplot(spec2[1, 0])
    ax4 = fig.add_subplot(spec2[1, 1])
    ax5 = fig.add_subplot(spec2[2, 0])
    ax6 = fig.add_subplot(spec2[2, 1])
    ax7 = fig.add_subplot(spec2[3, 0])
    ax8 = fig.add_subplot(spec2[3, 1])
    ax9 = fig.add_subplot(spec2[4, 0])
    # ax10 = fig.add_subplot(spec2[4, 1])

    fig.subplots_adjust(left=0.2, bottom=0.15, right=0.8, top=0.99, wspace=0.7, hspace=0.6)
    saving_filename = join(saving_dir, 'figure_ed2.png')

    plot_crossvalidation_metrics(axis=[ax4, ax5, ax6, ax7, ax8, ax9])
    plot_confusion_matrix_all(ax1, adjust_threshold=True)
    plot_confusion_matrix_all(ax2, adjust_threshold=False)
    # plot_confusion_matrix_all(ax7)
    plot_auc_all(ax3)

    # saving_filename = join(saving_dir, 'figure2.pdf')
    # fontproperties['size'] = 20
    # plt.text(0.1,0.1,'aedfcew,erw', fontproperties)
    plt.savefig(saving_filename, dpi=300)
    # saving_filename = join(saving_dir, 'figure_ed2.pdf')
    # plt.savefig(saving_filename)


if __name__ == "__main__":
    run()
