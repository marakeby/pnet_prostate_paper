from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker

from config_path import PROSTATE_LOG_PATH
from setup import saving_dir

custom_rcParams = {
    'figure.figsize': (8, 3),
    'font.family': 'Arial',
    'font.size': 14,
    'font.weight': 'regular',
    'axes.labelsize': 14,
    'axes.formatter.useoffset': False,
    'axes.formatter.limits': (-4, 4),
    'axes.titlesize': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'pdf.fonttype': 42
}


# mpl.rcParams.update(custom_rcParams)
# sns.set_context('paper', rc=custom_rcParams)
# sns.set_style("white", {"grid.linestyle": u'--', "axes.grid": True, "grid.color":"0.9"})


def plot_box_plot(df, save_dir):
    df.columns = df.columns.swaplevel(0, 1)
    columns = df.columns.levels[0]
    print 'columns', columns
    for c in columns:
        plt.figure(figsize=(7, 5))
        dd = df[c].copy()
        dd.columns = [mapping_dict_cols[a] for a in dd.columns]
        model_names = dd.columns

        avg = dd['P-NET'].median()
        sns.set_style("whitegrid")
        order = list(dd.median().sort_values().index)
        dd = dd.melt()
        ax = sns.boxplot(x="variable", y="value", data=dd, whis=np.inf, order=order, palette=my_pal)
        ax.axhline(avg, ls='--')

        plt.ylim([0.4, 1.05])
        ax.set_ylabel(mapping_dict[c], fontdict=dict(family='Arial', weight='bold', fontsize=14))

        ax.set_xlabel('')
        plt.tight_layout()
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=14)
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.gcf().subplots_adjust(bottom=0.3, left=0.2)
        plt.savefig(join(save_dir, c + '_boxplot'), dpi=100)


mapping_dict = {'accuracy': 'Accuracy', 'auc': 'Area Under Curve (AUC)',
                'aupr': 'AUPRC', 'f1': 'F1', 'precision': 'Precision', 'percision': 'Precision', 'recall': 'Recall'}

base_dir = PROSTATE_LOG_PATH
models_base_dir = join(base_dir, 'compare/crossvalidation_ML_test')

models = ['Decision Tree', 'L2 Logistic Regression', 'Random Forest', 'Ada. Boosting', 'Linear SVM', 'RBF SVM',
          'P-NET']

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

my_pal = {}
for i, m in enumerate(models):
    print current_palette[i]
    my_pal[m] = current_palette[i]


def run():
    df = pd.read_csv(join(models_base_dir, 'folds.csv'), sep=',', index_col=0, header=[0, 1])
    print df.head()

    pnet_base_dir = join(base_dir, 'pnet/crossvalidation_average_reg_10_tanh')
    pnet_df = pd.read_csv(join(pnet_base_dir, 'folds.csv'), sep=',', index_col=0, header=[0, 1])
    print pnet_df.head()

    df = pd.concat([pnet_df, df], axis=1)

    df = df.drop('dense_data_0', axis=1, level=0)
    df = df.drop('Logistic Regression_ALL', axis=1, level=0)

    plot_box_plot(df, saving_dir)
    sns.set_style(None)
    plt.close()
    plt.close()


if __name__ == "__main__":
    run()
