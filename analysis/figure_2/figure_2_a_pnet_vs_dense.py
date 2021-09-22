from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from scipy import stats

from config_path import PROSTATE_LOG_PATH, PROSTATE_DATA_PATH
from setup import saving_dir


def get_dense_sameweights(col='f1'):
    filename = join(PROSTATE_LOG_PATH, 'number_samples/crossvalidation_number_samples_dense_sameweights')
    filename = filename + '/folds.csv'
    df = pd.read_csv(filename, index_col=0, header=[0, 1])
    # print df.head()
    dd = df.swaplevel(0, 1, axis=1)[col].head()
    df_pnet_col = [c for c in dd.columns if 'dense' in c]
    df_pnet = dd[df_pnet_col]
    return df_pnet


def get_pnet_preformance(col='f1'):
    filename = join(PROSTATE_LOG_PATH, 'number_samples/crossvalidation_average_reg_10_tanh')
    filename = filename + '/folds.csv'
    df = pd.read_csv(filename, index_col=0, header=[0, 1])
    dd = df.swaplevel(0, 1, axis=1)[col].head()
    df_pnet_col = [c for c in dd.columns if 'P-net' in c]
    df_pnet = dd[df_pnet_col]
    return df_pnet


def plot_compaison(ax1, label, df_pnet, df_dense):
    y1 = df_pnet.mean()
    dy = df_pnet.std()
    x = sizes
    sns.set_color_codes('muted')
    current_palette = sns.color_palette()
    colors = current_palette[0:2]
    ax1.plot(x, y1, linestyle='-', marker='o', color=colors[0])
    ax1.fill_between(x, y1 - dy, y1 + dy, color=colors[0], alpha=0.2)
    y2 = df_dense.mean()
    dy = df_dense.std()
    ax1.plot(x, y2, linestyle='-', marker='o', color=colors[1])
    ax1.fill_between(x, y2 - dy, y2 + dy, color=colors[1], alpha=0.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylabel(label, fontdict=dict(family='Arial', weight='bold', fontsize=14))
    ax1.legend(['P-NET', 'Dense'], fontsize=8, loc='upper left')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


def plot_(df_pnet, df_dense):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 8), dpi=600)
    y1 = df_pnet.mean()
    dy = df_pnet.std()
    x = sizes
    ax1.plot(x, y1, linestyle='-', marker='o', color='red')
    ax1.fill_between(x, y1 - dy, y1 + dy, color='mistyrose', alpha=0.3)
    y2 = df_dense.mean()
    dy = df_dense.std()

    ax1.plot(x, y2, linestyle='-', marker='o', color='black')
    ax1.fill_between(x, y2 - dy, y2 + dy, color='gray', alpha=0.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.set_xticks([], [])

    plt.xlabel('Number of samples', fontdict=dict(weight='bold', fontsize=14))
    ax1.legend(['P-net', 'Dense'], fontsize=10)
    ax1.set_ylabel(l, fontdict=dict(weight='bold', fontsize=14))

    ratio = (y1.values - y2.values) / y2.values
    pvalues = get_stats(df_pnet, df_dense)

    print zip(sizes, ratio)
    print zip(sizes, pvalues)

    bar_width = 25
    fontsize = 10
    boxpad = 0.3
    ax2.bar(sizes, ratio, width=bar_width)
    for i, p in enumerate(pvalues):
        bar_centers = sizes[i] + np.array([0.5, 1.5]) * bar_width
        start = bar_centers[0]
        end = bar_centers[1]
        height = .005 + ratio[i]
        if p >= 0.05:
            displaystring = r'n.s.'
        elif p < 0.0001:
            displaystring = r'***'
        elif p < 0.001:
            displaystring = r'**'
        else:
            displaystring = r'*'

        ax2.text(sizes[i], height, displaystring, ha='center', va='center')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    ax2.set_ylabel('Average increase ratio', fontdict=dict(weight='bold', fontsize=14))


sizes = []
for i in range(0, 20, 3):
    df_split = pd.read_csv(join(PROSTATE_DATA_PATH, 'splits/training_set_{}.csv'.format(i)), index_col=0)
    sizes.append(df_split.shape[0])
print sizes


def get_stats(df_pnet, df_dense):
    print df_pnet.shape, df_dense.shape
    pvalues = []
    for c1, c2 in zip(df_pnet.columns, df_dense.columns):
        # print c
        x = df_pnet.loc[:, c1]
        y = df_dense.loc[:, c2]

        twosample_results = stats.ttest_ind(x, y)
        pvalue = twosample_results[1] / 2
        print pvalue
        pvalues.append(pvalue)
    return pvalues


cols = ['f1', 'auc', 'aupr', 'precision', 'recall', 'accuracy']

labels = ['F1 score', 'Area Under ROC Curve (AUC)', 'Area Under Precision Recall Curve', 'Precision', 'Recall',
          'Accuracy']


def plot_pnet_vs_dense_auc(ax):
    c = 'auc'
    df_dense_sameweights = get_dense_sameweights(c)
    df_pnet = get_pnet_preformance(col=c)
    plot_compaison(ax, 'AUC', df_pnet, df_dense_sameweights)


def plot_pnet_vs_dense_with_ratio(ax, c, label, plot_ratio=False):
    sns.set_color_codes('muted')
    current_palette = sns.color_palette()
    color = current_palette[3]

    sizes = []
    for i in range(0, 20, 3):
        df_split = pd.read_csv(join(PROSTATE_DATA_PATH, 'splits/training_set_{}.csv'.format(i)), index_col=0)
        sizes.append(df_split.shape[0])
    sizes = np.array(sizes)

    df_dense_sameweights = get_dense_sameweights(c)
    df_pnet = get_pnet_preformance(col=c)
    pvalues = get_stats(df_pnet, df_dense_sameweights)
    print c, zip(pvalues, sizes)
    plot_compaison(ax, label, df_pnet, df_dense_sameweights)
    # ax.legend(ax.legend.text, loc= 'upper left')

    y1 = df_pnet.mean()
    y2 = df_dense_sameweights.mean()
    height = map(max, zip(y1, y2))
    print 'height', height
    updated_values = []
    for i, (p, s) in enumerate(zip(pvalues, sizes)):
        if p >= 0.05:
            displaystring = r'n.s.'
        elif p < 0.0001:
            displaystring = r'***'
        elif p < 0.001:
            displaystring = r'**'
        else:
            displaystring = r'*'
        updated_values.append('{:.0f}\n({})'.format(s, displaystring))
        ax.axvline(x=s, ymin=0, linestyle='--', alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks([], [])

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis=u'x', which=u'both', length=0)

    ax.set_xticks(sizes)
    ax.set_xticklabels(updated_values)
    ax.set_xlim((min(sizes) - 5, max(sizes) + 50))

    if plot_ratio:
        ax2 = ax.twinx()
        y1 = df_pnet.mean()
        y2 = df_dense_sameweights.mean()
        ratio = (y1.values - y2.values) / y2.values
        new_x = np.linspace(min(sizes), max(sizes), num=np.size(sizes))
        coefs = np.polyfit(sizes, ratio, 3)
        new_line = np.polyval(coefs, new_x)

        ax2.plot(new_x, new_line, '-.', linewidth=1, color=color)
        ax2.set_ylim((0.005, .23))
        ax.set_ylim((.5, 1.05))
        ax2.set_ylabel('Performance increase', fontdict=dict(family='Arial', weight='bold', fontsize=14, color=color))
        vals = ax2.get_yticks()
        ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax.set_yticks([], minor=True)
        ax2.spines['right'].set_color(color)
        ax2.yaxis.label.set_color(color)
        ax2.tick_params(axis='y', colors=color)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

    ax.set_xlabel('Number of samples', fontdict=dict(family='Arial', weight='bold', fontsize=14))
    size_vals = ax.get_xticks()
    pvalues_dict = {}
    for p, s in zip(pvalues, sizes):
        pvalues_dict[s] = p
    return pvalues_dict


def plot_pnet_vs_dense_ratio(ax2, c='auc'):
    df_dense = get_dense_sameweights(c)
    df_pnet = get_pnet_preformance(col=c)
    y1 = df_pnet.mean()
    y2 = df_dense.mean()

    ratio = (y1.values - y2.values) / y2.values
    pvalues = get_stats(df_pnet, df_dense)
    print zip(sizes, ratio)
    print zip(sizes, pvalues)

    bar_width = 30
    fontsize = 10
    boxpad = 0.3
    ax2.bar(sizes, ratio, width=bar_width)
    for i, p in enumerate(pvalues):
        bar_centers = sizes[i] + np.array([0.5, 1.5]) * bar_width
        start = bar_centers[0]
        end = bar_centers[1]
        height = .005 + ratio[i]
        if p >= 0.05:
            displaystring = r'n.s.'
        elif p < 0.0001:
            displaystring = r'***'
        elif p < 0.001:
            displaystring = r'**'
        else:
            displaystring = r'*'

        ax2.text(sizes[i], height, displaystring, ha='center', va='center')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        vals = ax2.get_yticks()
        ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    ax2.set_ylabel('Performance increase', fontdict=dict(family='Arial', weight='bold', fontsize=12))
    ax2.set_xlabel('Number of samples', fontdict=dict(family='Arial', weight='bold', fontsize=12))


base_dir = PROSTATE_LOG_PATH
models_base_dir = join(base_dir, 'compare/onsplit_ML_test_Apr-11_11-34')


def run_pnet():
    # main figure AUC
    fig = plt.figure(figsize=(8, 5))
    ax = fig.subplots(1, 1)
    plot_pnet_vs_dense_with_ratio(ax, 'auc', 'Area Under Curve (AUC)', plot_ratio=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.subplots_adjust(bottom=0.15, right=0.85, left=0.15)
    filename = join(saving_dir, 'pnet_vs_dense_sameweights_auc_with_ratio.png')
    plt.close()
    # supp figures other metrics
    pvalus_list = []
    for c, l in zip(cols, labels):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.subplots(1, 1)
        pvalues = plot_pnet_vs_dense_with_ratio(ax, c, label=l, plot_ratio=False)
        pvalus_list.append(pvalues)
        plt.subplots_adjust(bottom=0.15)
        filename = join(saving_dir, 'pnet_vs_dense_sameweights_{}.png'.format(c))
        plt.savefig(filename.format(c), dpi=200)
        plt.close()

    df = pd.DataFrame(pvalus_list, index=labels)
    filename = join(saving_dir, 'pnet_vs_dense_sameweights_tests.csv')
    df.round(3).to_csv(join(saving_dir, filename))


if __name__ == '__main__':
    run_pnet()
