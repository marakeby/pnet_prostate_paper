from os.path import join

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from config_path import PROSTATE_DATA_PATH
from setup import saving_dir


def sigmoid(x, L, k, x0):
    b = 0.
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)


def run():
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5, 4), dpi=200)
    filename = join(PROSTATE_DATA_PATH, 'functional/Data S3 CRISPR and RO-5963 results.xlsx')
    df = pd.read_excel(filename, sheet_name='RO drug curves', header=[0, 1], index_col=0)
    cols = ['LNCaP'] * 6 + ['PC3'] * 6 + ['DU145'] * 6

    exps = ['LNCaP', 'PC3', 'DU145']
    print exps
    colors = {'LNCaP': 'maroon', 'PC3': '#577399', 'DU145': 'orange'}

    X = df.index.values
    legend_labels = []
    legnds = []
    for i, exp in enumerate(exps):
        print exp
        legend_labels.append(exp)
        df_exp = df[exp].copy()
        stdv = df_exp.std(axis=1)
        mean = df_exp.mean(axis=1)
        ydata = df_exp.values.flatten()
        xdata = np.repeat(X, 6)
        p0 = [1.0, -1., -.7]  # initial guess
        popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox', maxfev=60000)
        plt.errorbar(X, mean, yerr=stdv, fmt='o', ms=5, color=colors[exp], alpha=0.75, capsize=3, label=exp)
        x2 = np.linspace((min(xdata), max(xdata)), 10)
        y2 = sigmoid(x2, *popt)
        plt.plot(x2, y2, color=colors[exp], alpha=0.75, linewidth=2)

        legnds.append(mpatches.Patch(color=colors[exp], label=exp))

    plt.xscale('log')
    plt.ylim((-.6, 1.6))
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8)
    plt.ylabel('Relative Viability', fontdict=dict(family='Arial', weight='bold', fontsize=14))
    plt.xlabel(u'\u03bcM RO-5963', fontdict=dict(family='Arial', weight='bold', fontsize=14))
    ax.spines['bottom'].set_position(('data', 0.))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    plt.xlim((.02, 120))

    ax.legend(handles=legnds, bbox_to_anchor=(.9, 1.), framealpha=0.0)
    plt.savefig(join(saving_dir, 'drug.png'))


if __name__ == "__main__":
    run()
