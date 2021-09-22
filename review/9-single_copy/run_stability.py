import os
from os.path import join, exists

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config_path import PROSTATE_LOG_PATH, PLOTS_PATH


def get_stability_index(df, n_features):
    n = len(df.columns)
    pairs = []
    for i in range(n):
        for j in np.arange(i, n):
            if i != j:
                pairs.append((i, j))


    overlaps = []
    for pair in pairs:
        first, second = pair
        a = df.iloc[:, first].nlargest(n_features)
        b = df.iloc[:, second].nlargest(n_features)
        overlap = len(set(a.index).intersection(b.index))
        overlaps.append(overlap)

    avg_overlap = np.mean(overlaps) / n_features
    return avg_overlap


def plot_stability(model_name, n_features):
    filename = model_name
    df = pd.read_csv(filename, index_col=[0, 1])

    stability_indeces = []
    for f in n_features:
        stability_index = get_stability_index(df, f)
        stability_indeces.append(stability_index)
        print f, stability_index

    plt.plot(n_features, stability_indeces, '*-')
    plt.ylabel('stability index', fontsize=fontsize)
    plt.xlabel('# best features_processing', fontsize=fontsize)


n_features = [200, 100, 50, 20, 10]
base_dir1 = join(PROSTATE_LOG_PATH, 'review/9single_copy/crossvalidation_average_reg_10_tanh_single_copy/fs')
base_dir2 = join(PROSTATE_LOG_PATH, 'pnet/crossvalidation_average_reg_10_tanh/fs')

saving_dir = join(PLOTS_PATH, 'reviews/9-single_copy')

if not exists(saving_dir):
    os.mkdir(saving_dir)

files = []
f = join(base_dir1, 'coef.csv')
files.append(f)
f = join(base_dir2, 'coef.csv')
files.append(f)
models = ['single copy', 'two copies']

fig = plt.figure()
fig.set_size_inches(6, 4)
fontsize = 12
for m in files:
    print m
    plot_stability(m, n_features)

plt.legend([m.replace('.csv', '') for m in models], fontsize=fontsize)
plt.xlabel('Number of top features')
plt.subplots_adjust(bottom=0.15)
plt.savefig(join(saving_dir, 'compare.png'), dpi=200)
