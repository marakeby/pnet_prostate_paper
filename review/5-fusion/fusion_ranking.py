import os
from os.path import join, basename, dirname

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config_path import PLOTS_PATH, PROSTATE_LOG_PATH

base_dir = join(PROSTATE_LOG_PATH, 'review/fusion')
files = []
files.append(dict(Model='Fusion', file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_fusion')))
files.append(dict(Model='no-Fusion', file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_fusion_zero')))
files.append(
    dict(Model='Fusion (genes)', file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_inner_fusion_genes')))
dirs_df = pd.DataFrame(files)

f = 'fs/coef_P-net_ALL_layerinputs.csv'
base_dir = dirs_df[dirs_df.Model == 'Fusion'].file.values[0]
f = join(base_dir, f)

coef_df = pd.read_csv(f)
coef_df.columns = ['type', 'gene', 'feature', 'coef']
coef_df['coef_abs'] = coef_df.coef.abs()
plot_df = coef_df.groupby('feature').coef_abs.sum()
plot_df = 100 * plot_df / plot_df.sum()
plot_df.sort_values()

plt.figure(figsize=(6, 4))
ax = plt.subplot()

col = 'coef_abs'
importance = coef_df.sort_values(col, ascending=False)
importance['rank'] = range(1, len(importance) + 1)
importance_log = np.log(importance[col].values + 1)

plt.plot(importance_log, ".")
plt.ylabel('Log (importance score +1)')

ind = importance.feature == 'fusion_indicator'
y = importance_log[ind][0]
x = importance.loc[ind, 'rank'].values[0]
ax.annotate('Fusion indicator', (x, y),
            xycoords='data',
            fontsize=8,
            bbox=dict(boxstyle="round", fc="none", ec="gray"),
            xytext=(60, 40), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))

fontProperties = dict(family='Arial', weight='normal', size=14, rotation=0, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontProperties)

plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

current_dir = basename(dirname(__file__))
saving_dir = join(PLOTS_PATH, 'reviews/{}'.format(current_dir))
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

plt.savefig(join(saving_dir, 'fusion_indicator_ranking'), dpi=200)
