import os
from os.path import join, basename, dirname

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config_path import PROSTATE_DATA_PATH, PLOTS_PATH, BASE_PATH

print(PROSTATE_DATA_PATH)
extracted_dir = join(BASE_PATH, 'analysis/figure_3/extracted')

print (extracted_dir)
node_importance = pd.read_csv(join(extracted_dir, 'node_importance_graph_adjusted.csv'), index_col=0)
response = pd.read_csv(join(extracted_dir, 'response.csv'), index_col=0)

l = 1
high_nodes = node_importance[node_importance.layer == l]
col = 'coef'

importance = high_nodes.sort_values(col, ascending=False)
importance['rank'] = range(1, len(importance) + 1)
importance_log = np.log(importance[col].values + 1)

interesting = ['FOXA1', 'SPOP', 'MED12', 'CDK12', 'PIK3CA', 'CHD1', 'ZBTB7B']

annotations = (importance.loc[interesting]).sort_values('rank')
plt.figure(figsize=(6, 4))
ax = plt.subplot()
plt.plot(importance_log, ".")
plt.ylabel('Log (importance score +1)')

xytext = [(20, 60), (20, 50), (40, 40), (30, 30), (0, 10), (-10, 10), (0, 10)]
for i, gene in enumerate(annotations.index):
    x = annotations.loc[gene, 'rank']
    y = annotations.loc[gene, col]
    connectivity = annotations.loc[gene, 'coef_graph']
    print('gene {}, rank {}, importance,  {},  connectivity {}'.format(gene, x, y, connectivity))

    ax.annotate(gene, (x, y),
                xycoords='data',
                fontsize=8,
                bbox=dict(boxstyle="round", fc="none", ec="gray"),
                xytext=xytext[i], textcoords='offset points', ha='center',
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

filename = join(saving_dir, '7-all_genes.png')
plt.savefig(filename, dpi=200)
