import os
from os.path import join, basename, dirname

import pandas as pd
from matplotlib import pyplot as plt

from config_path import PLOTS_PATH, PROSTATE_LOG_PATH

base_dir = join(PROSTATE_LOG_PATH, 'review/9hotspot')

files = []

files.append(dict(Model='Mutations with Hotspot adjusted',
                  file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_hotspot')))
files.append(dict(Model='All mutations', file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_count')))
dirs_df = pd.DataFrame(files)


def get_contributions(fusion='no-Fusion'):
    f = 'fs/coef_P-net_ALL_layerinputs.csv'
    base_dir = dirs_df[dirs_df.Model == fusion].file.values[0]
    f = join(base_dir, f)
    coef_df = pd.read_csv(f)
    if fusion == 'Fusion':
        coef_df.columns = ['type', 'gene', 'feature', 'coef']
    else:
        coef_df.columns = ['gene', 'feature', 'coef']
    coef_df['coef_abs'] = coef_df.coef.abs()
    coef_df.head()
    plot_df = coef_df.groupby('feature').coef_abs.sum()
    plot_df = 100 * plot_df / plot_df.sum()
    plot_df.sort_values()
    plot_df = plot_df.to_frame()
    plot_df.columns = [fusion]
    return plot_df


models = ['Mutations with Hotspot adjusted', 'All mutations']

contibution_list = []
for m in models:
    df = get_contributions(fusion=m)
    contibution_list.append(df)

plot_df = pd.concat(contibution_list, axis=1, sort=False)

D_id_color = {'Amplification': [0.8784313725490196, 0.4823529411764706, 0.2235294117647059, 0.7],
              'Mutation': [0.4117647058823529, 0.7411764705882353, 0.8235294117647058, 0.7],
              'Deletion': [0.00392156862745098, 0.21568627450980393, 0.5803921568627451, 0.7],

              }

mapping = {'cnv_amp': 'Amplification', 'cnv_del': 'Deletion', 'mut_important': 'Mutation',
           'mut_important_plus_hotspots': 'Mutation', 'fusion_genes': 'Fusion (genes)',
           'fusion_indicator': 'Fusion (indicator)'}
plot_df = plot_df.rename(index=mapping)
plot_df.fillna(0, inplace=True)
color = [D_id_color[i] for i in plot_df.index]

fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(6, 7), dpi=200)

print plot_df
plot_df.T.plot.bar(stacked=True, color=color, rot=0)
plt.subplots_adjust(bottom=0.3)
plt.legend(fontsize=8, bbox_to_anchor=(.7, -0.1))
plt.ylabel('Percent of relative contribution of data types (%)')

current_dir = basename(dirname(__file__))
saving_dir = join(PLOTS_PATH, 'reviews/{}'.format(current_dir))
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

plt.savefig(join(saving_dir, 'contibutions.png'), dpi=400)
