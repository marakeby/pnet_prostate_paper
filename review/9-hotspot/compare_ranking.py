import os
from os.path import join, dirname, basename

import pandas as pd
from matplotlib import pyplot as plt

from config_path import PLOTS_PATH, PROSTATE_LOG_PATH

base_dir = join(PROSTATE_LOG_PATH, 'review/9hotspot')
files = []
files.append(dict(Model='Mutations with Hotspot adjusted',
                  file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_hotspot')))
files.append(dict(Model='All mutations', file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_count')))
dirs_df = pd.DataFrame(files)

layers = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']


def read_feature_ranks(dirs_df):
    model_dict = {}
    for l in layers:
        coef_df_list = []
        keys = []
        for i, row in dirs_df.iterrows():
            dir_ = row.file
            model = row.Model
            dir_ = join(dir_, 'fs')
            f = 'coef_P-net_ALL_layer{}.csv'.format(l)
            coef_file = join(dir_, f)
            coef_df = pd.read_csv(coef_file, index_col=0)
            coef_df.columns = [model]
            coef_df_list.append(coef_df)
            keys.append(model)
        coef_df = pd.concat(coef_df_list, axis=1)
        model_dict[l] = coef_df

    return model_dict


coef_df_dict = read_feature_ranks(dirs_df)

from scipy.stats import wilcoxon

layers = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']
fig = plt.figure()
pointbiserialr_list = []
pearsonr_list = []

wilcoxon_list = []
wilcoxon_fusion_genes_list = []
for l in layers:
    hotspot = coef_df_dict[l]['Mutations with Hotspot adjusted']
    base = coef_df_dict[l]['All mutations']

    w, p = wilcoxon(hotspot, base)
    wilcoxon_list.append((w, p))

# plt.plot(wilcoxon_list)

for w in wilcoxon_list:
    print w

n = 20
common_list = []
for l in layers:
    ranked = coef_df_dict[l].abs().rank(ascending=False)
    top10_hotspot = ranked['Mutations with Hotspot adjusted'].nsmallest(n).index
    top10 = ranked['All mutations'].nsmallest(n).index
    c = len(set(top10_hotspot).intersection(top10))
    print c
    common_list.append(c / float(n))

plt.plot(common_list, '-.')
plt.ylim(0, 1)
plt.ylabel('Percent of common nodes')
plt.xlabel('Layers')
plt.xticks(range(len(layers)), layers)

current_dir = basename(dirname(__file__))
saving_dir = join(PLOTS_PATH, 'reviews/{}'.format(current_dir))
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

plt.savefig(join(saving_dir, 'common_top'), dpi=200)
