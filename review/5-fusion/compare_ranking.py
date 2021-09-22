import os
from os.path import join, dirname, basename

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

wilcoxon_fusion_list = []
wilcoxon_fusion_genes_list = []
for l in layers:
    fusion_genes = coef_df_dict[l]['Fusion (genes)']
    fusion = coef_df_dict[l]['Fusion']
    base = coef_df_dict[l]['no-Fusion']

    w, p = wilcoxon(fusion_genes, base)
    wilcoxon_fusion_genes_list.append((w, p))

    w, p = wilcoxon(fusion, base)
    wilcoxon_fusion_list.append((w, p))

print 'Fusion'
for w in wilcoxon_fusion_list:
    print w
print 'Fusion (genes)'
for w in wilcoxon_fusion_genes_list:
    print w

n = 20
common_list = []
common_list_genes = []
for l in layers:
    ranked = coef_df_dict[l].abs().rank(ascending=False)
    top10_fusions_genes = ranked['Fusion (genes)'].nsmallest(n).index
    top10_fusions = ranked['Fusion'].nsmallest(n).index
    top10_nofusions = ranked['no-Fusion'].nsmallest(n).index

    c = len(set(top10_fusions).intersection(top10_nofusions))
    common_list.append(100. * c / float(n))

    c = len(set(top10_fusions_genes).intersection(top10_nofusions))
    common_list_genes.append(100. * c / float(n))

plt.plot(common_list, '-.')
plt.plot(common_list_genes, '-.')
plt.ylim(0, 100)
plt.ylabel('Percent of common nodes (%)')
plt.xlabel('Layers')
plt.legend(['Fusion', 'Fusion (genes)'])
plt.xticks(range(len(layers)), layers)

current_dir = basename(dirname(__file__))
saving_dir = join(PLOTS_PATH, 'reviews/{}'.format(current_dir))
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

plt.savefig(join(saving_dir, 'common_top'), dpi=200)
