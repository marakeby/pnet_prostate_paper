from os.path import join

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from upsetplot import UpSet

from config_path import PLOTS_PATH
from data.data_access import Data

selected_genes = ['AR', 'TP53', 'MDM4', 'CDK4', 'CDK6', 'CDKN2A', 'RB1']
data_params = {'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
               'drop_AR': False,
               'cnv_levels': 5,
               'mut_binary': True,
               'balanced_data': False,
               'combine_type': 'union',  # intersection
               'use_coding_genes_only': False,
               'selected_genes': selected_genes}

data_access_params = {'id': 'id', 'type': 'prostate_paper', 'params': data_params}

data_adapter = Data(**data_access_params)
x, y, info, columns = data_adapter.get_data()
x_df = pd.DataFrame(x, columns=columns, index=info)
x_df.head()

x_df_3 = x_df.copy()
x_df_3[x_df_3 < 1] = 0  # remove single copy

x_df_3 = x_df_3.T.reset_index().groupby('level_0').sum()  # all events (OR)
x_df_3[x_df_3 > 0] = 1  # binarize
print x_df_3.shape

x_df_3_binary = x_df_3.T > 0.
x_df_3_binary = x_df_3_binary.set_index(selected_genes)

y_ind = y > 0
x_df_mets_3 = x_df_3.T[y_ind].T

x_df_mets_3_binary = x_df_mets_3.T > 0.
print x_df_mets_3_binary.shape

x_df_mets_3_binary = x_df_mets_3_binary.set_index(selected_genes)

font = {'family': 'Arial',
        'weight': 'normal',
        'size': 5}
matplotlib.rc('font', **font)
dd = x_df_3_binary.reset_index().set_index(['AR', 'TP53', 'MDM4'])

upset = UpSet(dd, subset_size='count', intersection_plot_elements=6, show_counts=True, with_lines=True, element_size=10)
fig = plt.figure(constrained_layout=False, figsize=(8, 6))
upset.plot(fig)
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.08, right=0.99)

saving_dir = join(PLOTS_PATH, 'figure4')
filename = join(saving_dir, 'figure4_ar_tp53_mdm4.png')
plt.savefig(filename, dpi=300)
matplotlib.rcParams['pdf.fonttype'] = 42
filename = join(saving_dir, 'figure4_ar_tp53_mdm4.pdf')
plt.savefig(filename)
