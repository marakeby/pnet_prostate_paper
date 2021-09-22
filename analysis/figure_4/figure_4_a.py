from os.path import join

# matplotlib.style.use('ggplot')
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data.data_access import Data
from setup import saving_dir

data_params = {'id': 'cnv', 'type': 'prostate_paper', 'params': {'data_type': 'cnv', 'drop_AR': False}}
data = Data(**data_params)
x, y, info, columns = data.get_data()
df_cnv = pd.DataFrame(x, index=info, columns=list(columns))
df_cnv['y'] = y

data_params = {'id': 'mut', 'type': 'prostate_paper', 'params': {'data_type': 'mut_important', 'drop_AR': False}}
data = Data(**data_params)
x_mut, y_mut, info_mut, columns_mut = data.get_data()
df_mut = pd.DataFrame(x_mut, index=info_mut, columns=list(columns_mut))
df_mut['y'] = y_mut


# def plot_hist_mut(df, gene_name, ax):
#     primary_df = df[df['y'] == 0]
#     mets_df = df[df['y'] == 1]
#     p= primary_df[gene_name].value_counts()
#     m= mets_df[gene_name].value_counts()
#     summary = pd.concat([ m, p], axis=1, keys=[  'Metastatic', 'Primary'])
#     summary.fillna(0, inplace =True)
#     summary = 100. * summary / summary.sum()
#     summary.plot.bar(stacked=False, ax=ax, alpha=0.7)
#     # ax.set(xlabel='# mutations', ylabel='Count')
#     ax.set_xlabel('# mutations', fontdict=dict(family='Arial', weight='bold', fontsize=12))
#     ax.set_ylabel('Count', fontdict=dict(family='Arial', weight='bold', fontsize=12))
#
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     for i in ax.patches:
#         # get_x pulls left or right; get_height pushes up or down
#         if i.get_height()>1.0:
#             ax.text(i.get_x() + 0.3*i.get_width(), i.get_height() + 20.0, '{:5.1f} %'.format(i.get_height()), fontsize=10,
#                     color='dimgrey', rotation=90)


def plot_hist_cnv(df, gene_name, ax):
    ind = df[gene_name].unique()
    primary_df = df[df['y'] == 0]
    mets_df = df[df['y'] == 1]
    p = primary_df[gene_name].value_counts()
    m = mets_df[gene_name].value_counts()

    # index = pd.DataFrame(index=[-2., -1., 0., 1., 2.])
    index = pd.DataFrame(index=ind)
    summary = pd.concat([m, p], axis=1, keys=['Metastatic', 'Primary'])
    summary = summary.join(index, how='right')
    summary.fillna(0, inplace=True)
    summary = 100. * summary / summary.sum()
    summary.plot.bar(stacked=False, ax=ax, color=['red', 'black'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Copy Number', fontdict=dict(family='Arial', weight='bold', fontsize=12))
    ax.set_ylabel('Percent (%)', fontdict=dict(family='Arial', weight='bold', fontsize=12))

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        if i.get_height() > 1.0:
            ax.text(i.get_x() + 0.3 * i.get_width(), i.get_height() + 20.0, '{:5.1f} %'.format(i.get_height()),
                    fontsize=10,
                    color='dimgrey', rotation=90)
    # ax.set(xlabel='Copy Number', ylabel='Percent (%)')


# # current_dir = dirname(realpath(__file__))
# current_dir= join(saving_dir, 'output')

def plot_stacked_hist_cnv(df, gene_name, ax):
    ind = np.sort(df[gene_name].unique())
    print ind
    primary_df = df[df['y'] == 0]
    mets_df = df[df['y'] == 1]
    p = primary_df[gene_name].value_counts()
    m = mets_df[gene_name].value_counts()

    mapping = {0: 'Neutral', 1: 'Amplification', 2: 'High amplification', -1: 'Deletion', -2: 'Deep deletion'}

    index = pd.DataFrame(index=ind)
    summary = pd.concat([m, p], axis=1, keys=['Metastatic', 'Primary'])
    summary = summary.join(index, how='right')
    summary.fillna(0, inplace=True)
    summary = 100. * summary / summary.sum()

    summary = summary.rename(index=mapping)
    D_id_color = {'High amplification': 'maroon', 'Amplification': 'lightcoral', 'Neutral': 'gainsboro',
                  'Deletion': 'skyblue', 'Deep deletion': 'steelblue'}

    color = [D_id_color[i] for i in summary.index]
    bars = summary.T.plot.bar(stacked=True, ax=ax, color=color)

    # ax.set_xticklabels(['Metastatic', 'Primary'], fontsize=10, rotation=0)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=8, bbox_to_anchor=(.8, -0.1))

    ax.set_ylabel('Sample percent (%)', fontdict=dict(family='Arial', weight='bold', fontsize=12))

    # ax.set_title(gene_name, fontdict=dict(family='Arial', weight='bold', fontsize=12))
    ax.set_title('Copy number variations', fontdict=dict(family='Arial', weight='normal', fontsize=10))


def plot_stacked_hist_mut(df, gene_name, ax):
    ind = np.sort(df[gene_name].unique())
    print ind
    primary_df = df[df['y'] == 0]
    mets_df = df[df['y'] == 1]
    p = primary_df[gene_name].value_counts()
    m = mets_df[gene_name].value_counts()

    # mapping={0:'Neutral', 1:'Amplification', 2:'High amplification',-1:'Deletion',-2:'Deep deletion'}

    index = pd.DataFrame(index=ind)
    summary = pd.concat([m, p], axis=1, keys=['Metastatic', 'Primary'])
    summary = summary.join(index, how='right')
    summary.fillna(0, inplace=True)
    summary = 100. * summary / summary.sum()

    # summary = summary.rename(index=mapping)
    D_id_color = {0: 'gainsboro', 3: 'maroon', 1: 'lightcoral', 2: 'red'}

    # color= [D_id_color[i] for i in summary.index]

    # current_palette = sns.color_palette("dark:salmon_r", as_cmap=True)
    # current_palette = sns.light_palette("seagreen", as_cmap=False, start=0.5)
    current_palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=False)

    colors = [D_id_color[int(c)] for c in summary.index]
    bars = summary.T.plot.bar(stacked=True, ax=ax, colors=colors)

    # bars = summary.T.plot.bar(stacked=True, ax=ax, color=color )

    # ax.set_xticklabels(['Metastatic', 'Primary'], fontsize=8, rotation=0)
    ax.tick_params(axis='x', labelrotation=0.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=8, bbox_to_anchor=(.7, -0.1))

    ax.set_ylabel('Sample percent (%)', fontdict=dict(family='Arial', weight='bold', fontsize=12))

    # ax.set_title(gene_name, fontdict=dict(family='Arial', weight='bold', fontsize=12))
    ax.set_title('Number of mutations', fontdict=dict(family='Arial', weight='normal', fontsize=10))


selected_genes = ['AR', 'TP53', 'PTEN', 'FGFR1', 'MDM4', 'RB1', 'NOTCH1', 'MAML3', 'PDGFA', 'EIF3E']


# columns= set(df.columns).intersection(df_mut.columns)

def run():
    for g in selected_genes:
        # fig =plt.figure()
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 4), dpi=200)
        plt.ylim(0, 110)
        plt.subplots_adjust(bottom=0.3)
        # fig.set_size_inches(8, 6)
        # plt.subplot(121)
        if g in df_cnv.columns:
            # plot_hist_cnv(df_cnv, g, axes[0])
            plot_stacked_hist_cnv(df_cnv, g, axes[0])
        # plt.subplot(122)
        if g in df_mut.columns:
            # plot_hist_mut(df_mut, g, axes[1])
            plot_stacked_hist_mut(df_mut, g, axes[1])

        filename = join(saving_dir, g + '.png')
        fig.suptitle(g, fontdict=dict(family='Arial', weight='bold', fontsize=12))
        plt.savefig(filename, dpi=200)
        plt.close()


if __name__ == "__main__":
    run()
