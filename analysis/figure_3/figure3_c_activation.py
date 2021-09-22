from os.path import dirname, realpath
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from analysis.data_extraction_utils import get_pathway_names
from setup import saving_dir


def plot_high_genes_violinplot(df_in, ax, y, name, saving_dir):
    y = y.reindex(df_in.index)
    df = df_in.copy()
    features = list(df_in.columns)
    df["group"] = y
    df2 = pd.melt(df, id_vars='group', value_vars=features, value_name='value')
    df2['group'] = df2['group'].replace(0, 'Primary')
    df2['group'] = df2['group'].replace(1, 'Metastatic')
    sns.set_color_codes('muted')
    current_palette = sns.color_palette()
    palette = dict(Primary=current_palette[0], Metastatic=current_palette[1])
    sns.violinplot(y="variable", x="value", hue="group", data=df2, split=True, bw=.6, inner=None, palette=palette,
                   linewidth=0., ax=ax)
    # ax.legend(['Primary', 'Metastatic'],  fontsize=12, bbox_to_anchor = (1.0, 1.3))
    ax._legend.remove()
    filename = join(saving_dir, name + '_activation_violinplot.png')
    ax.tick_params(labelsize=16)

    ax.set_ylabel('')
    ax.set_xlabel('Activation', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig(filename)
    plt.close()


def plot_high_genes_pairplot(df_in, y, name, saving_dir):
    y = y.reindex(df_in.index)
    plt.figure(figsize=(10, 10))
    filename = join(saving_dir, name + '_pairplot.png')

    features = list(df_in.columns)
    dd = df_in[features].copy()
    dd["group"] = y
    dd['group'] = dd['group'].replace(0, '0-Primary')
    dd['group'] = dd['group'].replace(1, '1-Metastatic')
    print dd.head()
    g = sns.pairplot(dd, hue="group")

    g.map(corrfunc)
    print 'saving pairplot', filename
    plt.savefig(filename)
    plt.savefig(filename + '.pdf')
    plt.close()


def corrfunc(x, y, **kws):
    #     print kws
    if kws['label'] == '1-Metastatic':
        r, _ = pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(.1, 1.05), xycoords=ax.transAxes)


# if __name__=="__main__":
#     node_activation = pd.read_csv('./extracted/node_importance_graph_adjusted.csv', index_col=0)
#     response = pd.read_csv('./extracted/response.csv', index_col=0)
#     print node_activation.head()
#     print response.head()
#     # i=0
#     for l in range(1, 7):
#         df = pd.read_csv('./extracted/activation_{}.csv'.format(l), index_col=0)
#         df.columns = get_pathway_names(df.columns)
#         print 'layer {} {}'.format(l, df.shape)
#         high_nodes = node_activation[node_activation.layer == l].abs().nlargest(10, columns=['coef_combined'])
#         high_nodes = high_nodes.sort_values('coef_combined', ascending=False)
#         features = list(high_nodes.index)
#         print features
#         to_be_saved =  df[features].copy()
#         plot_high_genes_violinplot(to_be_saved, response, name='l{}'.format(l), saving_dir='./visualizations/activation')
#         to_be_saved =  to_be_saved[features[0:6]]
#         plot_high_genes_pairplot(to_be_saved, response, name='l{}'.format(l), saving_dir='./visualizations/activation')
current_dir = dirname(realpath(__file__))


def plot_activation(ax, l, column='coef_combined', layer=3, pad=200):
    # divider = make_axes_locatable(ax)
    # ax2 = divider.append_axes('left', size='100%', pad=0.6)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # ax2.set_yticks([])
    # ax2.set_xticks([])

    node_activation = pd.read_csv(join(current_dir, 'extracted/node_importance_graph_adjusted.csv'), index_col=0)
    response = pd.read_csv(join(current_dir, 'extracted/response.csv'), index_col=0)
    df = pd.read_csv(join(current_dir, 'extracted/activation_{}.csv'.format(layer)), index_col=0)
    df.columns = get_pathway_names(df.columns)
    if layer == 1:
        column = 'coef_combined'
    # high_nodes = node_activation[node_activation.layer == layer].abs().nlargest(10, columns=[column])
    high_nodes = node_activation[node_activation.layer == layer].abs().nlargest(10, columns=[column])
    high_nodes = high_nodes.sort_values(column, ascending=False)
    features = list(high_nodes.index)
    to_be_saved = df[features].copy()

    y = response.reindex(to_be_saved.index)
    df = to_be_saved.copy()
    features = list(to_be_saved.columns)
    df["group"] = y
    df2 = pd.melt(df, id_vars='group', value_vars=features, value_name='value')
    df2['group'] = df2['group'].replace(0, 'Primary')
    df2['group'] = df2['group'].replace(1, 'Metastatic')

    def short_names(name):
        if len(name) > 55:
            ret = name[:55] + '...'
        else:
            ret = name
        return ret

    df2.variable = df2['variable'].apply(short_names)
    sns.set_color_codes('muted')
    current_palette = sns.color_palette()
    palette = dict(Primary=current_palette[0], Metastatic=current_palette[1])
    sns.violinplot(y="variable", x="value", hue="group", data=df2, split=True, bw=.3, inner=None, palette=palette,
                   linewidth=.5, ax=ax)
    # ax.set_aspect(1)
    ax.autoscale(enable=True, axis='x', tight=True)

    ax.get_legend().remove()
    ax.set_xlim((-1.2, 1.2))
    # ax.legend
    # ax.legend(['Primary', 'Metastatic'], fontsize=8, bbox_to_anchor=(1.0, 1.3))
    # ax.tick_params(axis='y',labelsize=12)
    fontProperties = {'family': 'Arial', 'weight': 'normal', 'size': 10}
    ax.set_yticklabels(ax.get_yticklabels(), fontProperties)

    # ax.set_ylabel('', labelpad=30)
    ax.set_ylabel('')
    ax.set_xlabel('Activation', fontdict=dict(family='Arial', weight='bold', fontsize=14))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('left')

    ax.tick_params(axis="y", direction="out", pad=pad)
    ax.yaxis.set_ticks_position('none')

    ax.set_ylabel('Layer {}'.format(l), fontdict=dict(family='Arial', weight='bold', fontsize=14))

    # plt.tight_layout()
    # ax.subplots_adjust(left=0.35)
    # plot_high_genes_violinplot(to_be_saved, ax, response, name='l{}'.format(layer), saving_dir='./output/activation')
    # to_be_saved = to_be_saved[features[0:6]]
    # plot_high_genes_pairplot(to_be_saved, response, name='l{}'.format(l), saving_dir='./visualizations/activation')


def run():
    left_adjust = [0.1, 0.]
    pad = [50, 250, 200, 250, 200, 200]
    # fig = plt.figure(figsize=(8, 4))
    # fig = plt.figure(figsize=(8, 4 *7))
    for l in range(1, 7):
        fig = plt.figure(figsize=(9, 4))
        # ax = plt.subplot(6, 1,l)
        ax = fig.subplots(1, 1)
        plot_activation(ax, l, column='coef', layer=l, pad=pad[l - 1])
        # plt.subplots_adjust(left=0.5)
        if l == 1:
            shift = 0.3
        else:
            shift = 0.6
        plt.subplots_adjust(left=shift)
        filename = join(saving_dir, 'layer_activation{}.png'.format(l))
        plt.savefig(filename, dpi=200)
        plt.close()


if __name__ == "__main__":
    run()
    # plt.savefig('./output/layer_activation{}.png'.format('all'), dpi=200)
#
