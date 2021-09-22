from os.path import dirname, realpath
from os.path import join

import matplotlib.pyplot as plt
# from analysis.figure_3.data_extraction_utils import get_pathway_names
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import pearsonr

from analysis.data_extraction_utils import get_pathway_names
from config_path import PLOTS_PATH


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
    if kws['label'] == '1-Metastatic':
        r, _ = pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(.1, 1.05), xycoords=ax.transAxes)


current_dir = dirname(realpath(__file__))


def plot_activation(ax, column='coef_combined', layer=3, pad=200):
    node_activation = pd.read_csv(join(module_path, 'figure_3/extracted/node_importance_graph_adjusted.csv'),
                                  index_col=0)
    response = pd.read_csv(join(module_path, 'figure_3/extracted/response.csv'), index_col=0)
    df = pd.read_csv(join(module_path, 'figure_3/extracted/activation_{}.csv'.format(layer)), index_col=0)
    df.columns = get_pathway_names(df.columns)
    if layer == 1:
        column = 'coef_combined'
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
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.get_legend().remove()
    ax.set_xlim((-1.2, 1.2))
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties)
    # ax.set_xticklabels(ax.get_xticklabels(), fontproperties)
    # ax.set_ylabel('')
    if layer == 6:
        ax.set_xlabel('Activation', fontproperties)
    else:
        ax.set_xlabel('')
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
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.set_ylabel('Layer H{}'.format(layer), fontproperties, labelpad=20)


saving_dir = join(PLOTS_PATH, 'extended_data')
module_path = dirname(dirname(realpath(__file__)))


def shorten_names(name):
    if len(name) >= 60:
        name = name[:60] + ' ...'
    return name


fontsize = 5  # legends, axis
fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}


def plot_axis(axis):
    pad = [150] * 6
    for l in range(1, 7):
        ax = axis[l - 1]
        plot_activation(ax, column='coef', layer=l, pad=pad[l - 1])


def run():
    fig = plt.figure(constrained_layout=False, figsize=(7.2, 9.72))
    spec2 = gridspec.GridSpec(ncols=1, nrows=6, figure=fig)
    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[1, 0])
    ax3 = fig.add_subplot(spec2[2, 0])
    ax4 = fig.add_subplot(spec2[3, 0])
    ax5 = fig.add_subplot(spec2[4, 0])
    ax6 = fig.add_subplot(spec2[5, 0])

    plot_axis([ax1, ax2, ax3, ax4, ax5, ax6])
    fig.tight_layout()

    plt.gcf().subplots_adjust(left=0.5, right=0.8, bottom=0.15)
    filename = join(saving_dir, 'figure_ed7_activation.png')
    plt.savefig(filename, dpi=300)


if __name__ == "__main__":
    run()
# plt.savefig('./output/layer_activation{}.png'.format('all'), dpi=200)
#
