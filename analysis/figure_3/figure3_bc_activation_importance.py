from os.path import join, dirname, realpath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.data_extraction_utils import get_pathway_names
from setup import saving_dir


def plot_jitter(group_col, val_col, data, ax):
    vals = data.groupby(group_col)[val_col]
    sums = vals.sum().to_frame().sort_values(val_col, ascending=True)
    inds = np.arange(1, len(sums) + 1)
    for i, s in zip(inds, sums.index):
        print i, s
        ind = data[group_col] == s
        n = sum(ind)
        x = data.loc[ind, val_col]
        y = np.array([i - 0.4] * n)
        noise = np.random.normal(0, 0.02, n)
        y = y + noise
        ax.plot(x, y, '.', markersize=1)


def boxplot_csutom(group_col, val_col, data, ax):
    vals = data.groupby(group_col)[val_col]

    sums = vals.sum().to_frame().sort_values(val_col, ascending=True)
    quartile1 = vals.quantile(0.25).reindex(sums.index)
    medians = vals.quantile(0.5).reindex(sums.index)
    quartile3 = vals.quantile(0.75).reindex(sums.index)

    mins = vals.min().reindex(sums.index)
    maxs = vals.max().reindex(sums.index)

    def adjacent_values(mins, maxs, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, maxs)

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, mins, q1)
        return lower_adjacent_value, upper_adjacent_value

    whiskers = np.array([adjacent_values(mi, mx, q1, q3) for mi, mx, q1, q3 in zip(mins, maxs, quartile1, quartile3)])

    inds = np.arange(1, len(medians) + 1)
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.hlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)


def plot_high_genes2(ax, layer=1, graph='hist', direction='h'):
    if layer == 1:
        column = 'coef_combined'
    else:
        column = 'coef'

    node_importance = pd.read_csv(join(current_dir, './extracted/node_importance_graph_adjusted.csv'), index_col=0)
    high_nodes = node_importance[node_importance.layer == layer].abs().nlargest(10, columns=[column])
    features = list(high_nodes.index)
    response = pd.read_csv(join(current_dir, './extracted/response.csv'), index_col=0)
    df_in = pd.read_csv(join(current_dir, './extracted/gradient_importance_detailed_{}.csv').format(layer), index_col=0)
    df_in = df_in.copy()
    df_in = df_in.join(response)
    df_in['group'] = df_in.response
    df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')

    if graph == 'hist':
        df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
        bins = np.linspace(df2.value.min(), df2.value.max(), 20)
        g = sns.FacetGrid(df2, col="variable", hue="group", col_wrap=2)
        g.map(plt.hist, 'value', bins=bins, ec="k")
        g.axes[-1].legend(['primary', 'metastatic'])
    elif graph == 'viola':
        sns.violinplot(x="variable", y="value", hue="group", data=df2, split=True, bw=.6, inner=None, ax=ax)
        ax.legend(['primary', 'metastatic'])
        # fontProperties = dict(family= 'Arial', weight= 'normal', size= 14, rotation=30, ha='right')
        ax.set_xticklabels(ax.get_xticklabels(), fontproperties)
        ax.set_xlabel('')
        ax.set_ylabel('Importance Score', fontdict=dict(family='Arial', weight='bold', fontsize=14))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    elif graph == 'swarm':
        df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
        df2['group'] = df2['group'].replace(0, 'Primary')
        df2['group'] = df2['group'].replace(1, 'Metastatic')
        df2.value = df2.value.abs()

        current_palette = sns.color_palette()
        ax = sns.swarmplot(x="variable", y="value", data=df2, hue="group",
                           palette=dict(Primary=current_palette[0], Metastatic=current_palette[1]), ax=ax)
        plt.setp(ax.get_legend().get_texts(), fontsize='14')  # for legend text

        ax.set_xticklabels(ax.get_xticklabels(), fontproperties)
        ax.set_xlabel('')
        ax.set_ylabel('Importance Score', fontdict=fontproperties)
        ax.legend().set_title('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    elif graph == 'boxplot_custom':
        # fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
        ax1 = ax
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes('right', size='250%', pad=0.2)

        df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
        df2['group'] = df2['group'].replace(0, 'Primary')
        df2['group'] = df2['group'].replace(1, 'Metastatic')
        df2.value = df2.value.abs()
        sums = df2.groupby('variable')['value'].sum().sort_values(ascending=False).to_frame()
        ax1 = sns.barplot(y='variable', x='value', data=sums.reset_index(), palette="Blues_d", ax=ax1)
        ax1.invert_xaxis()
        ax1.set_xscale('log')
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        ax1.set_xticks([], [])
        ax1.set_yticks(ax1.get_yticks(), [])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlabel('Total importance score', labelpad=3, fontdict=fontproperties)
        ax1.spines['left'].set_visible(False)
        ax1.tick_params(left='off', which='both')

        for tick in ax1.yaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")
        ax1.tick_params(axis="y", direction="out", pad=10)
        ax1.yaxis.set_tick_params(labelsize=ticksize)

        df2 = df2[df2.value != 0]
        boxplot_csutom(val_col="value", group_col="variable", data=df2, ax=ax2)
        plot_jitter(val_col="value", group_col="variable", data=df2, ax=ax2)

        ax2.set_ylabel('')
        ax2.set_xlabel('Sample-level importance score', fontproperties, labelpad=labelpad)
        ax2.xaxis.set_tick_params(labelsize=ticksize)
        # ax2.set_xticks([], [])
        ax2.set_xlim(0, 1)
        ax2.set_yticks([], [])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        # ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(axis=u'both', which=u'both', length=0)


def plot_activation(ax, column='coef_combined', layer=3, pad=200, current_dir=''):
    node_activation = pd.read_csv(join(current_dir, 'extracted/node_importance_graph_adjusted.csv'), index_col=0)
    response = pd.read_csv(join(current_dir, 'extracted/response.csv'), index_col=0)
    df = pd.read_csv(join(current_dir, 'extracted/activation_{}.csv'.format(layer)), index_col=0)
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
    ax.set_ylabel('')
    ax.set_xlabel('Activation', fontproperties, labelpad=labelpad)
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
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)

    ax.tick_params(axis=u'both', which=u'both', length=0)


fontproperties = dict(family='Arial', weight='normal', fontsize=6)
current_dir = dirname(realpath(__file__))
ticksize = 5
labelpad = 0
fig = plt.figure(constrained_layout=False, figsize=(7, 1.5))
plt.subplots_adjust(right=0.9999, left=0.03, top=0.99, bottom=0.15)
spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig, width_ratios=[3, 1.5, 2])
ax1 = fig.add_subplot(spec2[0, 0])
ax2 = fig.add_subplot(spec2[0, 2])

plot_high_genes2(ax1, layer=1, graph='boxplot_custom')
plot_activation(ax2, column='coef', layer=3, pad=120, current_dir='')

# shift=0.6
# plt.subplots_adjust(left=shift)
filename = join(saving_dir, 'figure3.png')
plt.savefig(filename, dpi=200)
plt.close()
