import os
from os import makedirs
from os.path import join, dirname, realpath, exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from setup import saving_dir

module_path = dirname(realpath(__file__))


def plot_high_genes_sns(df, col='avg_score', name='', saving_directory='.'):
    df.index = df.index.map(shorten_names)
    x_pos = range(df.shape[0])
    ax = sns.barplot(y=df.index, x=col, data=df,
                     palette="Blues_d")
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Layer{}'.format(name))
    ax.set_xticks([], [])
    ax.set_yticks(ax.get_yticks(), [])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_high_genes(df, col='avg_score', name='', saving_directory='.'):
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    high_pos = df[df[col] > 0]
    high_neg = df[df[col] < 0].sort_values(col, ascending=False)

    high_pos_values = high_pos[col].values
    high_neg_values = high_neg[col].values

    neg_features = list(high_neg[col].index)
    pos_features = list(high_pos[col].index)

    npos = len(high_pos_values)
    nneg = len(high_neg_values)

    ax = plt.subplot(111)
    x_pos = range(npos)
    ax.barh(x_pos, high_pos_values, color='r', align='center')
    x_neg = range(npos + 1, npos + nneg + 1)
    ax.barh(x_neg, high_neg_values, color='b', align='center')

    x = range(npos) + range(npos + 1, npos + nneg + 1)
    plt.yticks(x, pos_features + neg_features)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.get_xaxis().set_ticks([])

    plt.gcf().subplots_adjust(left=0.35)
    filename = join(saving_directory, name + '_high.png')

    print 'saving histogram', filename
    plt.savefig(filename)


def plot_high_genes_histogram(df_in, features, y, name, saving_directory):
    df_in = df_in.copy()
    df_in = df_in.join(y)
    df_in['group'] = df_in.response
    df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
    print df2.head()
    plt.figure()
    bins = np.linspace(df2.value.min(), df2.value.max(), 20)
    g = sns.FacetGrid(df2, col="variable", hue="group", col_wrap=2)
    g.map(plt.hist, 'value', bins=bins, ec="k")
    g.axes[-1].legend(['primary', 'metastatic'])
    filename = join(saving_directory, name + '_importance_histogram.png')

    print 'saving histogram', filename
    plt.savefig(filename)
    plt.close()


def plot_high_genes_violinplot(df_in, features, y, name, saving_directory):
    df_in = df_in.copy()
    df_in = df_in.join(y)
    df_in['group'] = df_in.response
    filename = join(saving_directory, name + '_importance.csv')
    df_tobesaved = df_in[features]
    df_tobesaved.to_csv(filename)
    df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
    # plt.figure()
    plt.figure(figsize=(10, 7))
    ax = sns.violinplot(x="variable", y="value", hue="group", data=df2, split=True, bw=.6, inner=None)
    ax.legend(['primary', 'metastatic'])
    filename = join(saving_directory, name + '_importance_violinplot.png')
    ax.tick_params(labelsize=12)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    # plt.xticks(rotation=45)
    # plt.gcf().subplots_adjust(bottom=0.55)
    plt.tight_layout()
    plt.xlabel('Pathways')
    plt.ylabel('Importance')

    print 'saving violinplot', filename
    plt.savefig(filename)
    plt.close()


def plot_high_genes_swarm(df_in, features, y, name, saving_directory):
    df_in = df_in.copy()
    df_in = df_in.join(y)
    df_in['group'] = df_in.response
    print df_in.head()
    df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
    df2['group'] = df2['group'].replace(0, 'Primary')
    df2['group'] = df2['group'].replace(1, 'Metastatic')
    print df2.head()
    df2.value = df2.value.abs()
    fig, ax = plt.subplots(figsize=(10, 7))
    # df2['color'] =  'rgba(31, 119, 180, 0.7)'
    # df2['color'] =  (0.12, 0.46, 0.7, 0.7)
    # print len(list(df.color))
    # print df2.shape
    # ind = df2['group']=='Metastatic'
    # df2.loc[ind, 'color'] = (1., 0.5,  0.055, 0.7)
    # ax = sns.swarmplot(x="variable", y="value", data=df2, hue="group", color= df2.color.values)
    # muted = ["#4878CF", "#6ACC65", "#D65F5F",
    #          "#B47CC7", "#C4AD66", "#77BEDB"],
    current_palette = sns.color_palette()
    # ax = sns.swarmplot(x="variable", y="value", data=df2, hue="group", palette=dict(Primary = 'b', Metastatic = 'orange'))
    ax = sns.swarmplot(x="variable", y="value", data=df2, hue="group",
                       palette=dict(Primary=current_palette[0], Metastatic=current_palette[1]))
    # ax.legend(['Primary', 'Metastatic'], loc='upper right', fontsize=12)

    plt.setp(ax.get_legend().get_texts(), fontsize='14')  # for legend text
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.tick_params(labelsize=20)
    plt.xlabel('')
    plt.ylabel('Importance Score', fontsize=20)
    filename = join(saving_directory, name + '_importance_swarm.png')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.gcf().subplots_adjust(bottom=0.2)
    print 'saving swarm', filename
    plt.savefig(filename)
    plt.close()


# if __name__=="__main__":
#     node_importance= pd.read_csv('../extracted/node_importance_graph_adjusted.csv', index_col=0)
#     # important_node_connections_df =pd.read_csv('connections_others.csv', index_col=0)
#     # print important_node_connections_df.head()
#     response = pd.read_csv('../extracted/response.csv', index_col=0)
#     print response.head()
#     layers= list(node_importance.layer.unique())
#
#     for l in layers:
#         print l
#         high_nodes = node_importance[node_importance.layer ==l].abs().nlargest(10, columns=['coef_combined'])
#         high_nodes.to_csv('./output/layer_high_{}.csv'.format(l))
#         plot_high_genes(high_nodes, col='coef_combined', name=str(l), saving_directory='./output/importance')
#
#
#
#     layers= np.sort(list(node_importance.layer.unique()))
#     for l in layers[0:]:
#         print l
#         high_nodes = node_importance[node_importance.layer ==l].abs().nlargest(10, columns=['coef_combined'])
#         features= list(high_nodes.index)
#         df = pd.read_csv('../extracted/gradient_importance_detailed_{}.csv'.format(l), index_col=0)
#         y = response
#         # if l==1:
#         #     plot_high_genes_swarm(df, features, y, name=str(l), saving_directory='./output/importance')
#         plot_high_genes_histogram(df, features, y, name=str(l), saving_directory='./output/importance')
#         plot_high_genes_violinplot(df, features, y, name=str(l), saving_directory='./output/importance')
def plot_jitter(group_col, val_col, data, ax):
    vals = data.groupby(group_col)[val_col]
    sums = vals.sum().to_frame().sort_values(val_col, ascending=True)
    inds = np.arange(1, len(sums) + 1)
    for i, s in zip(inds, sums.index):
        print i, s
        ind = data[group_col] == s
        n = sum(ind)
        x = data.loc[ind, val_col]
        y = np.array([i - 0.3] * n)
        noise = np.random.normal(0, 0.02, n)
        y = y + noise
        ax.plot(x, y, '.', markersize=5)


def boxplot_csutom(group_col, val_col, data, ax):
    vals = data.groupby(group_col)[val_col]
    # quartile1 = vals.quantile(0.25)
    # medians = vals.quantile(0.5)
    # quartile3 = vals.quantile(0.75)
    #
    # mins = vals.min()
    # maxs = vals.max()

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

    # xticks = ax.xaxis.get_major_ticks()
    # xticks = ax.xaxis.get_minor_ticks()
    # xticks[0].label1.set_visible(False)
    # xticks[-1].label1.set_visible(False)

    # ax.set_yticks(inds)
    # ax.set_yticklabels(medians.index)


def plot_high_genes2(ax, layer=1, graph='hist', direction='h'):
    if layer == 1:
        column = 'coef_combined'
    else:
        column = 'coef'

    node_importance = pd.read_csv(join(module_path, './extracted/node_importance_graph_adjusted.csv'), index_col=0)
    high_nodes = node_importance[node_importance.layer == layer].abs().nlargest(10, columns=[column])
    # high_nodes = node_importance[node_importance.layer == layer].abs().nlargest(10, columns=['coef'])
    features = list(high_nodes.index)
    response = pd.read_csv(join(module_path, './extracted/response.csv'), index_col=0)
    df_in = pd.read_csv(join(module_path, './extracted/gradient_importance_detailed_{}.csv').format(layer), index_col=0)
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
        fontProperties = dict(family='Arial', weight='normal', size=14, rotation=30, ha='right')
        ax.set_xticklabels(ax.get_xticklabels(), fontProperties)
        ax.set_xlabel('')
        # ax.set_ylabel('')
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
        fontProperties = dict(family='Arial', weight='normal', size=12, rotation=30, ha='right')
        ax.set_xticklabels(ax.get_xticklabels(), fontProperties)
        # ax.tick_params(labelsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Importance Score', fontdict=dict(family='Arial', weight='bold', fontsize=14))
        ax.legend().set_title('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    elif graph == 'boxplot_custom':
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})

        df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
        df2['group'] = df2['group'].replace(0, 'Primary')
        df2['group'] = df2['group'].replace(1, 'Metastatic')
        df2.value = df2.value.abs()

        sums = df2.groupby('variable')['value'].sum().sort_values(ascending=False).to_frame()

        print sums
        ax1 = sns.barplot(y='variable', x='value', data=sums.reset_index(), palette="Blues_d", ax=ax1)
        ax1.invert_xaxis()
        ax1.set_xscale('log')
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        ax1.set_xticks([], [])
        ax1.set_yticks(ax1.get_yticks(), [])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        # ax1.spines['bottom'].set_visible(False)
        ax1.set_xlabel('Total importance score', labelpad=15, fontdict=dict(family='Arial', weight='bold', fontsize=12))
        ax1.spines['left'].set_visible(False)
        # ax1.tick_params(bottom='off', which='both')
        ax1.tick_params(left='off', which='both')

        df2 = df2[df2.value != 0]
        boxplot_csutom(val_col="value", group_col="variable", data=df2, ax=ax2)
        plot_jitter(val_col="value", group_col="variable", data=df2, ax=ax2)

        ax2.set_ylabel('')
        ax2.set_xlabel('Sample-level importance score', fontdict=dict(family='Arial', weight='bold', fontsize=12))

        # ax2.set_xticks([], [])
        ax2.set_xlim(0, 1)
        ax2.set_yticks([], [])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        # ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        #

        # print sums
        # ax2 = ax.twinx()
        #
        # color = 'r'
        # sns.barplot(x='variable', y='value', data=sums.reset_index() , errcolor=".2", edgecolor=".2", facecolor=(1, 1, 1, 0), ax=ax2)
        #
        # ax2.spines['right'].set_color(color)
        # ax2.yaxis.label.set_color(color)
        # ax2.tick_params(axis='y', colors=color)
        # ax2.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # ax2.spines['left'].set_visible(False)
        # ax2.spines['bottom'].set_visible(False)


# if __name__ == "__main__":
#     for l in range(1,7):
#         fig = plt.figure(figsize=(6, 4))
#         ax = fig.subplots(1, 1)
#         # plot_high_genes2(ax, layer=l, graph='swarm', direction='h')
#         # plot_high_genes2(ax, layer=l, graph='hist', direction='h')
#         plt.subplots_adjust(bottom=0.15)
#         plt.savefig('./output/importance_{}.png'.format(l), dpi=200)

def shorten_names(name):
    if len(name) >= 60:
        name = name[:60] + '...'
    return name


def run():
    node_importance = pd.read_csv(join(module_path, 'extracted/node_importance_graph_adjusted.csv'), index_col=0)
    response = pd.read_csv(join(module_path, 'extracted/response.csv'), index_col=0)
    print response.head()
    layers = list(node_importance.layer.unique())
    print layers
    # saving_directory = './output/importance'
    saving_directory = join(saving_dir, 'importance')
    if not exists(saving_directory):
        makedirs(saving_directory)

    plt.close()
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = fig.subplots(1, 1)
    # figure 3b
    plot_high_genes2(ax, layer=1, graph='boxplot_custom')
    filename = join(saving_directory, 'genes_high.png')
    plt.savefig(filename, dpi=200)
    plt.close()

    # layers = []
    for l in layers:
        fig = plt.figure(figsize=(8, 4), dpi=200)
        print l
        if l == 1:
            high_nodes = node_importance[node_importance.layer == l].abs().nlargest(10, columns=['coef_combined'])
            plot_high_genes_sns(high_nodes, col='coef_combined', name=str(l), saving_directory=saving_directory)

        else:
            high_nodes = node_importance[node_importance.layer == l].abs().nlargest(10, columns=['coef'])
            plot_high_genes_sns(high_nodes, col='coef', name=str(l), )

        if l == 2:
            shift = 0.7
        else:
            shift = 0.6

        plt.gcf().subplots_adjust(left=shift)
        filename = join(saving_directory, str(l) + '_high.png')
        print 'saving', filename
        plt.savefig(filename)
        high_nodes.to_csv(join(saving_directory, 'layer_high_{}.csv'.format(l)))


if __name__ == "__main__":
    run()
