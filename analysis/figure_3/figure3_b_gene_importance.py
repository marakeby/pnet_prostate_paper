from setup import saving_dir
import pandas as pd
import os
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns
import numpy as np
from os.path import join, dirname, realpath, exists
from os import makedirs

def plot_high_genes_sns(df, col='avg_score', name='', saving_directory='.'):
    df.index=df.index.map(shorten_names)
    x_pos = range(df.shape[0])
    ax = sns.barplot(y=df.index, x=col, data=df,
                     palette="Blues_d")
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Layer{}'.format(name))
    ax.set_xticks([],[])
    ax.set_yticks(ax.get_yticks(),[])

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
    df2.value= df2.value.abs()
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
    ax = sns.swarmplot(x="variable", y="value", data=df2, hue="group", palette=dict(Primary = current_palette[0], Metastatic = current_palette[1]))
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


def plot_high_genes2(ax, layer=1, graph ='hist', direction='h'):

    if layer==1:
        column = 'coef_combined'
    else:
        column = 'coef'

    node_importance = pd.read_csv('../extracted/node_importance_graph_adjusted.csv', index_col=0)
    high_nodes = node_importance[node_importance.layer == layer].abs().nlargest(10, columns=[column])
    # high_nodes = node_importance[node_importance.layer == layer].abs().nlargest(10, columns=['coef'])
    features = list(high_nodes.index)
    response = pd.read_csv('../extracted/response.csv', index_col=0)
    df_in = pd.read_csv('../extracted/gradient_importance_detailed_{}.csv'.format(layer), index_col=0)
    df_in = df_in.copy()
    df_in = df_in.join(response)
    df_in['group'] = df_in.response
    df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')

    if graph=='hist':
        df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
        bins = np.linspace(df2.value.min(), df2.value.max(), 20)
        g = sns.FacetGrid(df2, col="variable", hue="group", col_wrap=2)
        g.map(plt.hist, 'value', bins=bins, ec="k")
        g.axes[-1].legend(['primary', 'metastatic'])
    elif graph=='viola':
        sns.violinplot(x="variable", y="value", hue="group", data=df2, split=True, bw=.6, inner=None, ax=ax)
        ax.legend(['primary', 'metastatic'])
        fontProperties = dict(family= 'Arial', weight= 'normal', size= 14, rotation=30, ha='right')
        ax.set_xticklabels(ax.get_xticklabels(), fontProperties )
        ax.set_xlabel('')
        # ax.set_ylabel('')
        ax.set_ylabel('Importance Score', fontdict=dict(family= 'Arial',weight='bold', fontsize=14))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    elif graph=='swarm':
        df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
        df2['group'] = df2['group'].replace(0, 'Primary')
        df2['group'] = df2['group'].replace(1, 'Metastatic')
        df2.value = df2.value.abs()

        current_palette = sns.color_palette()
        ax = sns.swarmplot(x="variable", y="value", data=df2, hue="group",
                           palette=dict(Primary=current_palette[0], Metastatic=current_palette[1]),  ax=ax)
        plt.setp(ax.get_legend().get_texts(), fontsize='14')  # for legend text
        fontProperties = dict(family='Arial', weight='normal', size=12, rotation=30, ha='right')
        ax.set_xticklabels(ax.get_xticklabels(), fontProperties)
        # ax.tick_params(labelsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Importance Score', fontdict=dict(family='Arial',weight='bold', fontsize=14))
        ax.legend().set_title('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)


# if __name__ == "__main__":
#     for l in range(1,7):
#         fig = plt.figure(figsize=(6, 4))
#         ax = fig.subplots(1, 1)
#         # plot_high_genes2(ax, layer=l, graph='swarm', direction='h')
#         # plot_high_genes2(ax, layer=l, graph='hist', direction='h')
#         plt.subplots_adjust(bottom=0.15)
#         plt.savefig('./output/importance_{}.png'.format(l), dpi=200)

def shorten_names(name):
    if len(name)>=60:
        name= name[:60]+'...'
    return name

current_dir = dirname(realpath(__file__))

def run():
    node_importance = pd.read_csv(join(current_dir,'extracted/node_importance_graph_adjusted.csv'), index_col=0)
    response = pd.read_csv(join(current_dir, 'extracted/response.csv'), index_col=0)
    print response.head()
    layers = list(node_importance.layer.unique())
    # saving_directory = './output/importance'
    saving_directory =join(saving_dir, 'importance')
    if not exists(saving_directory):
        makedirs(saving_directory)

    for l in layers:
        fig = plt.figure(figsize=(8, 4), dpi=200)
        print l
        if l==1:
            high_nodes = node_importance[node_importance.layer == l].abs().nlargest(10, columns=['coef_combined'])
            plot_high_genes_sns(high_nodes, col='coef_combined', name=str(l), saving_directory=saving_directory)
        else:

            high_nodes = node_importance[node_importance.layer == l].abs().nlargest(10, columns=['coef'])
            plot_high_genes_sns(high_nodes, col='coef', name=str(l), )

        if l ==2:
            shift= 0.7
        else:
            shift=0.6

        plt.gcf().subplots_adjust(left=shift)
        filename = join(saving_directory, str(l) + '_high.png')
        print 'saving', filename
        plt.savefig(filename)
        high_nodes.to_csv(join(saving_directory,'layer_high_{}.csv'.format(l)))

if __name__ == "__main__":
    run()
