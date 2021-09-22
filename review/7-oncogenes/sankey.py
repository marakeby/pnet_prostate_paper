from os.path import join
from analysis.vis_utils import get_reactome_pathway_names
from config_path import BASE_PATH
module_path = join(BASE_PATH, 'analysis/figure_3')
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

'''
first layer
'''


def get_first_layer_df(nlargest):
    features_weights = pd.read_csv(join(module_path, './extracted/gradient_importance_0.csv'), index_col=[0, 1])
    features_weights['layer'] = 0
    nodes_per_layer0 = features_weights[['layer']]
    features_weights = features_weights[['coef']]


    all_weights = pd.read_csv(join(module_path, './extracted/node_importance_graph_adjusted.csv'), index_col=0)
    genes_weights = all_weights[all_weights.layer == 1]
    nodes_per_layer1 = genes_weights[['layer']]
    genes_weights = genes_weights[['coef_combined']]
    genes_weights.columns = ['coef']

    nodes_per_layer_df = pd.concat([nodes_per_layer0, nodes_per_layer1])
    print genes_weights.head()
    print 'genes_weights', genes_weights

    node_weights = [features_weights, genes_weights]

    df = get_first_layer(node_weights, number_of_best_nodes=nlargest[0], col_name='coef', include_others=True)
    saving_dir = './'
    df.to_csv(join(saving_dir, 'first_layer.csv'))
    first_layer_df = df[['source', 'target', 'value', 'layer']]
    return first_layer_df


'''
first layer 
df.head()
source	target	value	direction	layer

high_nodes_df
index coef	coef_combined	coef_combined2	coef_combined_zscore	coef_graph	layer	node_id

important_node_connections_df
source	target	layer	value	value_abs	child_sum_target	child_sum_source	value_normalized_by_target	value_normalized_by_source	target_importance	source_importance	A	B	value_final	value_old	source_fan_out	source_fan_out_error	target_fan_in	target_fan_in_error	value_final_corrected
'''


def encode_nodes(df):
    source = df['source']
    target = df['target']
    all_node_labels = list(np.unique(np.concatenate([source, target])))
    n_nodes = len(all_node_labels)
    df_encoded = df.replace(all_node_labels, range(n_nodes))
    return df_encoded, all_node_labels


def get_nlargeest_ind(S):
    ind_source = (S - S.median()).abs() > 2. * S.std()
    ret = min([10, int(sum(ind_source))])
    return ret


def get_pathway_names(all_node_ids):
    pathways_names = get_reactome_pathway_names()
    all_node_labels = pd.Series(all_node_ids).replace(list(pathways_names['id']), list(pathways_names['name']))
    return all_node_labels


def get_nodes_per_layer_filtered(nodes_per_layer_df, all_node_ids, all_node_labels):
    all_node_ids_df = pd.DataFrame(index=all_node_ids)
    nodes_per_layer_filtered_df = nodes_per_layer_df.join(all_node_ids_df, how='right')
    nodes_per_layer_filtered_df = nodes_per_layer_filtered_df.fillna(0)
    nodes_per_layer_filtered_df = nodes_per_layer_filtered_df.groupby(nodes_per_layer_filtered_df.index).min()
    mapping_dict = dict(zip(all_node_ids, all_node_labels))
    nodes_per_layer_filtered_df.index = nodes_per_layer_filtered_df.index.map(lambda x: mapping_dict[x])
    mapping_dict = {y: x for x, y in all_node_labels.to_dict().iteritems()}
    nodes_per_layer_filtered_df.index = nodes_per_layer_filtered_df.index.map(lambda x: mapping_dict[x])
    return nodes_per_layer_filtered_df


features_weights = pd.read_csv(join(module_path, './extracted/gradient_importance_0.csv'), index_col=[0, 1])
features_weights = features_weights.reset_index()
features_weights.columns = ['target', 'source', 'value']
features_weights['layer'] = 0
features_weights.head()


def get_links_with_first_layer():
    '''
        :return: all_links_df: dataframe with all the connections in the model (with first layer)
        '''
    links = []

    link = pd.read_csv(join(module_path, './extracted/gradient_importance_0.csv'), index_col=[0, 1])
    link = link.reset_index()
    link.columns = ['target', 'source', 'value']
    link['layer'] = 0
    links.append(link)
    for l in range(1, 7):
        link = pd.read_csv(join(module_path, './extracted/link_weights_{}.csv'.format(l)), index_col=0)
        link.index.name = 'source'
        link = link.reset_index()
        link_unpivoted = pd.melt(link, id_vars=['source'], var_name='target', value_name='value')
        link_unpivoted['layer'] = l
        link_unpivoted = link_unpivoted[link_unpivoted.value != 0.]
        link_unpivoted = link_unpivoted.drop_duplicates(subset=['source', 'target'])
        links.append(link_unpivoted)
    all_links_df = pd.concat(links, axis=0, sort=True)

    return all_links_df


def get_links():
    '''
    :return: all_links_df: dataframe with all the connections in the model (except first layer)
    '''
    links = []
    for l in range(1, 7):
        link = pd.read_csv(join(module_path, './extracted/link_weights_{}.csv'.format(l)), index_col=0)
        link.index.name = 'source'
        link = link.reset_index()
        link_unpivoted = pd.melt(link, id_vars=['source'], var_name='target', value_name='value')
        link_unpivoted['layer'] = l
        link_unpivoted = link_unpivoted[link_unpivoted.value != 0.]
        link_unpivoted = link_unpivoted.drop_duplicates(subset=['source', 'target'])
        links.append(link_unpivoted)
    all_links_df = pd.concat(links, axis=0)
    return all_links_df


def get_high_nodes(node_importance, nlargest, column):
    '''
    get n largest nodes in each layer
    :param:  node_importance: dataframe with coef_combined  and layer columns
    :return: list of high node names
    '''
    layers = np.sort(node_importance.layer.unique())
    high_nodes = []
    for i, l in enumerate(layers):
        if type(nlargest) == list:
            n = nlargest[i]
        else:
            n = nlargest
        high = list(node_importance[node_importance.layer == l].nlargest(n, columns=column).index)
        high_nodes.extend(high)
    return high_nodes


def filter_nodes(node_importance, high_nodes, add_others=True):
    high_nodes_df = node_importance[node_importance.index.isin(high_nodes)].copy()
    # add others:

    if add_others:
        layers = list(node_importance.layer.unique())
        names = ['others{}'.format(l) for l in layers]
        names = names + ['root']
        layers.append(np.max(layers) + 1)
        print layers
        print names
        data = {'index': names, 'layer': layers}
        df = pd.DataFrame.from_dict(data)
        df = df.set_index('index')
        high_nodes_df = high_nodes_df.append(df)
    return high_nodes_df


def filter_connections(df, high_nodes, add_unk=False):
    """
    :param df: dataframe [ source, target, layer]
    :param high_nodes: list of high node ids
    :param add_unk: boolean , add other connections or not
    :return: ret, dataframe [ source, target, layer] with filtered connection based on the high_nodes
    """
    high_nodes.append('root')

    def apply_others(row):
        # print row
        if not row['source'] in high_nodes_list:
            row['source'] = 'others' + str(row['layer'])

        if not row['target'] in high_nodes_list:
            row['target'] = 'others' + str(row['layer'] + 1)
        return row

    layers_id = np.sort(df.layer.unique())
    high_nodes_list = high_nodes
    layer_dfs = []
    for i, l in enumerate(layers_id):
        layer_df = df[df.layer == l].copy()
        ind1 = layer_df.source.isin(high_nodes_list)
        ind2 = layer_df.target.isin(high_nodes_list)
        if add_unk:
            layer_df = layer_df[ind1 | ind2]
            layer_df = layer_df.apply(apply_others, axis=1)
            # layer_df[~ind1].source = 'others{}'.format(i)
            # layer_df[~ind2].target = 'others{}'.format(i-1)

        else:
            layer_df = layer_df[ind1 & ind2]

        layer_df = layer_df.groupby(['source', 'target']).agg({'value': 'sum', 'layer': 'min'})
        #         layer_df = layer_df.groupby(['source', 'target']).mean()
        #         layer_df = layer_df.drop_duplicates(subset=['source', 'target'])
        layer_dfs.append(layer_df)
    ret = pd.concat(layer_dfs)
    return ret


def get_x_y(df_encoded, layers_nodes):
    '''
    :param df_encoded: datafrme with columns (source  target  value) representing the network
    :param layers_nodes: data frame with index (nodes ) and one columns (layer) representing the layer of the node
    :return: x, y positions onf each node
    '''

    # node_id = range(len(layers_nodes))
    # node_weights = pd.DataFrame([node_id, layers_nodes], columns=['node_id', 'node_name'])
    # print node_weights
    # node weight is the max(sum of input edges, sum of output edges)

    def rescale(val, in_min, in_max, out_min, out_max):
        print val, in_min, in_max, out_min, out_max
        if in_min == in_max:
            return val
        return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))

    source_weights = df_encoded.groupby(by='source')['value'].sum()
    target_weights = df_encoded.groupby(by='target')['value'].sum()

    node_weights = pd.concat([source_weights, target_weights])
    node_weights = node_weights.to_frame()
    node_weights = node_weights.groupby(node_weights.index).max()
    # print layers_nodes
    #     print node_weights

    # print node_weights

    node_weights = node_weights.join(layers_nodes)

    ind = node_weights.index.str.contains('others')

    others_value = node_weights.loc[ind, 'value']
    print 'others_value', others_value
    node_weights.loc[ind, 'value'] = 0.
    node_weights.sort_values(by=['layer', 'value'], ascending=False, inplace=True)
    print 'others_value', others_value
    node_weights.loc[others_value.index, 'value'] = others_value
    n_layers = len(layers_nodes['layer'].unique())
    node_weights['x'] = (node_weights['layer'] - 2) * 0.1 + 0.16
    ind = node_weights.layer == 0
    node_weights.loc[ind, 'x'] = 0.01
    ind = node_weights.layer == 1
    node_weights.loc[ind, 'x'] = 0.08
    ind = node_weights.layer == 2
    node_weights.loc[ind, 'x'] = 0.16
    print 'node_weights', node_weights
    node_weights.to_csv('node_weights.csv')
    node_weights['layer_weight'] = node_weights.groupby('layer')['value'].transform(pd.Series.sum)
    node_weights['y'] = node_weights.groupby('layer')['value'].transform(pd.Series.cumsum)
    node_weights['y'] = (node_weights['y'] - .5 * node_weights['value']) / (1.5 * node_weights['layer_weight'])

    print 'node_weights', node_weights['x'], node_weights['y']
    node_weights.sort_index(inplace=True)
    node_weights.to_csv('xy.csv')
    return node_weights['x'], node_weights['y']


def get_data_trace(linkes, all_node_labels, node_pos, layers, node_colors=None):
    '''
    linkes: dataframe [source, target, value]
    all_node_labels: list of real node names
    node_pos: tuples of (x, y) for neach node
    layers: dataframe with index is node name and columns is layer
    node_colors, list of colors for each node (Same order as all_node_labels )
    '''

    x = node_pos[0]
    y = node_pos[1]
    data_trace = dict(
        type='sankey',
        domain=dict(
            x=[0, 1],
            y=[0, 1]
        ),
        orientation="h",
        valueformat=".0f",
        #         hovertext=all_node_labels,
        node=dict(
            pad=2,
            thickness=30,
            line=dict(
                color="white",
                width=2.
            ),
            label=all_node_labels,

            x=x,
            y=y,
            color=node_colors if node_colors else None,
        ),
        link=dict(
            source=linkes['source'],
            target=linkes['target'],
            value=linkes['value'],
            color=linkes['color']

        )
    )

    layout = dict(
        # title="Neural Network Architecture",
        height=1000,
        width=2000,
        font=dict(
            size=13, family='Arial',
        )

    )
    return data_trace, layout


def get_node_colors(all_node_labels, remove_others=True):
    def color_to_hex(color):
        r, g, b, a = [255 * c for c in color]
        c = '#%02X%02X%02X' % (r, g, b)
        return c

    color_idx = np.linspace(1, 0, len(all_node_labels))
    cmp = plt.cm.Reds
    node_colors = {}
    for i, node in zip(color_idx, all_node_labels):
        if 'other' in node:
            if remove_others:
                c = (255, 255, 255, 0.0)
            else:
                c = (232, 232, 232, 0.5)
        else:
            colors = list(cmp(i))
            colors = [int(255 * c) for c in colors]
            colors[-1] = 0.7  # set alpha

            c = colors
        node_colors[node] = c

    return node_colors


def get_edge_colors(df, node_colors_dict, remove_others=True):
    colors = []
    for i, row in df.iterrows():
        if ('others' in row['source']) or ('others' in row['target']):
            if remove_others:
                colors.append('rgba(255, 255, 255, 0.0)')
            else:
                colors.append('rgba(192, 192, 192, 0.2)')
        else:
            #             colors.append('rgba(252, 186, 3, .4)')
            base_color = [l for l in node_colors_dict[row['source']]]
            base_color[-1] = 0.2
            base_color = 'rgba{}'.format(tuple(base_color))
            colors.append(base_color)
    return colors


def get_node_colors_ordered(high_nodes_df, col_name, remove_others=True):
    '''
    high_nodes_df: data frame with index (node names) andcolumns [layer, coef_combined]
    '''
    node_colors = {}
    layers = high_nodes_df.layer.unique()
    for l in layers:
        nodes_ordered = high_nodes_df[high_nodes_df.layer == l].sort_values(col_name, ascending=False).index
        node_colors.update(get_node_colors(nodes_ordered, remove_others))
    return node_colors


def get_first_layer(node_weights, number_of_best_nodes, col_name='coef', include_others=True):
    gene_weights = node_weights[1].copy()
    feature_weights = node_weights[0].copy()

    gene_weights = gene_weights[[col_name]]
    feature_weights = feature_weights[[col_name]]

    if number_of_best_nodes == 'auto':
        S = gene_weights[col_name].sort_values()
        n = get_nlargeest_ind(S)
        top_genes = list(gene_weights.nlargest(n, col_name).index)
    else:
        top_genes = list(gene_weights.nlargest(number_of_best_nodes, col_name).index)

    # gene normalization
    print 'top_genes', top_genes
    genes = gene_weights.loc[top_genes]
    genes[col_name] = np.log(1. + genes[col_name].abs())

    print genes.shape
    print genes.head()

    df = feature_weights

    if include_others:
        df = df.reset_index()
        df.columns = ['target', 'source', 'value']
        df['target'] = df['target'].map(lambda x: x if x in top_genes else 'others1')
        df = df.groupby(['source', 'target']).sum()

        df = df.reset_index()
        df.columns = ['source', 'target', 'value']

        print 'df groupby'
        print df

    else:
        df = feature_weights.loc[top_genes]
        df = df.reset_index()
        df.columns = ['target', 'source', 'value']

    df['direction'] = df['value'] >= 0.
    df['value'] = abs(df['value'])
    df['source'] = df['source'].replace('mut_important', 'mutation')
    df['source'] = df['source'].replace('cnv', 'copy number')
    df['source'] = df['source'].replace('cnv_amp', 'amplification')
    df['source'] = df['source'].replace('cnv_del', 'deletion')
    df['layer'] = 0

    df['value'] = df['value'] / df.groupby('target')['value'].transform(np.sum)
    df = df[df.value > 0.0]

    # multiply by the gene importance
    # genes
    df = pd.merge(df, genes, left_on='target', right_index=True, how='left')
    print df.shape
    df.coef.fillna(10.0, inplace=True)
    df.value = df.value * df.coef * 150.

    print df.shape

    return df


def get_fromated_network(links, high_nodes_df, col_name, remove_others):
    # get node colors
    node_colors_dict = get_node_colors_ordered(high_nodes_df, col_name, remove_others)

    print 'node_colors_dict', node_colors_dict
    # exception for first layer
    node_colors_dict['amplification'] = (224, 123, 57, 0.7)  # amps
    node_colors_dict['deletion'] = (1, 55, 148, 0.7)  # deletion
    node_colors_dict['mutation'] = (105, 189, 210, 0.7)  # mutation

    # get colors
    links['color'] = get_edge_colors(links, node_colors_dict, remove_others)

    # get node colors
    for key, value in node_colors_dict.items():
        node_colors_dict[key] = 'rgba{}'.format(tuple(value))

    # enocde nodes (convert node names into numbers)
    linkes_filtred_encoded_df, all_node_labels = encode_nodes(links)

    # get node per layers
    node_layers = high_nodes_df[['layer']]

    # remove self connection
    ind = linkes_filtred_encoded_df.source == linkes_filtred_encoded_df.target
    linkes_filtred_encoded_df = linkes_filtred_encoded_df[~ind]

    # make sure we positive values for all edges
    linkes_filtred_encoded_df.value = linkes_filtred_encoded_df.value.abs()
    x, y = get_x_y(links, node_layers)

    # shorten names
    all_node_labels_short = []
    for n in all_node_labels:
        to_be_added = str(n)
        if len(to_be_added) > 30:
            to_be_added = to_be_added[:20] + ' ...'
        if 'others' in to_be_added:
            to_be_added = 'residual'
        if 'root' in to_be_added:
            to_be_added = 'outcome'
        all_node_labels_short.append(to_be_added)

    node_colors_list = []
    for l in all_node_labels:
        node_colors_list.append(node_colors_dict[l])

    pos = (x, y)
    return linkes_filtred_encoded_df, all_node_labels_short, pos, node_layers, node_colors_list


# #remove self connections


def get_MDM4_nodes(links_df):
    import networkx as nx
    net = nx.from_pandas_edgelist(links_df, 'target', 'source', create_using=nx.DiGraph())
    net.name = 'reactome'
    # add root node
    roots = [n for n, d in net.in_degree() if d == 0]
    root_node = 'root'
    edges = [(root_node, n) for n in roots]
    net.add_edges_from(edges)
    # convert to tree
    tree = nx.bfs_tree(net, 'root')

    traces = list(nx.all_simple_paths(tree, 'root', 'MDM4'))
    print len(traces)
    all_odes = []
    for t in traces:
        print '-->'.join(t)
        all_odes.extend(t)

    nodes = np.unique(all_odes)
    return nodes
    # MDM4_subnet = tree.subgraph(nodes)


def run():
    # get reactome pathway ids and names
    reactome_pathway_df = get_reactome_pathway_names()
    id_to_name_dict = dict(zip(reactome_pathway_df.id, reactome_pathway_df.name))
    name_to_id_dict = dict(zip(reactome_pathway_df.name, reactome_pathway_df.id))

    # nlargest= [10, 8, 8, 8, 7, 6]
    nlargest = [10, 10, 10, 10, 6, 6, 6]
    # nlargest= 10

    node_importance = pd.read_csv(join(module_path, './extracted/node_importance_graph_adjusted.csv'), index_col=0)
    #####
    # node_importance.coef_combined = node_importance.coef

    node_id = []
    for x in node_importance.index:
        if x in name_to_id_dict.keys():
            node_id.append(name_to_id_dict[x])
        else:
            node_id.append(x)
    node_importance['node_id'] = node_id

    col_name = 'coef'
    interesting_genes = ['FOXA1', 'SPOP', 'MED12', 'CDK12', 'PIK3CA', 'CHD1', 'ZBTB7B']
    first_layer_nodes = node_importance[node_importance.layer == 1].copy()
    other_layer_nodes = node_importance[node_importance.layer != 1].copy()
    high_nodes_first_layer = get_high_nodes(first_layer_nodes, nlargest=nlargest, column='coef_combined')
    print('high_nodes_first_layer', high_nodes_first_layer)
    high_nodes_pathways = get_high_nodes(other_layer_nodes, nlargest=nlargest, column='coef')
    high_nodes = high_nodes_first_layer + high_nodes_pathways + interesting_genes
    print 'high_nodes', high_nodes
    high_nodes_df = filter_nodes(node_importance, high_nodes)

    high_nodes_ids = list(high_nodes_df.node_id.values)

    links_df = get_links()
    '''
    MDM4
    '''
    mdm4_nodes = get_MDM4_nodes(links_df)
    mdm4_nodes_names = []
    for n in mdm4_nodes:
        if n in id_to_name_dict.keys():
            mdm4_nodes_names.append(id_to_name_dict[n])
        else:
            mdm4_nodes_names.append(n)

    print 'mdm4_nodes', mdm4_nodes_names

    ind = links_df.source == links_df.target
    links_df = links_df[~ind]

    # # keep important nodes only
    links_df = filter_connections(links_df, high_nodes_ids, add_unk=True)

    links_df = links_df.reset_index()

    links_df['value_abs'] = links_df.value.abs()

    links_df['child_sum_target'] = links_df.groupby('target').value_abs.transform(np.sum)
    links_df['child_sum_source'] = links_df.groupby('source').value_abs.transform(np.sum)
    links_df['value_normalized_by_target'] = 100 * links_df.value_abs / links_df.child_sum_target
    links_df['value_normalized_by_source'] = 100 * links_df.value_abs / links_df.child_sum_source

    #
    node_importance['coef_combined_normalized_by_layer'] = 100. * node_importance[col_name] / \
                                                           node_importance.groupby('layer')[col_name].transform(np.sum)

    node_importance_ = node_importance[['node_id', 'coef_combined_normalized_by_layer', col_name]].copy()

    node_importance_['coef_combined_normalized_by_layer'] = np.log(
        1. + node_importance_.coef_combined_normalized_by_layer)
    node_importance_normalized = node_importance_[['node_id', 'coef_combined_normalized_by_layer']]
    node_importance_normalized = node_importance_normalized.set_index('node_id')
    node_importance_normalized.columns = ['target_importance']
    #
    links_df_ = pd.merge(links_df, node_importance_normalized, left_on='target', right_index=True, how='left')
    node_importance_normalized.columns = ['source_importance']
    links_df_ = pd.merge(links_df_, node_importance_normalized, left_on='source', right_index=True, how='left')
    #
    df = links_df_.copy()
    df['A'] = df.value_normalized_by_source * df.source_importance
    df['B'] = df.value_normalized_by_target * df.target_importance
    df['value_final'] = df[["A", "B"]].min(axis=1)
    #
    df['value_old'] = df.value
    df.value = df.value_final
    #
    df['source_fan_out'] = df.groupby('source').value_final.transform(np.sum)
    df['source_fan_out_error'] = np.abs(df.source_fan_out - 100. * df.source_importance)

    df['target_fan_in'] = df.groupby('target').value_final.transform(np.sum)
    df['target_fan_in_error'] = np.abs(df.target_fan_in - 100. * df.target_importance)
    #
    #
    ind = df.source.str.contains('others')
    df['value_final_corrected'] = df.value_final
    df.loc[ind, 'value_final_corrected'] = df[ind].value_final + df[ind].target_fan_in_error
    ind = df.target.str.contains('others')

    df.loc[ind, 'value_final_corrected'] = df[ind].value_final_corrected + df[ind].source_fan_out_error

    df.value = df.value_final_corrected

    important_node_connections_df = df.replace(id_to_name_dict)

    important_node_connections_df.to_csv('important_node_connections_df.csv')
    high_nodes_df.to_csv('high_nodes_df.csv')

    high_nodes_df = high_nodes_df[[col_name, 'layer']]

    # add feature nodes
    high_nodes_df.loc['mutation'] = [1, 0]
    high_nodes_df.loc['amplification'] = [1, 0]
    high_nodes_df.loc['deletion'] = [1, 0]
    high_nodes_df.loc['other1'] = [1, 1]

    # add first layer
    first_layer_df = get_first_layer_df(nlargest)
    links_df = pd.concat([first_layer_df, important_node_connections_df])

    linkes_filtred_, all_node_labels, pos, node_layers, node_colors_list = get_fromated_network(links_df, high_nodes_df,
                                                                                                col_name=col_name,
                                                                                                remove_others=False)
    data_trace, layout = get_data_trace(linkes_filtred_, all_node_labels, pos, node_layers,
                                        node_colors=node_colors_list)
    # #
    fig = dict(data=[data_trace], layout=layout)
    #
    from plotly.offline import plot
    saving_dir = './'
    filename = join(saving_dir, 'sankey.html')
    plot(fig, filename=filename)


if __name__ == "__main__":
    run()
