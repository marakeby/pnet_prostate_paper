from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.vis_utils import get_data_trace, get_reactome_pathway_names
from plotly.offline import plot
from config_path import BASE_PATH
module_path = join(BASE_PATH, 'analysis/figure_3')


def get_node_colors(all_node_labels):
    def color_to_hex(color):
        r, g, b, a = [255 * c for c in color]
        c = '#%02X%02X%02X' % (r, g, b)
        return c

    color_idx = np.linspace(1, 0, len(all_node_labels))
    cmp = plt.cm.Reds

    node_colors = {}
    for i, node in zip(color_idx, all_node_labels):
        colors = list(cmp(i))
        colors = [int(255 * c) for c in colors]
        colors[-1] = 0.7  # set alpha
        # c = color_to_hex(c)
        c = 'rgba{}'.format(tuple(colors))
        print c
        node_colors[node] = c

    return node_colors


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


def plot_layers(layers, model_name, suffix):
    layers_df = pd.concat(layers)
    layers_df = layers_df[layers_df['source'] != layers_df['target']]
    layers_df = layers_df.reset_index()
    layers_df = layers_df[layers_df['value'] > 0.0]
    print 'layers_df\n', layers_df.head()
    encoded_top_genes, all_node_ids = encode_nodes(layers_df)
    # rename pathways
    print 'encoded_top_genes\n', encoded_top_genes.head()
    if reactome_labels:
        all_node_labels = get_pathway_names(all_node_ids)
    else:
        all_node_labels = all_node_ids

    nodes_per_layer_filtered = get_nodes_per_layer_filtered(nodes_per_layer_df, all_node_ids, all_node_labels)

    selected_genes_weights = genes_weights.loc[all_node_labels]

    gene_names = list(df.groupby('target').sum().sort_values('value', ascending=False).index)
    node_colors = get_node_colors(gene_names)
    node_colors['amplification'] = 'rgba(224, 123, 57, 0.7)'  # amps
    node_colors['deletion'] = 'rgba(1, 55, 148, 0.7)'  # deletion
    node_colors['mutation'] = 'rgba(105, 189, 210, 0.7)'  # mutation
    node_colors_list = [node_colors[n] for n in all_node_labels]

    print 'node_colors', node_colors_list

    if color_direction:
        encoded_top_genes['color'] = encoded_top_genes['direction'].replace(
            {True: 'rgba(255, 0 0, 0.2)', False: 'rgba(0, 0, 255, 0.2)'})
    else:
        encoded_top_genes['color'] = 'rgba(255, 0 0, 0.2)'

    print encoded_top_genes.head()
    data_trace, layout = get_data_trace(encoded_top_genes, all_node_labels, nodes_per_layer_filtered, node_colors_list)
    fig = dict(data=[data_trace], layout=layout)


    filename = ''
    filename = join(filename, model_name + suffix + '.html')

    plot(fig, filename=filename)


def get_first_layer(node_weights, number_of_best_nodes, interesting_genes=[], col_name='coef', include_others=True):
    # type: (list, int, list, str, bool) -> pd.DataFrame

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

    top_genes = top_genes + interesting_genes
    # gene normalization
    print 'top_genes', top_genes
    genes = gene_weights.loc[top_genes]
    genes[col_name] = np.log(1. + genes[col_name].abs())

    print genes.shape
    print genes.head()

    df = feature_weights

    if include_others:
        df = df.reset_index()
        print 'df'
        print df.head()
        df.columns = ['target', 'source', 'value']
        df['target'] = df['target'].map(lambda x: x if x in top_genes else 'other0')
        df = df.groupby(['source', 'target']).sum()

    else:
        df = feature_weights.loc[top_genes]

    df = df.reset_index()
    df.columns = ['target', 'source', 'value']
    df['direction'] = df['value'] >= 0.
    df['value'] = abs(df['value'])
    # normalize per layer

    df['source'] = df['source'].replace('mut_important', 'mutation')
    df['source'] = df['source'].replace('cnv', 'copy number')
    df['source'] = df['source'].replace('cnv_amp', 'amplification')
    df['source'] = df['source'].replace('cnv_del', 'deletion')
    df['layer'] = 0

    # normalize features by gene
    groups = df.groupby('target')

    sum1 = groups['value'].transform(np.sum)
    df['value'] = df['value'] / sum1
    df = df[df.value > 0.0]

    # multiply by the gene importance
    df = pd.merge(df, genes, left_on='target', right_index=True, how='inner')
    print df.shape
    df.value = df.value * df.coef

    print df.shape

    return df


reactome_labels = True
color_direction = False
features_weights = pd.read_csv(join(module_path, './extracted/gradient_importance_0.csv'), index_col=[0, 1])
features_weights['layer'] = 0
nodes_per_layer0 = features_weights[['layer']]

features_weights = features_weights[['coef']]

print features_weights.head()

all_weights = pd.read_csv(join(module_path, './extracted/node_importance_graph_adjusted.csv'), index_col=0)
genes_weights = all_weights[all_weights.layer == 1]
nodes_per_layer1 = genes_weights[['layer']]
genes_weights = genes_weights[['coef_combined']]
genes_weights.columns = ['coef']

nodes_per_layer_df = pd.concat([nodes_per_layer0, nodes_per_layer1])
print genes_weights.head()
print 'genes_weights', genes_weights

node_weights = [features_weights, genes_weights]
number_of_best_nodes = 1
interesting_genes = ['FOXA1', 'SPOP', 'MED12', 'CDK12', 'PIK3CA', 'CHD1', 'ZBTB7B']
df = get_first_layer(node_weights, number_of_best_nodes, interesting_genes=interesting_genes, col_name='coef',
                     include_others=False)
print df.head()
plot_layers([df], 'first_layer_sankey', '')
