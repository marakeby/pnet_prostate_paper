from os.path import join

import numpy as np
import pandas as pd

from config_path import REACTOM_PATHWAY_PATH
from model.model_utils import get_coef_importance


def get_data(loader, use_data, dropAR):
    x_train, x_test, y_train, y_test, info_train, info_test, columns = loader.get_data()
    if use_data == 'All':
        X = np.concatenate([x_train, x_test], axis=0)
        Y = np.concatenate([y_train, y_test], axis=0)
        info = list(info_train) + list(info_test)
    elif use_data == 'Test':
        X = x_test
        Y = y_test
        info = list(info_test)
    elif use_data == 'Train':
        X = x_train
        Y = y_train
        info = list(info_train)

    if dropAR:
        x = pd.DataFrame(X, columns=columns)
        data_types = x.columns.levels[1].unique()
        print data_types
        if 'cnv' in data_types:
            ind = (x[('AR', 'cnv')] <= 0.) & (x[('AR', 'mut_important')] == 0)
        elif 'cnv_amp' in data_types:
            ind = (x[('AR', 'cnv_amp')] <= 0.) & (x[('AR', 'mut_important')] == 0)

        if len(ind.shape) > 1:
            ind = ind.all(axis=1)
        print ind
        x = x.ix[ind.values,]

        X = x.values
        Y = Y[ind.values]
        info = [info[i] for i in ind.values if i]

        print X.shape, Y.shape, len(info)

        use_data = use_data + '_dropAR'
    print 'shapes'
    print X.shape, Y.shape, len(info)
    return X, Y, info


# returns a dataframe of pathway ids and names
def get_reactome_pathway_names():
    reactome_pathways_df = pd.read_csv(join(REACTOM_PATHWAY_PATH, 'ReactomePathways.txt'), sep='	', header=None)
    reactome_pathways_df.columns = ['id', 'name', 'species']
    reactome_pathways_df_human = reactome_pathways_df[reactome_pathways_df['species'] == 'Homo sapiens']
    reactome_pathways_df_human.reset_index(inplace=True)
    return reactome_pathways_df_human


def get_pathway_names(all_node_ids):
    #     pathways_names = get_reactome_pathway_names()
    #     all_node_labels = pd.Series(all_node_ids).replace(list(pathways_names['id']), list(pathways_names['name']))

    pathways_names = get_reactome_pathway_names()
    ids = list(pathways_names['id'])
    names = list(pathways_names['name'])
    ret_list = []
    for f in all_node_ids:
        # print f
        if f in ids:
            ind = ids.index(f)
            f = names[ind]
            ret_list.append(f)
        else:
            # print 'no'
            ret_list.append(f)

    return ret_list


def get_node_importance(nn_model, x_train, y_train, importance_type, target):
    """

    :param nn_model: nn.Model object
    :param x_train: numpy array
    :param y_train: numpy array/list of outputs
    :param importance_type: ['gradient'|'linear']
    :return: list of pandas dataframes with weights for each layer
    """
    # model = Model(nn_model.model.input, nn_model.model.outputs)
    # model.compile('sgd', 'mse')
    model = nn_model.model
    ret = get_coef_importance(model, x_train, y_train, target=target, feature_importance=importance_type, detailed=True)
    print type(ret)
    if type(ret) is tuple:
        coef, coef_detailed = ret
        print 'coef_detailed', len(coef_detailed)

    else:
        coef = ret
        # empty
        coef_detailed = [c.T for c in coef]

    node_weights_dfs = {}
    node_weights_samples_dfs = {}
    # layers = []
    # for i, (w, w_samples, name) in enumerate(zip(coef, coef_detailed, nn_model.feature_names)):
    for i, k in enumerate(nn_model.feature_names.keys()):
        name = nn_model.feature_names[k]
        w = coef[k]
        w_samples = coef_detailed[k]
        features = get_pathway_names(name)
        df = pd.DataFrame(abs(w.ravel()), index=name, columns=['coef'])
        layer = pd.DataFrame(index=name)
        layer['layer'] = i
        # node_weights_dfs.append(df)
        node_weights_dfs[k] = df
        # layers.append(layer)
        df_samples = pd.DataFrame(w_samples, columns=features)
        # node_weights_samples_dfs.append(df_samples)
        node_weights_samples_dfs[k] = (df_samples)
    return node_weights_dfs, node_weights_samples_dfs


from model.layers_custom import SparseTF
from scipy.sparse import csr_matrix


def get_link_weights_df(link_weights, features):
    link_weights_df = []
    df = pd.DataFrame(link_weights[0], index=features[0])
    link_weights_df.append(df)
    print df.head()
    for i, (rows, cols) in enumerate(zip(features[1:], features[2:])):
        print len(rows), len(cols)
        df = pd.DataFrame(link_weights[i + 1], index=rows, columns=cols)
        link_weights_df.append(df)

    df = pd.DataFrame(link_weights[-1], index=features[-1], columns=['root'])
    link_weights_df.append(df)

    return link_weights_df


def get_layer_weights(layer):
    w = layer.get_weights()[0]
    if type(layer) == SparseTF:
        row_ind = layer.nonzero_ind[:, 0]
        col_ind = layer.nonzero_ind[:, 1]
        w = csr_matrix((w, (row_ind, col_ind)), shape=layer.kernel_shape)
        w = w.todense()
    return w


def get_link_weights_df_(model, features, layer_names):
    # first layer
    # layer_name= layer_names[1]
    # layer= model.get_layer(layer_name)
    link_weights_df = {}
    # df = pd.DataFrame( layer.get_weights()[0], index=features[layer_names[0]])
    # link_weights_df[layer_name]=df

    for i, layer_name in enumerate(layer_names[1:]):
        layer = model.get_layer(layer_name)
        w = get_layer_weights(layer)
        layer_ind = layer_names.index(layer_name)
        previous_layer_name = layer_names[layer_ind - 1]

        print  i, previous_layer_name, layer_name
        if i == 0 or i == (len(layer_names) - 2):
            cols = ['root']
        else:
            cols = features[layer_name]
        rows = features[previous_layer_name]
        w_df = pd.DataFrame(w, index=rows, columns=cols)
        link_weights_df[layer_name] = w_df

    # last layer
    # layer_name = layer_names[-1]
    # layer = model.get_layer(layer_name)
    # link_weights_df = {}
    # df = pd.DataFrame(layer.get_weights()[0], index=features[layer_names[0]])
    # link_weights_df[layer_name] = df

    return link_weights_df


def get_link_weights(model):
    layers = model.layers
    n = len(layers)
    hidden_layers_weights = []
    next = 0
    for i, l in enumerate(layers):
        #         print i, n, l.name
        #         if l.name.startswith('h') or l.name =='o_linear7':
        if l.name.startswith('h') or l.name == 'o_linear6':
            w = l.get_weights()[0]
            if type(l) == SparseTF:
                #                 print l.nonzero_ind
                row_ind = l.nonzero_ind[:, 0]
                col_ind = l.nonzero_ind[:, 1]
                w = csr_matrix((w, (row_ind, col_ind)), shape=l.kernel_shape)
                w = w.todense()
            hidden_layers_weights.append(w)
            print l.name, len(l.get_weights()), w.shape
    return hidden_layers_weights


## Adjust importance
# ------------------------------------------------------------------------------------
def adjust_layer(df):
    # graph coef
    z1 = df.coef_graph
    z1 = (z1 - z1.mean()) / z1.std(ddof=0)

    # gradient coef
    z2 = df.coef
    z2 = (z2 - z2.mean()) / z2.std(ddof=0)

    z = z2 - z1

    z = (z - z.mean()) / z.std(ddof=0)
    x = np.arange(len(z))
    df['coef_combined2'] = z
    return df


def get_degrees(maps, layers):
    stats = {}
    for i, (l1, l2) in enumerate(zip(layers[1:], layers[2:])):

        layer1 = maps[l1]
        layer2 = maps[l2]

        layer1[layer1 != 0] = 1.
        layer2[layer2 != 0] = 1.

        fan_out1 = layer1.abs().sum(axis=1)
        fan_in1 = layer1.abs().sum(axis=0)

        fan_out2 = layer2.abs().sum(axis=1)
        fan_in2 = layer2.abs().sum(axis=0)

        if i == 0:
            print i
            l = layers[0]
            df = pd.concat([fan_out1, fan_out1], keys=['degree', 'fanout'], axis=1)
            df['fanin'] = 1.
            stats[l] = df

        print '{}- layer {} :fan-in {}, fan-out {}'.format(i, l1, fan_in1.shape, fan_out2.shape)
        print '{}- layer {} :fan-in {}, fan-out {}'.format(i, l1, fan_in2.shape, fan_out1.shape)

        df = pd.concat([fan_in1, fan_out2], keys=['fanin', 'fanout'], axis=1)
        df['degree'] = df['fanin'] + df['fanout']
        stats[l1] = df

    return stats


def adjust_coef_with_graph_degree(node_importance_dfs, stats, layer_names, saving_dir):
    ret = []
    # for i, (grad, graph) in enumerate(zip(node_importance_dfs, degrees)):
    for i, l in enumerate(layer_names):
        print l
        grad = node_importance_dfs[l]
        graph = stats[l]['degree'].to_frame(name='coef_graph')

        graph.index = get_pathway_names(graph.index)
        grad.index = get_pathway_names(grad.index)
        d = grad.join(graph, how='inner')

        mean = d.coef_graph.mean()
        std = d.coef_graph.std()
        ind = d.coef_graph > mean + 5 * std
        divide = d.coef_graph.copy()
        divide[~ind] = divide[~ind] = 1.
        d['coef_combined'] = d.coef / divide
        z = d.coef_combined
        z = (z - z.mean()) / z.std(ddof=0)
        d['coef_combined_zscore'] = z
        d = adjust_layer(d)
        #         d['coef_combined'] = d['coef_combined']/sum(d['coef_combined'])
        filename = join(saving_dir, 'layer_{}_graph_adjusted.csv'.format(i))
        d.to_csv(filename)
        d['layer'] = i + 1
        ret.append(d)
    node_importance = pd.concat(ret)
    node_importance = node_importance.groupby(node_importance.index).min()
    return node_importance


# get important connections
# -------------------------------------
def get_nlargeest_ind(S, sigma=2.):
    # ind_source = (S - S.median()).abs() > 3. * S.std()
    ind_source = (S - S.median()).abs() > sigma * S.std()
    ret = int(sum(ind_source))
    return ret


def get_connections(maps, layer_names):
    layers = []
    for i, l in enumerate(layer_names):
        layer = maps[l]
        # layer = layer.unstack().reset_index(name='value')
        layer = layer.unstack().reset_index()
        layer = layer[layer.value != 0]
        layer['layer'] = i + 1
        print layer.head()
        layers.append(layer)
    conn = pd.concat(layers)

    conn['level_0'] = get_pathway_names(conn['level_0'])
    conn['level_1'] = get_pathway_names(conn['level_1'])
    conn.columns = ['target', 'source', 'value', 'layer']
    conn.head()
    return conn


def get_high_nodes(node_importance, sigma=2):
    node_importance_layers = []
    layers_id = np.sort(node_importance.layer.unique())
    for l in layers_id:
        node_importance_layer = node_importance[node_importance.layer == l]
        n = get_nlargeest_ind(node_importance_layer.coef_combined, sigma)
        print l, n
        node_importance_layer = node_importance_layer.coef_combined.abs().nlargest(n)
        node_importance_layers.append(node_importance_layer)

    high_nodes = pd.concat([pd.DataFrame(l) for l in node_importance_layers], keys=layers_id)
    root = pd.DataFrame([1.], columns=['coef_combined'], index=pd.MultiIndex.from_tuples([(-1., 'root')]))
    high_nodes = pd.concat([root, high_nodes])
    return high_nodes


def filter_connections(df, high_nodes, add_unk=False):
    def apply_others(row):
        # print row
        if not row['source'] in high_nodes_list:
            row['source'] = 'others' + str(row['layer'])

        if not row['target'] in high_nodes_list:
            row['target'] = 'others' + str(row['layer'] + 1)
        return row

    layers_id = np.sort(df.layer.unique())
    high_nodes_list = high_nodes.index.levels[1]
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
            layer_df = layer_df.drop_duplicates()
        else:
            layer_df = layer_df[ind1 & ind2]
        layer_dfs.append(layer_df)
    ret = pd.concat(layer_dfs)
    return ret
