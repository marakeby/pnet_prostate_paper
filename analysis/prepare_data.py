import sys
from os import makedirs
from os.path import dirname, realpath, exists

current_dir = dirname(dirname(realpath(__file__)))
sys.path.insert(0, dirname(current_dir))
import pandas as pd
import os
from config_path import *
from data_extraction_utils import get_node_importance, get_link_weights_df_, \
    get_data, get_degrees, adjust_coef_with_graph_degree
from utils.loading_utils import DataModelLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

current_dir = dirname(realpath(__file__))

saving_dir = join(current_dir, 'extracted')

if not exists(saving_dir):
    makedirs(saving_dir)


def save_gradient_importance(node_weights_, node_weights_samples_dfs, info):
    for i, k in enumerate(layers[:-1]):
        n = node_weights_[k]
        filename = join(saving_dir, 'gradient_importance_{}.csv'.format(i))
        n.to_csv(filename)

    for i, k in enumerate(layers[:-1]):
        n = node_weights_samples_dfs[k]
        if i > 0:
            n['ind'] = info
            n = n.set_index('ind')
            filename = join(saving_dir, 'gradient_importance_detailed_{}.csv'.format(i))
            n.to_csv(filename)


def save_link_weights(link_weights_df, layers):
    for i, l in enumerate(layers):
        link = link_weights_df[l]
        filename = join(saving_dir, 'link_weights_{}.csv'.format(i))
        link.to_csv(filename)


def save_activation(layer_outs_dict, feature_names, info):
    for l_name, l_outut in sorted(layer_outs_dict.iteritems()):
        if l_name.startswith('h'):
            print l_name, l_outut.shape
            l = int(l_name[1:])
            features = feature_names[l_name]
            layer_output_df = pd.DataFrame(l_outut, index=info, columns=features)
            layer_output_df = layer_output_df.round(decimals=3)
            filename = join(saving_dir, 'activation_{}.csv'.format(l + 1))
            layer_output_df.to_csv(filename)


def save_graph_stats(degrees, fan_outs, fan_ins, layers):
    i = 1

    df = pd.concat([degrees[0], fan_outs[0]], axis=1)
    df.columns = ['degree', 'fan_out']
    df['fan_in'] = 0
    filename = join(saving_dir, 'graph_stats_{}.csv'.format(i))
    df.to_csv(filename)

    for i, (d, fin, fout) in enumerate(zip(degrees[1:], fan_ins, fan_outs[1:])):
        df = pd.concat([d, fin, fout], axis=1)
        df.columns = ['degree', 'fan_in', 'fan_out']
        print df.head()
        filename = join(saving_dir, 'graph_stats_{}.csv'.format(i + 2))
        df.to_csv(filename)


base_dir = join(PROSTATE_LOG_PATH, 'pnet')
model_name = 'onsplit_average_reg_10_tanh_large_testing'

importance_type = ['deepexplain_deeplift']
target = 'o6'
use_data = 'Test'  # {'All', 'Train', 'Test'}
dropAR = False

layers = ['inputs', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'o_linear6']


def run():
    # load model and data --------------------
    model_dir = join(base_dir, model_name)
    model_file = 'P-net_ALL'
    params_file = join(model_dir, model_file + '_params.yml')
    loader = DataModelLoader(params_file)
    nn_model = loader.get_model(model_file)
    feature_names = nn_model.feature_names
    X, Y, info = get_data(loader, use_data, dropAR)
    response = pd.DataFrame(Y, index=info, columns=['response'])
    print response.head()
    filename = join(saving_dir, 'response.csv')
    response.to_csv(filename)
    #
    print 'saving gradeint importance'
    # #gradeint importance --------------------
    node_weights_, node_weights_samples_dfs = get_node_importance(nn_model, X, Y, importance_type[0], target)
    save_gradient_importance(node_weights_, node_weights_samples_dfs, info)
    #
    print 'saving link weights'
    # # link weights --------------------
    link_weights_df = get_link_weights_df_(nn_model.model, feature_names, layers)
    save_link_weights(link_weights_df, layers[1:])
    #
    print 'saving activation'
    # # activation --------------------
    layer_outs_dict = nn_model.get_layer_outputs(X)
    save_activation(layer_outs_dict, feature_names, info)
    #
    print 'saving graph stats'
    # # graph stats --------------------
    stats = get_degrees(link_weights_df, layers[1:])
    import numpy as np
    keys = np.sort(stats.keys())
    for k in keys:
        filename = join(saving_dir, 'graph_stats_{}.csv'.format(k))
        stats[k].to_csv(filename)
    # save_graph_stats(degrees,fan_outs, fan_ins)
    #
    print 'adjust weights with graph stats'
    # # graph stats --------------------
    degrees = []
    for k in keys:
        degrees.append(stats[k].degree.to_frame(name='coef_graph'))

    node_importance = adjust_coef_with_graph_degree(node_weights_, stats, layers[1:-1], saving_dir)
    filename = join(saving_dir, 'node_importance_graph_adjusted.csv')
    node_importance.to_csv(filename)




if __name__ == "__main__":
    run()
