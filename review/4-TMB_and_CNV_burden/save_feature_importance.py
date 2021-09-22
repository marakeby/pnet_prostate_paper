import sys
from os import makedirs
from os.path import dirname, realpath, exists

current_dir = dirname(dirname(realpath(__file__)))
sys.path.insert(0, dirname(current_dir))
import pandas as pd
import os
from config_path import *
from analysis.data_extraction_utils import get_node_importance, get_data
from utils.loading_utils import DataModelLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

current_dir = dirname(realpath(__file__))


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


base_dir = join(PROSTATE_LOG_PATH, 'review/cnv_burden_training')

# model_name = 'onsplit_average_reg_10_tanh_large_testing_cnv_burden2'
model_name = 'onsplit_average_reg_10_tanh_large_testing_TMB2'
saving_dir = join(current_dir, model_name)
if not exists(saving_dir):
    makedirs(saving_dir)

importance_type = ['deepexplain_deeplift']
# target = 'o6'
target = 'combined_outcome'
use_data = 'Test'  # {'All', 'Train', 'Test'}
dropAR = False

layers = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'o_linear6']


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

    print 'saving gradeint importance'
    # #gradeint importance --------------------
    node_weights_, node_weights_samples_dfs = get_node_importance(nn_model, X, Y, importance_type[0], target)
    save_gradient_importance(node_weights_, node_weights_samples_dfs, info)


if __name__ == "__main__":
    run()
