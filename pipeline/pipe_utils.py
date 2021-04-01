import logging
from os import makedirs
from os.path import join, exists

import pandas as pd

abs_ = True
from matplotlib import pyplot as plt
from utils.plots import plot_roc
import numpy as np


def save_coef(fs_model_list, columns, directory, relevant_features):
    coef_df = pd.DataFrame(index=columns)

    dir_name = join(directory, 'fs')
    if not exists(dir_name):
        makedirs(dir_name)

    if hasattr(columns, 'levels'):
        genes_list = columns.levels[0]
    else:
        genes_list = columns

    for model, model_params in fs_model_list:
        print model_params

        model_name = get_model_id(model_params)

        c_ = model.get_coef()
        logging.info('saving coef ')

        model_name_col = model_name

        if hasattr(model, 'get_named_coef'):
            print 'save_feature_importance'
            file_name = join(dir_name, 'coef_' + model_name)
            coef = model.get_named_coef()
            if type(coef) == list:
                for i, c in enumerate(coef):
                    if type(c) == pd.DataFrame:
                        c.to_csv(file_name + str(i) + '.csv')

        if type(c_) == list:
            coef_df[model_name_col] = c_[0]
        else:
            coef_df[model_name_col] = c_

        # special case: multi level coef for pnet
        # if hasattr(model.model, 'coef_') and type(c_) == list:
        #
        #     if 'arch' in model_params['params']['model_params']:
        #         arch = model_params['params']['model_params']['arch']
        #         logging.info('coef len {} '.format(len(c_)))
        #         if isfile(join(data_dir, arch)):
        #             mapp, genes, pathways = get_gene_map(genes_list, arch)
        #             genes_df = pd.DataFrame(c_[1], index=genes, columns=['genes'])
        #
        #             file_name = join(dir_name, 'coef_genes' + model_name + '.csv')
        #             genes_df.to_csv(file_name)
        #
        #             genes_df_sorted = genes_df.sort_values('genes', ascending=False)
        #             file_name = join(dir_name, 'coef_genes' + model_name + '_sorted.csv')
        #             genes_df_sorted.to_csv(file_name)
        #
        #
        #             pathways_df = pd.DataFrame(c_[2], index=pathways, columns=['pathways'])
        #             file_name = join(dir_name, 'coef_pathways' + model_name + '.csv')
        #             pathways_df.to_csv(file_name)
        #
        #             pathways_df_sorted = pathways_df.sort_values('pathways', ascending=False)
        #             file_name = join(dir_name, 'coef_pathways' + model_name + '_sorted.csv')
        #             pathways_df_sorted.to_csv(file_name)
        #
        #
        #         else:
        #             files = get_pathway_files(arch)
        #             # genes = genes_list.levels[0]
        #             genes = genes_list
        #
        #             # if len(files) <= len(c_):
        #             #     n = len(files)
        #             # else:
        #             n = len(c_)-2
        #             # n = min(len(files), len(c_))
        #             print 'n', n
        #             for i, (f, coef_layer) in enumerate(zip (files, c_[1:]) ):
        #
        #                 mapp, genes, pathways = get_gene_map(genes, f)
        #                 print i, f
        #
        #                 print 'coef_layer {} index {}'.format(coef_layer.shape, genes.shape)
        #
        #                 genes_df = pd.DataFrame(coef_layer, index=genes, columns = ['coef'])
        #                 file_name = join(dir_name, 'coef_' + str(i) + model_name + '.csv')
        #                 genes_df.to_csv(file_name)
        #
        #                 #soreted
        #                 genes_df_sorted = genes_df.sort_values('coef', ascending = False)
        #                 file_name = join(dir_name, 'coef_'+ str(i) + model_name + '_sorted.csv')
        #                 genes_df_sorted.to_csv(file_name)
        #                 genes = pathways
        #

    if not relevant_features is None:
        for model, model_name in fs_model_list:
            c = model.get_coef()
            if type(c_) == list:
                c = c_[0]
            else:
                c = c_
            plot_roc(relevant_features, c, dir_name, label=model_name)

    plt.savefig(join(dir_name, 'auc_curves'))
    file_name = join(dir_name, 'coef.csv')
    coef_df.to_csv(file_name)


def report_density(model_list):
    logging.info('model density')

    for model, model_params in model_list:
        model_name = get_model_id(model_params)
        logging.info('' + model_name + ': ' + str(model.get_density()))


def get_model_id(model_params):
    if 'id' in model_params:
        model_name = model_params['id']
    else:
        model_name = model_params['type']
    return model_name


def get_coef(coef_):
    if coef_.ndim == 1:
        coef = np.abs(coef_)
    else:
        coef = np.sum(np.abs(coef_), axis=0)
    return coef


def get_coef_from_model(model):
    coef = None
    if hasattr(model, 'coef_'):
        if type(model.coef_) == list:
            coef = [get_coef(c) for c in model.coef_]
        elif type(model.coef_) == dict:
            coef = [get_coef(model.coef_[c]) for c in model.coef_.keys()]
        else:
            coef = get_coef(model.coef_)

    if hasattr(model, 'scores_'):
        coef = model.scores_

    if hasattr(model, 'feature_importances_'):
        coef = np.abs(model.feature_importances_)
    return coef


# get balanced x and y where the size of postivie samples equal the number of negative samples
def get_balanced(x, y, info):
    print type(x), type(y), type(info)
    pos_ind = np.where(y == 1.)[0]
    neg_ind = np.where(y == 0.)[0]
    n_pos = pos_ind.shape[0]
    n_neg = neg_ind.shape[0]
    n = min(n_pos, n_neg)
    # if debug_:
    print 'n_pos {} n_nge {} n {}'.format(n_pos, n_neg, n)
    # pos_ind = pos_ind[:n]
    # neg_ind = neg_ind[:n]

    pos_ind = np.random.choice(pos_ind, size=n, replace=False)
    neg_ind = np.random.choice(neg_ind, size=n, replace=False)

    print 'pos_ind', pos_ind
    print 'neg_ind', neg_ind
    ind = np.concatenate([pos_ind, neg_ind])
    y = y[ind]
    x = x[ind, :]
    # print type(info), info.head()
    info = info.iloc[ind].copy()
    return x, y, info
