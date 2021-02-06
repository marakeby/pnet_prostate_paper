import os
from os.path import dirname

from sklearn.model_selection import ParameterGrid

base_dirname = dirname(dirname(__file__))
print base_dirname
filename = os.path.basename(__file__)
task = 'classification_binary'

selected_genes = '~/DATA/P1000/tcga_prostate_expressed_genes_and_cancer_genes.csv'
data_base = {'id': 'ALL', 'type': 'prostate_paper',
             'params': {
                 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mut_binary': True,
                 'balanced_data': False,
                 'combine_type': 'union',
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 0,
             }
             }
data = [data_base]

pre = {'type': None}
features = {}

logistic_params = []
print len(logistic_params)
class_weight = {0: 0.75, 1: 1.5}
adaboost_params = []
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
    'class_weight': [class_weight]
}

param_grid_list = list(ParameterGrid(param_grid))
for i, param in enumerate(param_grid_list):
    adaboost_params.append({'type': 'adaboost', 'id': 'Adaptive Boosting_{}'.format(i), 'params': param})

models = adaboost_params

pipeline = {'type': 'one_split', 'params': {'save_train': True, 'eval_dataset': 'validation'}}
