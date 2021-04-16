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
grid = {"alpha": [0.0001, 0.001, .009, 0.01, .09, 1, 5, 10], "penalty": ["l1", "l2"], 'loss': ['log'],
        'class_weight': [class_weight]}  # l1 lasso l2 ridge
param_grid_list = list(ParameterGrid(grid))
for i, param in enumerate(param_grid_list):
    logistic_params.append({'loss': 'log', 'type': 'sgd', 'id': 'L2 Logistic Regression_{}'.format(i), 'params': param})
print logistic_params

models = logistic_params

pipeline = {'type': 'crossvalidation', 'params': {'n_splits': 5, 'save_train': True}}
