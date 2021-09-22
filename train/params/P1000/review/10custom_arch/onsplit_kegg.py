from os.path import join

from config_path import PATHWAY_PATH
from model.builders.prostate_models import build_pnet_KEGG

task = 'classification_binary'
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
data_base = {'id': 'ALL', 'type': 'prostate_paper',
             'params': {
                 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mut_binary': True,
                 'balanced_data': False,
                 'combine_type': 'union',  # intersection
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 0,
             }
             }
data = [data_base]

pre = {'type': None}

nn_pathway = {
    'type': 'nn',
    'id': 'kegg',
    'params':
        {
            'build_fn': build_pnet_KEGG,
            'model_params': {
                'arch': join(PATHWAY_PATH, 'MsigDB/c2.cp.kegg.v6.1.symbols.gmt'),
                'use_bias': True,
                'w_reg': 0.01,
                'dropout': 0.05,
                'optimizer': 'Adam',
                'activation': 'tanh',
                'data_params': data_base,
                'kernel_initializer': 'lecun_uniform',
            }, 'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=False,
                                      monitor='val_o6_f1',
                                      verbose=2,
                                      epoch=100,
                                      shuffle=True,
                                      batch_size=50,
                                      save_name='pnet',
                                      debug=False,
                                      save_gradient=False,
                                      class_weight='auto',
                                      n_outputs=3,
                                      prediction_output='average',
                                      early_stop=False,
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                      lr=0.001,
                                      max_f1=True
                                      ),
            'feature_importance': 'deepexplain_deeplift'
        },
}
features = {}
models = [nn_pathway]

class_weight = {0: 0.75, 1: 1.5}
logistic = {'type': 'sgd', 'id': 'Logistic Regression',
            'params': {'loss': 'log', 'penalty': 'l2', 'alpha': 0.01, 'class_weight': class_weight}}
models.append(logistic)

pipeline = {'type': 'one_split', 'params': {'save_train': True, 'eval_dataset': 'test'}}
