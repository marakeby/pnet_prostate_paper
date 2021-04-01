import pandas as pd

from data.melanoma.data_reader_melanoma import MelanomaData

selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes_and_memebr_of_reactome.csv'

data_params = {'id': 'ALL', 'type': 'melanoma',
             'params': {
                 # 'data_type': ['mut_important', 'cnv_del', 'cnv_amp', 'gene_expression'],
                 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                 # 'data_type': [ 'mut_important'],
                 # 'data_type': ['mut_important',  'gene_expression'],
                #  'data_type':  ['CNV_burden', 'TMB'],
                 'account_for_data_type' : None,
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mut_binary': True,
                 'balanced_data': False,
                 'combine_type': 'intersection',  # intersection
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 0,
             }
             }

data = MelanomaData(**data_params['params'])

x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = data.get_train_validate_test()

print x_train.shape, y_train.shape
print x_validate.shape, y_validate.shape
print x_test.shape, y_test.shape

