from data.data_access import Data
import pandas as pd
# selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
# selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes_and_memebr_of_reactome.csv'
selected_samples= 'samples_with_fusion_data.csv'
data_params = {'id': 'ALL', 'type': 'prostate_paper',
             'params': {
                 # 'data_type': ['mut_important', 'cnv_del', 'cnv_amp', 'gene_expression'],
                 # 'data_type': ['mut_important_plus_hotspots'],
                 # 'data_type': ['mut_important'],
                 # 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                 # 'data_type': ['mut_important_plus_hotspots', 'cnv_del', 'cnv_amp'],
                 'data_type': ['mut_hotspots', 'cnv_del', 'cnv_amp'],
                 # 'data_type': ['mut_important', 'cnv_del', 'fusion_genes'],
                 # 'data_type': ['mut_important',  'gene_expression'],
                #  'data_type':  ['CNV_burden', 'TMB'],
                #  'account_for_data_type' : ['fusions'],
                 'account_for_data_type' : None,
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mut_binary': False,
                 'balanced_data': False,
                 'combine_type': 'union',  # intersection
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 # 'selected_samples': selected_samples,
                 'selected_samples': None,
                 'training_split': 0,
             }
             }

data_adapter = Data(**data_params)
x, y, info, columns = data_adapter.get_data()

print x.shape, y.shape, len(columns), len(info)

x_train, x_test, y_train, y_test, info_train, info_test, columns = data_adapter.get_train_test()
x_train_df = pd.DataFrame(x_train, columns = columns, index=info_train )

print columns.levels
print x_train.shape, x_test.shape, y_train.shape, y_test.shape
print x_train.sum().sum()


x,y, info, columns = data_adapter.get_data()
x_df = pd.DataFrame(x, columns = columns, index=info )
print x_df.shape
print x_df.sum().sum()
# print x_train_df['genomics'].shape
# print x_train_df['account_for'].shape