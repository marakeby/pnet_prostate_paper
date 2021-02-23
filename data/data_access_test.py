from data.data_access import Data

selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'

data_params = {'id': 'ALL', 'type': 'prostate_paper',
             'params': {
                 # 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                 'data_type': ['mut_important',  'gene_expression'],
                #  'data_type':  ['CNV_burden', 'TMB'],
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

data_adapter = Data(**data_params)
x, y, info, columns = data_adapter.get_data()

print x.shape, y.shape, len(columns), len(info)
print (info)
