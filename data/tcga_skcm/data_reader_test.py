from data.data_access import Data
import logging
from matplotlib import pyplot as plt
# from data.prostate.data_reader import ProstateData

# data = ProstateData(data_type= 'mut_seq')
#
# x, y, info, cols = data.get_data()
# print x.shape, y.shape, info.shape, cols.shape
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes_and_memebr_of_reactome.csv'


# params= {'type': 'prostate', 'params': {'data_type':'mut_seq' }}
params= {'id': 'ALL','type': 'tcga_skcm', 'params': {'data_type': ['ge', 'methylation'], 'complete_features': False, 'output_type': 'BRAF', 'selected_genes':selected_genes }}
data = Data(**params)

x, y, info, columns = data.get_data()
print x.shape, y.shape

print sum(y)
# print columns

# print [x for x in y.columns if x.startswith('BRAF')]
# plt.hist(y['BRAF'])
# plt.savefig('BRAF.png')
# print x[0,:]

