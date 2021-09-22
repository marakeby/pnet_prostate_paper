import os
from os.path import join, exists

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config_path import PROSTATE_DATA_PATH, PLOTS_PATH

# base_dir  = '../../data/prostate/input'
base_dir = join(PROSTATE_DATA_PATH, 'raw_data')
# filename = 'final/P1013_Clinical_Annotations_MAF.txt'
# filename = 'study_view_clinical_data.txt'
filename = '41588_2018_78_MOESM5_ESM.xlsx'
# 'Sample.Type'
# 'Patient.ID'
# 'Mutation_count'
# 'Mutation burden (Mutations per Megabase)'
# 'Fraction of genome altered'

saving_dir = join(PLOTS_PATH, 'reviews')

if not exists(saving_dir):
    os.makedirs(saving_dir)
# df= pd.read_csv(join(base_dir, filename), sep='\t')
df = pd.read_excel(join(base_dir, filename), skiprows=2)

# print(df.head)

# df['type']= df['Sample Type'] == 'Metastasis'
# print df.head()

df_mets = df[df['Sample.Type'] == 'Metastasis'].copy()
df_primary = df[df['Sample.Type'] == 'Primary'].copy()
fig = plt.figure()
fig.set_size_inches(12, 8)
plt.subplot(3, 1, 1)

mutations_mets = df_mets['Mutation_count']
cnv_mets = df_mets['Fraction of genome altered']

mutations_primary = df_primary['Mutation_count']
cnv_primary = df_primary['Fraction of genome altered']

plt.plot(np.log(1 + mutations_mets), cnv_mets, 'r.')
# plt.xlabel('log(1+ # mutations)', fontsize=18)
plt.title('Metastatic')
plt.ylabel('CNA', fontsize=18)
plt.xlim((0, 8))
plt.subplot(3, 1, 2)

plt.plot(np.log(1 + mutations_primary), cnv_primary, 'b.')
# plt.xlabel('log(1+ # mutations)', fontsize=18)
plt.title('Primary')
plt.ylabel('CNA', fontsize=18)
plt.xlim((0, 8))
plt.subplot(3, 1, 3)

plt.scatter(np.log(1 + mutations_mets), cnv_mets, edgecolors='r', facecolors='none')
plt.scatter(np.log(1 + mutations_primary), cnv_primary, edgecolors='b', facecolors='none')
plt.xlim((0, 8))
# plt.title('Primary and Metastatic')

plt.xlabel('log(1+ # mutations)', fontsize=18)
plt.ylabel('CNA', fontsize=18)
plt.legend(['Primary', 'Metastatic'])
filename = join(saving_dir, 'mut_cn.png')
plt.savefig(filename)
# 'Mutation Count' 'Copy Number Alterations'
