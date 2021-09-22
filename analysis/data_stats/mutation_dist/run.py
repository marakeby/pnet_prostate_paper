from os.path import join

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# base_dir  = '../../data/prostate/input'
from config_path import DATA_PATH

base_dir = DATA_PATH
# filename = 'final/P1013_Clinical_Annotations_MAF.txt'
# filename = 'study_view_clinical_data.txt'
filename = 'prostate_paper/processed/P1000_final_analysis_set_cross_important_only.csv'

df = pd.read_csv(join(base_dir, filename), sep='\t')
print df.columns

df['type'] = df['Sample Type'] == 'Metastasis'
print df.head()

df = pd.read_csv(join(base_dir, filename), sep='\t')
df['type'] = df['Sample Type'] == 'Metastasis'
print df.head()

df['Mutation Count'] = df['Mutation Count'].fillna(0)

df_mets = df[df['type']]
df_primary = df[~df['type']]

fig = plt.figure()
fig.set_size_inches(10.5, 6.5)

plt.hist(np.log(1 + df_mets['Mutation Count']), 50, color='r', alpha=0.5)
plt.hist(np.log(1 + df_primary['Mutation Count']), 50, color='b', alpha=0.5)
plt.xlabel('log(1+ #mutations)', fontsize=18)
plt.ylabel('count', fontsize=18)
plt.legend(['Primary', 'Metastatic'])
plt.savefig('mutations')
