from os.path import join
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


base_dir  = '../../data/prostate/input'

# filename = 'final/P1013_Clinical_Annotations_MAF.txt'
filename = 'study_view_clinical_data.txt'


df= pd.read_csv(join(base_dir, filename), sep='\t')
df['type']= df['Sample Type'] == 'Metastasis'
print df.head()

df_mets= df[df['type']]
df_primary = df[~df['type']]
fig = plt.figure()
fig.set_size_inches(12, 8)
plt.subplot(3,1,1)
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 18}
#
# matplotlib.rc('font', **font)


plt.plot(np.log(1+df_mets['Mutation Count']), df_mets['Copy Number Alterations'], 'r.')
# plt.xlabel('log(1+ # mutations)', fontsize=18)
plt.title('Metastatic')
plt.ylabel('CNA', fontsize=18)
plt.xlim((0,8))
plt.subplot(3,1,2)

plt.plot(np.log(1+df_primary['Mutation Count']), df_primary['Copy Number Alterations'], 'b.')
# plt.xlabel('log(1+ # mutations)', fontsize=18)
plt.title('Primary')
plt.ylabel('CNA', fontsize=18)
plt.xlim((0,8))
plt.subplot(3,1,3)

plt.scatter(np.log(1+df_mets['Mutation Count']), df_mets['Copy Number Alterations'], edgecolors ='r', facecolors='none')
plt.scatter(np.log(1+df_primary['Mutation Count']), df_primary['Copy Number Alterations'], edgecolors= 'b', facecolors='none')
plt.xlim((0,8))
# plt.title('Primary and Metastatic')

plt.xlabel('log(1+ # mutations)', fontsize=18)
plt.ylabel('CNA', fontsize=18)
plt.savefig('mut_cn.png')
# 'Mutation Count' 'Copy Number Alterations'
