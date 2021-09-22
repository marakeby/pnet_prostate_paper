from os.path import join

import pandas as pd
from matplotlib import pyplot as plt

# chr	pos	type

input_dir = '../../data/prostate/input/final/'
# output_dir= './processed'
output_dir = '../../data/prostate/processed'

# filename = 'P1000_final_analysis_set.maf'
filename = 'P1013_Clinical_Annotations_MAF.txt'
df = pd.read_csv(join(input_dir, filename), sep='\t', low_memory=False)

id_col = 'cBio.Patient.ID'
df = df.set_index(id_col)
# print df.head()

response_file = 'response_final.csv'
labels = pd.read_csv(join(output_dir, response_file))
labels = labels.set_index('id')
# print labels.head()

all = df.join(labels, how='inner')
print all.head()

mets = all[all['Primary_Met'] == 'Metastasis']
print mets['type'].value_counts()

primary = all[all['Primary_Met'] != 'Metastasis']
print primary['type'].value_counts()

# primary['type'].value_counts().plot(kind ='bar')
fig, ax = plt.subplots(figsize=(15, 7))
# print pd.DataFrame(mets['type'].value_counts(), primary['type'].value_counts())
print all.groupby(['Primary_Met', 'type']).count()['Tumor_Sample_Barcode'].unstack()
all.groupby(['type', 'Primary_Met']).count()['Tumor_Sample_Barcode'].sort('').unstack().plot(kind='bar')

# mets['type'].value_counts().plot(kind ='bar')
plt.gcf().subplots_adjust(bottom=0.4)
# plt.xlabel('mutation type', fontsize = 18)
plt.savefig('test')

# bins = range(1, 26)
# df2 = df.sort_values('chr')
#
# sizes = pd.read_csv('chromosome_size.txt', sep='\t')
# print sizes.shape
# df2['chr'].value_counts().plot(kind ='bar')
#
# print df2['chr'].value_counts()
# plt.xlabel('chromosome')
#
# plt.savefig('chr_counts')
#
# fig =plt.figure()
# fig.set_size_inches(10.5, 6.5)
#
# print df['type'].value_counts()
# df['type'].value_counts().plot(kind ='bar')
# plt.gcf().subplots_adjust(bottom=0.4)
# plt.xlabel('mutation type', fontsize = 18)
# # plt.plot(df['type'].value_counts())
# plt.savefig('mutation_type_counts')
