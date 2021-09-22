from os.path import join

import numpy as  np
import pandas as pd
# chr	pos	type
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style='ticks', palette='Set2')
sns.despine()
from config_path import DATA_PATH

input_dir = join(DATA_PATH, 'prostate_paper/raw_data')

# input_dir= '../../data/prostate/input/final/'
# input_dir= '~/DATA/P1000/paper'
# output_dir= './processed'

# filename = 'P1000_final_analysis_set.maf'
filename = '41588_2018_78_MOESM4_ESM.txt'
# df = pd.read_csv(join(input_dir, filename), sep='\t', low_memory=False)
df = pd.read_csv(join(input_dir, filename), sep='\t', low_memory=False, skiprows=1)

print df.columns
bins = range(1, 26)
df2 = df.sort_values('chr')

sizes = pd.read_csv('chromosome_size.txt', sep='\t')
print sizes.shape
# value_counts = np.log(1+df2['chr'].value_counts())
value_counts = df2['chr'].value_counts()
# np.log(1+value_counts]

height = value_counts
bars = value_counts.index
y_pos = np.arange(len(bars))

plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(y_pos, bars)

for spine in plt.gca().spines.values():
    spine.set_visible(False)

# value_counts.plot(kind ='barh')
# value_counts.plot(kind='pie')

print value_counts
plt.xlabel('chromosome number')
plt.ylabel('number of mutatations')

plt.savefig('chr_counts')

fig = plt.figure()
fig.set_size_inches(10.5, 6.5)

print df['type'].value_counts()
# df['type'].value_counts().plot(kind ='bar')
value_counts = np.log(1 + df['type'].value_counts())
height = value_counts
bars = value_counts.index
y_pos = np.arange(len(bars))

fig = plt.gcf()
fig.clf()
ax = plt.subplot(111)

plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))
# plt.bar(y_pos, height, color=(.7, .55, 0, .67))
plt.xticks(y_pos, bars, rotation='vertical')

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.gcf().subplots_adjust(bottom=0.4)
plt.xlabel('Mutation type', fontsize=18)
plt.ylabel('Log (1+mutation count)', fontsize=18)
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
ax.grid(axis='y', color='white', linestyle='-')

# plt.plot(df['type'].value_counts())
plt.savefig('mutation_type_counts')
