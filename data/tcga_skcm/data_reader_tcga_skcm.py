import logging
import pandas as pd
import numpy as np
from os.path import dirname, join
# from sklearn.model_selection import train_test_split
import os

from data.gmt_reader import GMT
# from data.utils.utils import combine

# dir_path = dirname(dirname(__file__))
data_path= '~/DATA'
# print dir_path
input_dir = os.path.join(data_path, 'tcga_skcm')
# input_dir = os.path.join(input_dir, 'input')

def get_sparsity(df):
    nulls = sum(df.isnull().sum())
    sparsity=   nulls*1.0/np.prod(df.shape) * 100.0
    logging.info ('sparsity :  %0.3f %% of X has null values' % sparsity)
    return sparsity

def load_data(filename):
    filename= join(input_dir, filename)
    logging.info('loading data from %s,' %filename)
    data = pd.read_csv(filename, sep='\t', index_col=0)
    del data['Entrez_Gene_Id']
    data=data.T
    data.dropna(axis=0, how='all', inplace=True)
    data.dropna(axis=1, how='all', inplace=True)
    logging.info( data.shape)

    data = data.fillna(data.mean())

    logging.info( 'loaded data %d samples, %d variables'%(data.shape[0] ,data.shape[1]))
    get_sparsity(data)

    # print data.index

    # remove extreme values
    # TDOD: enhance this
    # data[data > 10] = 10
    # data[data < -10] = -10
    # data.set_index('Hugo_Symbol')
    return data

def load_mutation(filename):
    df = pd.read_csv(join(input_dir, filename), sep='\t', low_memory=False)
    # df= df.T
    filtered = df[df['Variant_Classification'] != 'Silent'].copy()

    grouped = filtered.pivot_table(index='Tumor_Sample_Barcode', columns='Hugo_Symbol', values='Start_Position', aggfunc='count')
    grouped = grouped.fillna(0)

    #sanity check
    counts= filtered.groupby('Tumor_Sample_Barcode')['Tumor_Sample_Barcode'].count()
    assert np.sum(counts- grouped.sum(axis=1))==0
    grouped = grouped > 0

    logging.info('loaded mutation data %d samples, %d variables' % (grouped.shape[0], grouped.shape[1]))
    return grouped

def load_mutation_pathway(filename):
    x = load_mutation(filename).T
    d = GMT()
    pathways = d.load_data('c4.all.v6.0.symbols.gmt')
    # pathways.to_csv('pathway.csv')
    print pathways.head()
    print pathways.shape

    all = x.merge(pathways, right_on='gene', left_index=True, how='left')

    print all.head()
    print all.shape
    #
    grouped = all.groupby(['group']).sum()
    grouped = grouped.astype(bool)
    return grouped.T

def load_methylation():
    filename = 'data_methylation_hm450.txt'
    x = load_data(filename)
    # x.set_index('Hugo_Symbol')
    print 'methylation'
    # print x.index
    # print x.head()
    # x= x.astype(bool)
    return x

def load_data_type(data_type= 'ge', selected_genes =None ):
    # filename = None
    x = None
    if data_type =='cnv':
        filename = 'data_CNA.txt'
        x = load_data(filename)
    if data_type =='methylation':
        x = load_methylation()

    if data_type == 'mutation':
        filename = 'data_mutations_extended.txt'
        x = load_mutation(filename)

    if data_type == 'mutation_pathways':
        filename = 'data_mutations_extended.txt'
        x = load_mutation_pathway(filename)

    if data_type =='rppa':
        filename = 'data_rppa_Zscores.txt'
        x = load_data(filename)

    if data_type == 'ge':
        filename = 'data_RNA_Seq_v2_mRNA_median_Zscores.txt'
        x = load_data(filename)

    print 'duplicated x!!'
    print np.sum(x.index.duplicated())

    # if not filename is None:
    #     try:
    #         x = load_data(filename)
    #     except :
    #         logging.error('cannot load data')
    #         raise
    genes = x.columns
    if not selected_genes is None:
        intersect = set.intersection(set(genes), selected_genes)
        if len(intersect) < len(selected_genes):
            # raise Exception('wrong gene')
            logging.warning('some genes dont exist in the original data set')
        x = x.loc[:, intersect]
        genes = intersect
    # logging.info('loaded data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], response.shape[0]))
    logging.info(len(genes))

    return x



# def combine(x_list, y_list, rows_list, cols_lis, data_type_list, complete_features):
#
#     df_list = []
#     for x, y, r, c in zip(x_list, y_list, rows_list, cols_lis):
#         df = pd.DataFrame(x, columns=c, index=r)
#         df['y'] = y
#         df_list.append(df)
#
#     all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1,)
#
#
#     y = all_data[data_type_list[0]]['y']
#
#     all_data = all_data.drop('y', axis=1, level=1)
#
#     print 'all_data head', all_data.head()
#
#     x = all_data.as_matrix()
#     cols = all_data.columns
#     np.savetxt('combined_cols',cols,fmt='%s')
#     rows = all_data.index
#     logging.info( 'After combining, loaded data %d samples, %d variables, %d responses '%(x.shape[0] ,x.shape[1], y.shape[0]))
#
#     return x, y, rows, cols

def get_data(data_type, complete_features, selected_genes):
    if type(data_type) == list:
        x_list = []
        cols = []
        for t in data_type:
            x = load_data_type(t, selected_genes)
            x_list.append(x)
            cols.extend(x.columns.values)


        if complete_features:
            all_cols = np.unique(cols)
            print 'all_cols {}'.format(len(all_cols))
            all_cols_df = pd.DataFrame(index=all_cols)
            for i, x in enumerate(x_list):
                x = x.T.join(all_cols_df, how='outer')
                x = x.T
                x = x.fillna(0)
                x_list[i] = x

        x = pd.concat(x_list, join='inner', axis=1,keys=data_type )
        x = x.swaplevel(i=0, j=1, axis=1)

        # order the columns based on genes
        order = x.columns.levels[0]
        x = x.reindex(columns=order, level=0)

        # x.set_index('Hugo_Symbol')
        # print x.head()
        # x = pd.concat(x_list, keys=data_type, join='inner', axis=1, )

    else:
        x = load_data_type(data_type, selected_genes)

    # print x.index
    # print 'duplicated!!'
    # print np.sum(x.index.duplicated())

    return x

def load_y(output_type):
    y = get_data('mutation', False, None)
    if output_type =='BRAF':
        y = y['BRAF']>0.
    else:
        # y = y.loc[:, ['NRAS', 'BRAF',  'AKT1', 'AKT2', 'AKT3', 'LAMTOR3']]
        y = y.loc[:, ['NRAS', 'BRAF']]
        y = y.sum(axis=1)
        y= y>0.
        y=y.to_frame()
        # print y.head()
        y.columns= ['BRAF']
    print 'duplicated!!'
    print np.sum(y.index.duplicated())
    return y

class SKCMData():
    def __init__(self, data_type='ge', output_type='BRAF', complete_features=False, selected_genes=None):

        # print y.index
        #make sure x and y has the same examples
        # all_data = pd.concat([x,y], keys=['left', 'right'], join='inner', axis=1)

        if not selected_genes is None:
            selected_genes_file= join(data_path, 'genes')
            selected_genes_file= join(selected_genes_file, selected_genes)
            df = pd.read_csv(selected_genes_file, header=0)
            selected_genes = list(df['genes'])

        x = get_data(data_type, complete_features, selected_genes)
        y = load_y(output_type)

        self.columns = x.columns

        # print 'columns', self.columns
        x.columns = x.columns.droplevel()

        # print x.head()

        all = x.join(y, how='inner', rsuffix = '_y')

        print 'all  data'
        # print all.head()
        # print all.index
        # self.x = all_data['left']
        # self.y = all_data['right']

        self.x = all
        self.y =  all['BRAF']
        self.x = self.x.drop('BRAF', axis= 1)

        self.info = np.array(self.x.index)
        # x = self.x.as_matrix()
        # y = self.y.as_matrix()

        # self.x = x
        # self.y = y


        print self.x.head()
        print self.y.head()

        self.x = self.x.as_matrix()
        self.y = self.y.as_matrix().ravel()

        print self.x.shape ,self.y.shape
        # print self.info

    def get_data(self):
            return self.x, self.y, self.info, self.columns

