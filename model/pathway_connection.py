from os.path import join

import numpy as np
# from data.gmt_reader import GMT
import pandas as pd
from scipy import sparse as sp

from data.data_access import Data
# from data.pathways.pathway_loader import data_dir
from data.gmt_reader import GMT


def get_map(data, gene_dict, pathway_dict):
    genes = data['gene']
    pathways = data['group'].fillna('')

    n_genes = len(gene_dict)
    n_pathways = len(pathway_dict) + 1
    n = data.shape[0]
    row_index = np.zeros((n,))
    col_index = np.zeros((n,))

    for i, g in enumerate(genes):
        row_index[i] = gene_dict[g]

    for i, p in enumerate(pathways):
        # print p, type(p)
        if p == '':
            col_index[i] = n_pathways - 1
        else:
            col_index[i] = pathway_dict[p]

    print n_genes, n_pathways
    print np.max(col_index)
    mapp = sp.coo_matrix(([1] * n, (row_index, col_index)), shape=(n_genes, n_pathways))
    return mapp


def get_dict(listt):
    unique_list = np.unique(listt)
    output_dict = {}
    for i, gene in enumerate(unique_list):
        output_dict[gene] = i
    return output_dict


def get_connection_map(data_params):
    data = Data(**data_params)
    x, y, info, columns = data.get_data()
    x = pd.DataFrame(x.T, index=columns)

    # print x.head()
    # print x.shape
    # print x.index

    d = GMT()
    # pathways = d.load_data ('c4.all.v6.0.entrez.gmt')
    pathways = d.load_data('c4.all.v6.0.symbols.gmt')
    # pathways.to_csv('pathway.csv')

    # print pathways.head()
    # print pathways.shape

    n_genes = len(pathways['gene'].unique())
    n_pathways = len(pathways['group'].unique())
    print 'number of gene {}'.format(n_genes)
    print 'number of pathways {}'.format(n_pathways)
    density = pathways.shape[0] / (n_pathways * n_genes + 0.0)
    print 'density {}'.format(density)

    all = x.merge(pathways, right_on='gene', left_index=True, how='left')

    n_genes = len(all['gene'].unique())
    n_pathways = len(all['group'].unique())
    print 'number of gene {}'.format(n_genes)
    print 'number of pathways {}'.format(n_pathways)
    density = all.shape[0] / (n_pathways * n_genes + 0.0)
    print 'density {}'.format(density)

    # genes = all['gene']
    # pathways = all['group']
    # print all.shape
    gene_dict = get_dict(columns)
    pathway_dict = get_dict(pathways['group'])

    # print gene_dict
    # print pathway_dict

    mapp = get_map(all, gene_dict, pathway_dict)

    return mapp


# return: list of genes, list of pathways, list of input shapes, list of gene pathway memberships
def get_connections(data_params):
    # data_params = {'type': 'prostate', 'params': {'data_type': ['gene_final_cancer', 'cnv_cancer']}}
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print x.shape, y.shape, info.shape, cols.shape
    # print cols

    x_df = pd.DataFrame(x, columns=cols)
    print x_df.head()

    genes = cols.get_level_values(0).unique()
    genes_list = []
    input_shapes = []
    for g in genes:
        g_df = x_df.loc[:, g].as_matrix()
        input_shapes.append(g_df.shape[1])
        genes_list.append(g_df)

    # get pathways
    d = GMT()

    pathways = d.load_data_dict(
        'c2.cp.kegg.v6.1.symbols.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    pathway_genes_map = []
    for p in pathways.keys():
        common_genes = set(genes).intersection(set(pathways[p]))
        indexes = [i for i, e in enumerate(genes) if e in common_genes]
        print len(indexes)
        pathway_genes_map.append(indexes)

    return genes, pathways.keys(), input_shapes, pathway_genes_map

    # data_params = {'type': 'prostate', 'params': {'data_type': 'gene_final'}}


# mapp = get_connection_map(data_params)
# # print mapp
# plt.imshow(mapp)
# plt.savefig('genes_pathways')


def get_input_map(cols):
    index = cols
    col_index = list(index.labels[0])

    # print row_index, col_index
    n_genes = len(np.unique(col_index))
    n_inputs = len(col_index)

    row_index = range(n_inputs)
    n = len(row_index)
    mapp = sp.coo_matrix(([1.] * n, (row_index, col_index)), shape=(n_inputs, n_genes))
    return mapp.toarray()


# params:
# input_list: list of inputs under consideration (e.g. genes)
# filename : a gmt formated file e.g. pathway1 gene1 gene2 gene3
#                                     pathway2 gene4 gene5 gene6
# genes_col: the start index of the gene columns
# shuffle_genes: {True, False}
# return mapp: dataframe with rows =genes and columns = pathways values = 1 or 0 based on the membership of certain gene in the corresponding pathway
def get_layer_map(input_list, filename='c2.cp.kegg.v6.1.symbols.gmt', genes_col=1, shuffle_genes=False):
    d = GMT()
    df = d.load_data(filename, genes_col)
    print 'map # ones  before join {}'.format(df.shape[0])

    df['value'] = 1
    mapp = pd.pivot_table(df, values='value', index='gene', columns='group', aggfunc=np.sum)
    mapp = mapp.fillna(0)
    # print mapp.head()
    print 'map # ones  before join {}'.format(np.sum(mapp.as_matrix()))
    cols_df = pd.DataFrame(index=input_list)
    mapp = cols_df.merge(mapp, right_index=True, left_index=True, how='left')
    mapp = mapp.fillna(0)
    mapp.to_csv(join(data_dir, filename + '.csv'))
    genes = mapp.index
    pathways = mapp.columns
    print 'pathways', pathways
    # print mapp.head()

    mapp = mapp.as_matrix()
    print 'filename', filename
    print 'map # ones  after join {}'.format(np.sum(mapp))

    if shuffle_genes:
        # print mapp[0:10, 0:10]
        # print sum(mapp)
        # logging.info('shuffling the map')
        # mapp = mapp.T
        # np.random.shuffle(mapp)
        # mapp= mapp.T
        # print mapp[0:10, 0:10]
        # print sum(mapp)
        ones_ratio = np.sum(mapp) / np.prod(mapp.shape)

        mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])

    return mapp, genes, pathways


def get_gene_map(input_list, filename='c2.cp.kegg.v6.1.symbols.gmt', genes_col=1, shuffle_genes=False):
    d = GMT()
    # returns a pthway dataframe
    # e.g.  pathway1 gene1
    #       pathway1 gene2
    #       pathway1 gene3
    pathways = d.load_data(filename,
                           genes_col)  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('PathwayCommons9.kegg.hgnc.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('ReactomePathways.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('ReactomePathways_terminal.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('h.all.v6.1.symbols.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('c2.all.v6.1.symbols.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('c2.cp.reactome.v6.1.symbols.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('ReactomePathways_roots.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('PathwayCommons9.All.hgnc.gmt')  # http://www.pathwaycommons.org/archives/PC2/v9/PathwayCommons9.All.hgnc.gmt.gz
    # pathways = d.load_data('c5.bp.v6.1.symbols.gmt')  # Go Biological pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('c4.cm.v6.1.symbols.gmt')  # Cancer modules from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # print 'pathways'
    # print pathways.head()
    print filename
    cols_df = pd.DataFrame(index=input_list)
    # print 'cols_df'
    # print cols_df.shape
    # print cols_df.head()
    # print 'pathways'
    # print pathways.shape
    # print pathways.head()
    print 'map # ones  before join {}'.format(pathways.shape[0])

    # limit the rows to the input_lis only
    all = cols_df.merge(pathways, right_on='gene', left_index=True, how='left')
    # print 'joined'
    # print all.shape
    # print all.head()
    print 'UNK pathway', sum(pd.isnull(all['group']))

    # ind = pd.isnull(all['group'])
    # print ind
    # print 'Known pathway', sum(~pd.isnull(all['group']))

    all = all.fillna('UNK')
    # print 'UNK genes', len(ind), sum(ind)
    # print all.loc[ind, :]
    # all = all.dropna()

    all = all.set_index(['gene', 'group'])
    # print all.head()
    index = all.index

    col_index = list(index.labels[1])
    row_index = list(index.labels[0])

    # print row_index, col_index
    n_pathways = len(np.unique(col_index))
    n_genes = len(np.unique(row_index))

    # row_index = range(n_inputs)
    n = len(row_index)
    # print 'pathways', [index.levels[1][i] for i in col_index]
    # print 'pathways',
    # for p in index.levels[1]:
    #     print p

    mapp = sp.coo_matrix(([1.] * n, (row_index, col_index)), shape=(n_genes, n_pathways))

    pathways = list(index.levels[1])
    genes = index.levels[0]
    mapp = mapp.toarray()

    print 'map # ones  after join {}'.format(np.sum(mapp))

    if shuffle_genes:
        # print mapp[0:10, 0:10]
        # print sum(mapp)
        # logging.info('shuffling the map')
        # mapp = mapp.T
        # np.random.shuffle(mapp)
        # mapp= mapp.T
        # print mapp[0:10, 0:10]
        # print sum(mapp)

        ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
        mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])

    # print pathways

    # if 'UNK' in pathways:
    #     ind= list(pathways).index('UNK')
    #     mapp = np.delete(mapp, ind, 1)
    #     pathways.remove('UNK')

    map_df = pd.DataFrame(mapp, index=genes, columns=pathways)
    map_df.to_csv(join(data_dir, filename + '.csv'))
    return mapp, genes, pathways
    # return map_df
