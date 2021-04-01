import logging
from copy import deepcopy
from os.path import dirname, join
from  config_path import *
import numpy as np
import pandas as pd


data_path= DATA_PATH
processed_path = join(MELANOMA_DATA_PATH , 'processed')

# use this one
# gene_final_no_silent_no_intron = 'P1000_final_analysis_set_cross__no_silent_no_introns_not_from_the_paper.csv'
cnv_filename = 'M1000_data_CNA_paper.csv'
response_filename = 'M1000_clinical.csv'
gene_important_mutations_only = 'M1000_final_analysis_set_cross_important_only.csv'
# gene_truncating_mutations_only = 'P1000_final_analysis_set_cross_truncating_only.csv'
# gene_expression = 'P1000_data_tpm.csv'
# gene_expression = 'P1000_adjusted_TPM.csv'
# fusions_filename = 'p1000_onco_ets_fusions.csv'

cached_data = {}

def load_data(filename, selected_genes=None):
    filename = join(processed_path, filename)
    logging.info('loading data from %s,' % filename)
    if filename in cached_data:
        logging.info('loading from memory cached_data')
        data = cached_data[filename]
    else:
        data = pd.read_csv(filename, index_col=0)
        cached_data[filename] = data
    logging.info(data.shape)

    if 'response' in cached_data:
        logging.info('loading from memory cached_data')
        labels = cached_data['response']
    else:
        labels = get_response()
        cached_data['response'] = labels

    # remove all zeros columns (note the column may be added again later if another feature type belongs to the same gene has non-zero entries).
    # zero_cols = data.sum(axis=0) == 0
    # data = data.loc[:, ~zero_cols]

    # join with the labels
    all = data.join(labels, how='inner')
    all = all[~all['response'].isnull()]

    response = all['response']
    samples = all.index

    del all['response']
    # np.savetxt(filename+'loaded_cols', all.columns, fmt='%s')
    x = all
    genes = all.columns

    if not selected_genes is None:
        intersect = set.intersection(set(genes), selected_genes)
        if len(intersect) < len(selected_genes):
            # raise Exception('wrong gene')
            logging.warning('some genes dont exist in the original data set')
        x = x.loc[:, intersect]
        genes = intersect
    logging.info('loaded data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], response.shape[0]))
    logging.info(len(genes))
    return x, response, samples, genes


def load_TMB(filename):
    x, response, samples, genes = load_data(filename)
    x = np.sum(x, axis=1)
    x = np.array(x)
    x = np.log(1. + x)
    n = x.shape[0]
    response = response.values.reshape((n, 1))
    samples = np.array(samples)
    cols = np.array(['TMB'])
    return x, response, samples, cols


def load_data_type(data_type='gene', cnv_levels=5, cnv_filter_single_event=True, mut_binary=False, selected_genes=None):
    logging.info('loading {}'.format(data_type))
    if data_type == 'TMB':
        x, response, info, genes = load_TMB(gene_important_mutations_only)
    if data_type == 'mut_no_silent_no_intron':
        x, response, info, genes = load_data(gene_final_no_silent_no_intron, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.

    if data_type == 'mut_important':
        x, response, info, genes = load_data(gene_important_mutations_only, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.
    if data_type == 'truncating_mut':
        x, response, info, genes = load_data(gene_truncating_mutations_only, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.
            # x[x<1.]=-1.

    if data_type == 'gene_final_no_silent':
        x, response, info, genes = load_data(gene_final_no_silent, selected_genes)
    if data_type == 'cnv':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        if cnv_levels == 3:
            logging.info('cnv_levels = 3')
            # remove single amplification and single delteion, they are usually noisey
            if cnv_levels == 3:
                if cnv_filter_single_event:
                    x[x == -1.] = 0.0
                    x[x == -2.] = 1.0
                    x[x == 1.] = 0.0
                    x[x == -2.] = 1.0
                else:
                    x[x < 0.] = -1.
                    x[x > 0.] = 1.

    if data_type == 'cnv_del':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x >= 0.0] = 0.
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == -1.] = 0.0
                x[x == -2.] = 1.0
            else:
                x[x < 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == -1.] = 0.5
            x[x == -2.] = 1.0

    if data_type == 'cnv_amp':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x <= 0.0] = 0.
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == 1.0] = 0.0
                x[x == 2.0] = 1.0
            else:
                x[x > 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == 1.] = 0.5
            x[x == 2.] = 1.0

    if data_type == 'cnv_single_del':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == -1.] = 1.0
        x[x != -1.] = 0.0
    if data_type == 'cnv_single_amp':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == 1.] = 1.0
        x[x != 1.] = 0.0
    if data_type == 'cnv_high_amp':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == 2.] = 1.0
        x[x != 2.] = 0.0
    if data_type == 'cnv_deep_del':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == -2.] = 1.0
        x[x != -2.] = 0.0

    if data_type == 'gene_expression':
        x, response, info, genes = load_data(gene_expression, selected_genes)

    if data_type == 'fusions':
        x, response, info, genes = load_data(fusions_filename, None)

    return x, response, info, genes


def get_response():
    logging.info('loading response from %s' % response_filename)
    labels = pd.read_csv(join(processed_path, response_filename),  index_col=0)
    # labels = labels.set_index('id')
    df = labels[['subtype']].copy()
    df.columns= ['response']
    pos_ind = df.response == 'BRAF'
    df.loc[:, 'response'] = 0
    df.loc[pos_ind, 'response'] = 1
    return df


# complete_features: make sure all the data_types have the same set of features_processing (genes)
def combine(x_list, y_list, rows_list, cols_list, data_type_list, combine_type, use_coding_genes_only=False):
    cols_list_set = [set(list(c)) for c in cols_list]

    if combine_type == 'intersection':
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)

    if use_coding_genes_only:
        f = join(data_path, 'genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt')
        coding_genes_df = pd.read_csv(f, sep='\t',header=None)
        coding_genes_df.columns = ['chr', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'].unique())
        cols = cols.intersection(coding_genes)

    # the unique (super) set of genes
    all_cols = list(cols)

    all_cols_df = pd.DataFrame(index=all_cols)

    df_list = []
    for x, y, r, c in zip(x_list, y_list, rows_list, cols_list):
        df = pd.DataFrame(x, columns=c, index=r)
        df = df.T.join(all_cols_df, how='right')
        df = df.T
        df = df.fillna(0)
        df_list.append(df)

    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )

    # put genes on the first level and then the data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)

    x = all_data.values

    reordering_df = pd.DataFrame(index=all_data.index)
    y = reordering_df.join(y, how='left')

    y = y.values
    cols = all_data.columns
    rows = all_data.index
    logging.info(
        'After combining, loaded data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], y.shape[0]))

    return x, y, rows, cols


def split_cnv(x_df):
    genes = x_df.columns.levels[0]
    x_df.rename(columns={'cnv': 'CNA_amplification'}, inplace=True)
    for g in genes:
        x_df[g, 'CNA_deletion'] = x_df[g, 'CNA_amplification'].replace({-1.0: 0.5, -2.0: 1.0})
        x_df[g, 'CNA_amplification'] = x_df[g, 'CNA_amplification'].replace({1.0: 0.5, 2.0: 1.0})
    x_df = x_df.reindex(columns=genes, level=0)
    return x_df


class MelanomaData():

    def __init__(self, data_type='mut', account_for_data_type=None,cnv_levels=5, cnv_filter_single_event=True, mut_binary=False,
                 selected_genes=None, combine_type='intersection', use_coding_genes_only=False, drop_genes=True,
                 balanced_data=False, cnv_split=False, shuffle=False, training_split=0):

        # TMB = False
        self.training_split = training_split
        if not selected_genes is None:
            selected_genes_file= join(data_path, 'genes')
            selected_genes_file= join(selected_genes_file, selected_genes)
            df = pd.read_csv(selected_genes_file, header=0)
            selected_genes = list(df['genes'])

        if type(data_type) == list:
            x_list = []
            y_list = []
            rows_list = []
            cols_list = []
            # if 'TMB' in data_type:
            #     data_type = deepcopy(data_type)
            #     data_type.remove('TMB')
            #     TMB = True
            # else:
            #     TMB = False

            for t in data_type:
                x, y, rows, cols = load_data_type(t, cnv_levels, cnv_filter_single_event, mut_binary, selected_genes)
                x_list.append(x), y_list.append(y), rows_list.append(rows), cols_list.append(cols)
            x, y, rows, cols = combine(x_list, y_list, rows_list, cols_list, data_type, combine_type,
                                       use_coding_genes_only)
            x = pd.DataFrame(x, columns=cols)

        else:
            x, y, rows, cols = load_data_type(data_type, cnv_levels, cnv_filter_single_event, mut_binary,
                                              selected_genes)


        #
        # BRAF mutant, NRAS mutant, NF1 loss and triple wild type
        #

        if drop_genes:
            col= 'BRAF'
            x.loc[:, col] = 0.
            col = 'NRAS'
            x.loc[:, col] = 0.
            col = 'NF1'
            x.loc[:, col] = 0.
            # del x['BRAF']
            # del x['BRAF']
            # del x['BRAF']
            # data_types = x.columns.levels[1].unique()
            # # ind = True
            # if 'cnv' in data_types:
            #     ind = x[('NF1', 'cnv')] <= 0.
            #     x.loc[ind,:] = 0.
            #
            # if 'cnv_del' in data_types:
            #     col=('NF1', 'cnv_del')
            #     ind = x[col] > 0.
            #     x.loc[ind, col] = 0.
            #
            # if 'mut_important' in data_types:
            #     col = ('BRAF', 'mut_important')
            #     # ind = x[col] > 0.
            #     x.loc[:, col] = 0.
            #
            #     col = ('NRAS', 'mut_important')
            #     # ind = x[col] > 0.
            #     x.loc[:, col] = 0.
            #
            #     col = ('NF1', 'mut_important')
            #     x.loc[:, col] = 0.


            # x = x.loc[ind,]
            # y = y[ind]
            # rows = rows[ind]

        if cnv_split:
            x = split_cnv(x)

        if type(x) == pd.DataFrame:
            x = x.values

        if balanced_data:
            pos_ind = np.where(y == 1.)[0]
            neg_ind = np.where(y == 0.)[0]

            n_pos = pos_ind.shape[0]
            n_neg = neg_ind.shape[0]
            n = min(n_pos, n_neg)

            pos_ind = np.random.choice(pos_ind, size=n, replace=False)
            neg_ind = np.random.choice(neg_ind, size=n, replace=False)

            ind = np.sort(np.concatenate([pos_ind, neg_ind]))

            y = y[ind]
            x = x[ind,]
            rows = rows[ind]

        if shuffle:
            n = x.shape[0]
            ind = np.arange(n)
            np.random.shuffle(ind)
            x = x[ind, :]
            y = y[ind, :]
            rows = rows[ind]

        # if TMB:
        #     x_tmb, y_tmb, rows_tmb, cols_tmb = load_data_type('TMB', cnv_levels, cnv_filter_single_event, mut_binary,
        #                                                       selected_genes)
        #     print type(x), type(x_tmb)
        #     x_df = pd.DataFrame(x, index=rows, columns=cols)
        #     n = len(data_type)
        #     columns = [('TMB1', 'TMB2')] * n
        #     x_tmb = np.tile(x_tmb, (n, 1))
        #     x_tmb = pd.DataFrame(x_tmb.T, index=rows_tmb, columns=columns)
        #
        #     joined_df = pd.concat([x_df, x_tmb], axis=1, join='inner')
        #     x = joined_df.values
        #     rows = joined_df.index
        #     cols = joined_df.columns
        if account_for_data_type is not None:
            x_genomics = pd.DataFrame(x, columns = cols, index= rows)
            y_genomics = pd.DataFrame(y,  index= rows, columns=['response'])
            x_list = []
            y_list = []
            rows_list = []
            cols_list = []
            for t in account_for_data_type:
                x_, y_, rows_, cols_ = load_data_type(t, cnv_levels, cnv_filter_single_event, mut_binary, selected_genes)
                x_df = pd.DataFrame(x_, columns=cols_, index=rows_)
                x_list.append(x_df), y_list.append(y_), rows_list.append(rows_), cols_list.append(cols_)

            x_account_for = pd.concat(x_list, keys=account_for_data_type, join='inner', axis=1)
            x_all = pd.concat([x_genomics, x_account_for], keys=['genomics', 'account_for'], join='inner', axis=1)

            common_samples = set(rows).intersection(x_all.index)
            x_all = x_all.loc[common_samples,:]
            y = y_genomics.loc[common_samples,:]

            y = y['response'].values
            x= x_all.values
            cols= x_all.columns
            rows= x_all.index


        self.x = x
        self.y = y
        self.info = rows
        self.columns = cols

    def get_data(self):
        return self.x, self.y, self.info, self.columns

    def get_train_validate_test(self):
        info = self.info
        x = self.x
        y = self.y
        columns = self.columns
        splits_path = join(MELANOMA_DATA_PATH , 'splits')

        training_file = 'training_set_{}.csv'.format(self.training_split)
        training_set = pd.read_csv(join(splits_path, training_file))

        validation_set = pd.read_csv(join(splits_path, 'validation_set.csv'))
        testing_set = pd.read_csv(join(splits_path, 'test_set.csv'))

        info_train = list(set(info).intersection(training_set.id))
        info_validate = list(set(info).intersection(validation_set.id))
        info_test = list(set(info).intersection(testing_set.id))

        ind_train = info.isin(info_train)
        ind_validate = info.isin(info_validate)
        ind_test = info.isin(info_test)

        # ind_train = info.isin(training_set.id)
        # ind_validate = info.isin(validation_set.id)
        # ind_test = info.isin(testing_set.id)

        x_train = x[ind_train]
        x_test = x[ind_test]
        x_validate = x[ind_validate]

        y_train = y[ind_train]
        y_test = y[ind_test]
        y_validate = y[ind_validate]

        info_train = info[ind_train]
        info_test = info[ind_test]
        info_validate = info[ind_validate]

        return x_train, x_validate, x_test, y_train, y_validate, y_test, info_train.copy(), info_validate, info_test.copy(), columns
