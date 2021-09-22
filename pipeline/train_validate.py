import logging
import numpy as np
import pandas as pd
import scipy.sparse
import yaml
from os import makedirs
from os.path import join, exists, dirname, realpath
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from data.data_access import Data
from model.model_factory import get_model
from preprocessing import pre
from utils.evaluate import evalualte
from utils.plots import plot_confusion_matrix
from utils.rnd import set_random_seeds


def plot_2D(x, y, keys, marker='o'):
    classes = np.unique(y)
    for c in classes:
        plt.scatter(x[y == c, 0], x[y == c, 1], marker=marker)
    plt.legend(keys)


def get_validation_primary(cols, cnv_split):
    current_dir = dirname(dirname(realpath(__file__)))
    validation_data_dir = join(current_dir, '_database/prostate/external_validation/')

    valid_cnv = pd.read_csv(join(validation_data_dir, 'PRAD/cnv_matrix.csv'), index_col=0)
    valid_mut = pd.read_csv(join(validation_data_dir, 'PRAD/mut_matrix.csv'), index_col=0)

    genes = cols.get_level_values(0).unique()
    genes_df = pd.DataFrame(index=genes)

    valid_mut_df = genes_df.merge(valid_mut.T, how='left', left_index=True, right_index=True).T
    valid_cnv_df = genes_df.merge(valid_cnv.T, how='left', left_index=True, right_index=True).T

    df_list = [valid_mut_df, valid_cnv_df]
    data_type_list = ['gene_final', 'cnv']

    if cnv_split:
        valid_cnv_ampl = valid_cnv_df.copy()
        valid_cnv_ampl[valid_cnv_ampl <= 0.0] = 0.
        valid_cnv_ampl[valid_cnv_ampl > 0.0] = 1.0

        valid_cnv_del = valid_cnv_df.copy()

        valid_cnv_del[valid_cnv_del >= 0.0] = 0.
        valid_cnv_del[valid_cnv_del < 0.0] = 1.0
        df_list = [valid_mut_df, valid_cnv_del, valid_cnv_ampl]
        data_type_list = ['mut', 'cnv_del', 'cnv_amp']

    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)
    all_data.fillna(0, inplace=True)
    x = all_data.as_matrix()
    cols = all_data.columns
    rows = pd.DataFrame(index=all_data.index)
    y = np.zeros((x.shape[0],))
    return x, y, rows, cols


def get_validation_quigley(cols, cnv_split):
    current_dir = dirname(dirname(realpath(__file__)))
    validation_data_dir = join(current_dir, 'data/prostate_paper/external_validation/')

    valid_cnv = pd.read_csv(join(validation_data_dir, '/Quigley/cnv_design_matrix.csv'), index_col=0)
    valid_mut = pd.read_csv(join(validation_data_dir, 'Quigley/2018_04_15_matrix_rna_tpm.txt'), index_col=0)

    genes = cols.get_level_values(0).unique()
    genes_df = pd.DataFrame(index=genes)

    valid_mut_df = genes_df.merge(valid_mut, how='left', left_index=True, right_index=True).T
    valid_cnv_df = genes_df.merge(valid_cnv, how='left', left_index=True, right_index=True).T

    df_list = [valid_mut_df, valid_cnv_df]
    data_type_list = ['gene_final', 'cnv']
    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)
    all_data.fillna(0, inplace=True)
    x = all_data.as_matrix()
    cols = all_data.columns
    rows = pd.DataFrame(index=all_data.index)
    y = np.ones((x.shape[0],))
    return x, y, rows, cols


def get_validation_metastatic(cols, cnv_split):
    common_samples = ['MO_1008',
                      'MO_1012',
                      'MO_1013',
                      'MO_1014',
                      'MO_1015',
                      'MO_1020',
                      'MO_1040',
                      'MO_1074',
                      'MO_1084',
                      'MO_1094',
                      'MO_1095',
                      'MO_1096',
                      'MO_1114',
                      'MO_1118',
                      'MO_1124',
                      'MO_1128',
                      'MO_1130',
                      'MO_1132',
                      'MO_1139',
                      'MO_1161',
                      'MO_1162',
                      'MO_1176',
                      'MO_1179',
                      'MO_1184',
                      'MO_1192',
                      'MO_1202',
                      'MO_1215',
                      'MO_1219',
                      'MO_1232',
                      'MO_1241',
                      'MO_1244',
                      'MO_1249',
                      'MO_1262',
                      'MO_1277',
                      'MO_1316',
                      'MO_1337',
                      'MO_1339',
                      'MO_1410',
                      'MO_1421',
                      'MO_1447',
                      'MO_1460',
                      'MO_1473',
                      'TP_2001',
                      'TP_2010',
                      'TP_2020',
                      'TP_2032',
                      'TP_2034',
                      'TP_2054',
                      'TP_2060',
                      'TP_2061',
                      'TP_2064',
                      'TP_2069',
                      'TP_2077',
                      'TP_2078',
                      'TP_2079']

    prostate_samples = ['MO_1008', 'MO_1012', 'MO_1013', 'MO_1014', 'MO_1015', 'MO_1020', 'MO_1040', 'MO_1066',
                        'MO_1074', 'MO_1084',
                        'MO_1093', 'MO_1094', 'MO_1095', 'MO_1096', 'MO_1112', 'MO_1114', 'MO_1118', 'MO_1124',
                        'MO_1128', 'MO_1130',
                        'MO_1132', 'MO_1139', 'MO_1161', 'MO_1162', 'MO_1176', 'MO_1179', 'MO_1184', 'MO_1192',
                        'MO_1200', 'MO_1201',
                        'MO_1202', 'MO_1214', 'MO_1215', 'MO_1219', 'MO_1221', 'MO_1232', 'MO_1240', 'MO_1241',
                        'MO_1244', 'MO_1249',
                        'MO_1260', 'MO_1262', 'MO_1263', 'MO_1277', 'MO_1307', 'MO_1316', 'MO_1336', 'MO_1337',
                        'MO_1339', 'MO_1410',
                        'MO_1420', 'MO_1421', 'MO_1437', 'MO_1443', 'MO_1446', 'MO_1447', 'MO_1460', 'MO_1469',
                        'MO_1472', 'MO_1473',
                        'MO_1482', 'MO_1490', 'MO_1492', 'MO_1496', 'MO_1499', 'MO_1510', 'MO_1511', 'MO_1514',
                        'MO_1517', 'MO_1541',
                        'MO_1543', 'MO_1553', 'MO_1556', 'TP_2001', 'TP_2009', 'TP_2010', 'TP_2020', 'TP_2032',
                        'TP_2034', 'TP_2037',
                        'TP_2043', 'TP_2054', 'TP_2060', 'TP_2061', 'TP_2064', 'TP_2069', 'TP_2077', 'TP_2078',
                        'TP_2079', 'TP_2080',
                        'TP_2081', 'TP_2090', 'TP_2093', 'TP_2096', 'TP_2156']

    met500_samples = set(prostate_samples).difference(common_samples)
    common_samples = pd.DataFrame(index=met500_samples)
    current_dir = dirname(dirname(realpath(__file__)))
    validation_data_dir = join(current_dir, '_database/prostate/external_validation/')

    valid_cnv = pd.read_csv(join(validation_data_dir, 'Met500/Met500_cnv.txt'), index_col=0, sep='\t')
    valid_mut = pd.read_csv(join(validation_data_dir, 'Met500/Met500_mut_matrix.csv'), index_col=0)

    valid_cnv = valid_cnv.T
    valid_cnv[valid_cnv > 1.] = 1.
    valid_cnv[valid_cnv < 0.] = -1.
    valid_mut.index = valid_mut.index.str.split('.', 1).str[0]

    genes = cols.get_level_values(0).unique()
    genes_df = pd.DataFrame(index=genes)

    valid_mut_df = common_samples.merge(valid_mut, how='inner', left_index=True, right_index=True)
    valid_cnv_df = common_samples.merge(valid_cnv, how='inner', left_index=True, right_index=True)

    valid_mut_df = genes_df.merge(valid_mut_df.T, how='left', left_index=True, right_index=True).T
    valid_cnv_df = genes_df.merge(valid_cnv_df.T, how='left', left_index=True, right_index=True).T

    df_list = [valid_mut_df, valid_cnv_df]
    data_type_list = ['gene_final', 'cnv']

    if cnv_split:
        valid_cnv_ampl = valid_cnv_df.copy()
        valid_cnv_ampl[valid_cnv_ampl <= 0.0] = 0.
        valid_cnv_ampl[valid_cnv_ampl > 0.0] = 1.0

        valid_cnv_del = valid_cnv_df.copy()

        valid_cnv_del[valid_cnv_del >= 0.0] = 0.
        valid_cnv_del[valid_cnv_del < 0.0] = 1.0
        df_list = [valid_mut_df, valid_cnv_del, valid_cnv_ampl]
        data_type_list = ['mut', 'cnv_del', 'cnv_amp']

    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)
    all_data.fillna(0, inplace=True)

    print 'validation x'
    print all_data.head()
    all_data.to_csv('validatoin_met500.csv')
    x = all_data.as_matrix()

    cols = all_data.columns
    rows = pd.DataFrame(index=all_data.index)
    y = np.ones((x.shape[0],))
    print  'x validation shape', x.shape
    return x, y, rows, cols


class TrainValidatePipeline:
    def __init__(self, data_params, pre_params, feature_params, model_params, pipeline_params, exp_name):

        self.data_params = data_params
        self.pre_params = pre_params
        self.features_params = feature_params
        self.model_params = model_params
        self.exp_name = exp_name
        self.pipeline_params = pipeline_params
        print pipeline_params
        if 'save_train' in pipeline_params['params']:
            self.save_train = pipeline_params['params']['save_train']
        else:
            self.save_train = False
        self.prapre_saving_dir()

    def prapre_saving_dir(self):
        self.directory = self.exp_name
        if not exists(self.directory):
            makedirs(self.directory)

    def save_prediction(self, info, y_pred, y_pred_score, y_test, model_name, training=False):

        if training:
            file_name = join(self.directory, model_name + '_traing.csv')
        else:
            file_name = join(self.directory, model_name + '_testing.csv')
        logging.info("saving : %s" % file_name)
        info['pred'] = y_pred
        info['score'] = y_pred_score
        info['y'] = y_test
        info.to_csv(file_name)

    def get_list(self, x, cols):
        x_df = pd.DataFrame(x, columns=cols)
        print x_df.head()

        genes = cols.get_level_values(0).unique()
        genes_list = []
        input_shapes = []
        for g in genes:
            g_df = x_df.loc[:, g].as_matrix()
            input_shapes.append(g_df.shape[1])
            genes_list.append(g_df)
        return genes_list

    def run(self):
        # logging
        logging.info('loading data....')
        data = Data(**self.data_params[0])
        # will use the whole dataset for training
        x_train, y_train, info_train, cols_train = data.get_data()

        data_types = cols_train.get_level_values(1).unique()
        if len(data_types) > 2:
            cnv_split = True
        else:
            cnv_split = False

        # divide the trainig set into two blanced training sets. Later we will train 2 models and combine their outputs

        index_pos = np.where(y_train == 1)[0]
        index_neg = np.where(y_train == 0)[0]
        n_pos = index_pos.shape[0]
        # select the same number of samples as the positive class
        index_neg1 = index_neg[0:n_pos]

        x_train_pos = x_train[index_pos, :]
        x_train_neg = x_train[index_neg1, :]
        x_train1 = np.concatenate((x_train_pos, x_train_neg))

        y_train_pos = y_train[index_pos, :]
        y_train_neg = y_train[index_neg1, :]
        y_train1 = np.concatenate((y_train_pos, y_train_neg))

        info_train_pos = info_train[index_pos]
        info_train_neg1 = info_train[index_neg1]
        info_train1 = np.concatenate((info_train_pos, info_train_neg1))

        # second training set
        index_neg2 = index_neg[n_pos:]
        x_train_neg2 = x_train[index_neg2, :]
        x_train2 = np.concatenate((x_train_pos, x_train_neg2))

        y_train_neg2 = y_train[index_neg2, :]
        y_train2 = np.concatenate((y_train_pos, y_train_neg2))

        info_train_pos = info_train[index_pos]
        info_train_neg2 = info_train[index_neg2]
        info_train2 = np.concatenate((info_train_pos, info_train_neg2))

        print 'training shape: '
        print x_train1.shape, y_train1.shape, info_train1.shape, cols_train.shape, sum(y_train1)
        print x_train2.shape, y_train2.shape, info_train2.shape, cols_train.shape, sum(y_train2)

        # get testing data set (external validation)
        # 1- Primary data set (write the paper here)
        # 2- Metastatic dataset ()
        # 3- New dataset ()
        x_test_mets, y_test_mets, info_test_mets, cols_test_mets = get_validation_metastatic(cols_train, cnv_split)
        x_test_primary, y_test_primary, info_test_primary, cols_test_primary = get_validation_primary(cols_train,
                                                                                                      cnv_split)
        # x_test_new, y_test_new, info_test_new, cols_test_new = get_validation_primary(cols_train)

        print 'testing shape: '
        print x_test_mets.shape, y_test_mets.shape, info_test_mets.shape, cols_test_mets.shape
        print x_test_primary.shape, y_test_primary.shape, info_test_primary.shape, cols_test_primary.shape
        # print x_test.shape, y_test.shape, info_test.shape, cols_test.shape

        # pre-processing
        logging.info('preprocessing....')
        _, x_test_mets = self.preprocess(x_train, x_test_mets)
        _, x_test_primary = self.preprocess(x_train, x_test_primary)
        _, x_train1 = self.preprocess(x_train, x_train1)
        _, x_train2 = self.preprocess(x_train, x_train2)


        test_scores = []
        # model_names = []
        # model_list = []
        # cnf_matrix_list = []
        fig = plt.figure()
        fig.set_size_inches((10, 6))
        pred_scores = []
        if type(self.model_params) == list:
            for m in self.model_params:
                # get model
                set_random_seeds(random_seed=20080808)

                model1 = get_model(m)
                model2 = get_model(m)
                logging.info('fitting')

                model1 = model1.fit(x_train1, y_train1)
                model2 = model2.fit(x_train2, y_train2)

                logging.info('predicting')

                def predict(x_test, y_test, info_test, model_name, test_set_name):
                    pred = {}
                    y_pred_test2, y_pred_test_scores2 = self.predict(model2, x_test, y_test)
                    y_pred_test1, y_pred_test_scores1 = self.predict(model1, x_test, y_test)

                    y_pred_test_scores = (y_pred_test_scores1 + y_pred_test_scores2) / 2.
                    y_pred_test = y_pred_test_scores > 0.5

                    logging.info('scoring ...')
                    test_score = evalualte(y_test, y_pred_test, y_pred_test_scores)
                    cnf_matrix = confusion_matrix(y_test, y_pred_test)

                    pred['model'] = model_name
                    pred['data_set'] = test_set_name
                    pred = dict(pred, **test_score)
                    pred_scores.append(pred)

                    logging.info('saving results')

                    model_name = model_name + '_' + test_set_name
                    self.save_score(test_score, model_name)
                    self.save_prediction(info_test, y_pred_test, y_pred_test_scores, y_test, model_name)
                    self.save_cnf_matrix([cnf_matrix], [model_name])

                if 'id' in m:
                    model_name = m['id']
                else:
                    model_name = m['type']

                predict(x_test_mets, y_test_mets, info_test_mets, model_name, '_mets')
                predict(x_test_primary, y_test_primary, info_test_primary, model_name, '_primary')

                pred_scores_df = pd.DataFrame(pred_scores)
                pred_scores_df.to_csv(join(self.directory, 'testing_scores.csv'))

        return test_scores

    def save_layer_outputs(self, x_train_layer_outputs, y_train, y_train_pred, x_test_layer_outputs, y_test):
        fig = plt.figure(1, figsize=(10, 9))
        for i, (x_train, x_test) in enumerate(zip(x_train_layer_outputs[:-2], x_test_layer_outputs[:-2])):
            print x_train[0].shape

            pca = decomposition.PCA(n_components=50)
            # pca.fit(xx)
            X_embedded_train = pca.fit_transform(x_train[0])
            X_embedded_test = pca.transform(x_test[0])

            X_embedded = np.concatenate((X_embedded_train, X_embedded_test))
            tsne = TSNE(n_components=2)
            X_embedded = tsne.fit_transform(X_embedded)
            n = X_embedded_train.shape[0]
            X_embedded_train = X_embedded[0:n, :]
            X_embedded_test = X_embedded[n:, :]

            # print X_embedded.shape, y.shape
            plt.figure(figsize=(10, 9))
            plot_2D(X_embedded_train, y_train[:, 0], ['Primary', 'Metastatic'], 'o')
            plot_2D(X_embedded_test, y_test[:, 0], ['Primary', 'Metastatic'], 'X')

            # https://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data
            # create meshgrid
            resolution = 100  # 100x100 background pixels
            X2d_xmin, X2d_xmax = np.min(X_embedded_train[:, 0]), np.max(X_embedded_train[:, 0])
            X2d_ymin, X2d_ymax = np.min(X_embedded_train[:, 1]), np.max(X_embedded_train[:, 1])
            xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution),
                                 np.linspace(X2d_ymin, X2d_ymax, resolution))

            # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
            background_model = KNeighborsClassifier(n_neighbors=1).fit(X_embedded_train, y_train_pred)
            voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
            voronoiBackground = voronoiBackground.reshape((resolution, resolution))

            # plot
            cmap = plt.get_cmap('jet')
            plt.contourf(xx, yy, voronoiBackground, cmap=cmap, alpha=.1)

            file_name = join(self.directory, 'layer_output_' + str(i))
            plt.savefig(file_name)
            plt.close()

    def save_cnf_matrix(self, cnf_matrix_list, model_list):
        for cnf_matrix, model in zip(cnf_matrix_list, model_list):
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=['Primary', 'Metastatic'],
                                  title='Confusion matrix, without normalization')
            file_name = join(self.directory, 'confusion_' + model)
            plt.savefig(file_name)

            plt.figure()
            plot_confusion_matrix(cnf_matrix, normalize=True, classes=['Primary', 'Metastatic'],
                                  title='Normalized confusion matrix')
            file_name = join(self.directory, 'confusion_normalized_' + model)
            plt.savefig(file_name)


    def plot_coef(self, model_list):
        for model, model_name in model_list:
            plt.figure()
            file_name = join(self.directory, 'coef_' + model_name)
            for coef in model.coef_:
                plt.hist(coef, bins=20)
            plt.savefig(file_name)

    def save_all_scores(self, scores):
        file_name = join(self.directory, 'all_scores.csv')
        scores.to_csv(file_name)

    def save_score(self, score, model_name):
        file_name = join(self.directory, model_name + '_params.yml')
        logging.info("saving yml : %s" % file_name)
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump([self.data_params, self.model_params, self.pre_params, str(score)], default_flow_style=False))

    def predict(self, model, x_test, y_test):
        logging.info('predicitng ...')
        y_pred_test = model.predict(x_test)
        if hasattr(model, 'predict_proba'):
            y_pred_test_scores = model.predict_proba(x_test)[:, 1]
        else:
            y_pred_test_scores = y_pred_test

        return y_pred_test, y_pred_test_scores

    def preprocess(self, x_train, x_test):
        logging.info('preprocessing....')
        proc = pre.get_processor(self.pre_params)
        if proc:
            proc.fit(x_train)
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)

            if scipy.sparse.issparse(x_train):
                x_train = x_train.todense()
                x_test = x_test.todense()
        return x_train, x_test

    def extract_features(self, x_train, x_test):
        if self.features_params == {}:
            return x_train, x_test
        logging.info('feature extraction ....')

        proc = feature_extraction.get_processor(self.features_params)
        if proc:
            # proc.fit(x_train)
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)

            if scipy.sparse.issparse(x_train):
                x_train = x_train.todense()
                x_test = x_test.todense()
        return x_train, x_test
