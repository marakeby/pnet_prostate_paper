import datetime
import logging
from copy import deepcopy
from os import makedirs
from os.path import join, exists
from posixpath import abspath

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

from data.data_access import Data
# from features_processing.feature_scaler import FeatureScaler
from model.model_factory import get_model
from pipeline.one_split import OneSplitPipeline
from utils.plots import plot_box_plot
from utils.rnd import set_random_seeds
from functools import partial
timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())

import multiprocessing as mp

def save_model(model, model_name, directory_name):
    filename = join(abspath(directory_name), 'fs')
    logging.info('saving model {} coef to dir ({})'.format(model_name, filename))
    if not exists(filename.strip()):
        makedirs(filename)
    filename = join(filename, model_name + '.h5')
    logging.info('FS dir ({})'.format(filename))
    model.save_model(filename)


def get_mean_variance(scores):
    df = pd.DataFrame(scores)
    return df, df.mean(), df.std()


class LeaveOneOutPipeline(OneSplitPipeline):
    def __init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params, exp_name):
        OneSplitPipeline.__init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params,
                                  exp_name)

    def run(self):
        for data_params in self.data_params:
            data_id = data_params['id']
            # logging
            logging.info('loading data....')
            data = Data(**data_params)

            x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_validate_test()

            X = np.concatenate((x_train, x_validate_, x_test_), axis=0)
            y = np.concatenate((y_train, y_validate_, y_test_), axis=0)
            info = np.concatenate((info_train, info_validate_, info_test_), axis=0)

            # get model
            logging.info('fitting model ...')

            for model_param in self.model_params:
                # model_name = m['type']
                if 'id' in model_param:
                    model_name = model_param['id']
                else:
                    model_name = model_param['type']

                set_random_seeds(random_seed=20080808)
                model_name = model_name + '_' + data_id
                m_param = deepcopy(model_param)
                m_param['id'] = model_name
                logging.info('fitting model ...')

                prediction_df = self.train_predict_crossvalidation(m_param, X, y, info, cols, model_name)
                filename = join(self.directory, model_name + '.csv')
                prediction_df.to_csv(filename)
                # scores_df, scores_mean, scores_std = get_mean_variance(scores)
                # list_model_scores.append(scores_df)
                # model_names.append(model_name)
                # self.save_score(scores_df, scores_mean, scores_std, model_param['type'])
                # self.save_score(data_params, m_param, scores_df, scores_mean, scores_std, model_name)
                # logging.info('scores')
                # logging.info(scores_df)
                # logging.info('mean')
                # logging.info(scores_mean)
                # logging.info('std')
                # logging.info(scores_std)

        # df = pd.concat(list_model_scores, axis=1, keys=model_names)
        # df.to_csv(join(self.directory, 'folds.csv'))
        # plot_box_plot(df, self.directory)

        return

    def save_prediction(self, info, y_pred, y_pred_score, y_test, fold_num, model_name, training=False):
        if training:
            file_name = join(self.directory, model_name + '_traing_fold_' + str(fold_num) + '.csv')
        else:
            file_name = join(self.directory, model_name + '_testing_fold_' + str(fold_num) + '.csv')
        logging.info("saving : %s" % file_name)
        info['pred'] = y_pred
        info['pred_score'] = y_pred_score
        info['y'] = y_test
        info.to_csv(file_name)

    def train_predict_crossvalidation(self, model_params, X, y, info, cols, model_name):
        logging.info('model_params: {}'.format(model_params))
        # n_splits = self.pipeline_params['params']['n_splits']
        # skf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)
        splitter = LeaveOneOut()
        folds = list(splitter.split(X, y.ravel()))
        print('folds', len(folds))

        # pool = mp.Pool(5)
        # prediction = []
        # def log_result(x):
        #     prediction.append(x)
        #     print ('------------- {} ---------------'.format(len(prediction)))
        fold_ids = range(len(folds))
        f = partial(eval_model, model_params, X, y, info, folds)

        p= mp.Pool(5)
        prediction = p.map(f,fold_ids )
        # for fold in folds:
        #     print len(fold)
        #     pool.apply_async(eval_model, args=(self, model_params, X, y, info, fold), callback=log_result)

        # Close the pool for new tasks
        # pool.close()
        # Wait for all tasks to complete at this point
        # pool.join()
        # result = pd.concat(prediction, axis=0, sort=True)
            # i += 1
            # if i>10:
            #     break
        prediction_df = pd.concat(prediction, axis=0)
        return prediction_df

    def save_score(self, data_params, model_params, scores, scores_mean, scores_std, model_name):
        file_name = join(self.directory, model_name + '_params' + '.yml')
        logging.info("saving yml : %s" % file_name)
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump({'data': data_params, 'models': model_params, 'pre': self.pre_params,
                           'pipeline': self.pipeline_params, 'scores': scores.to_json(),
                           'scores_mean': scores_mean.to_json(), 'scores_std': scores_std.to_json()},
                          default_flow_style=False))
    #
    #

def predict( model, x_test):
    y_pred_test = model.predict(x_test)
    if hasattr(model, 'predict_proba'):
        y_pred_test_scores = model.predict_proba(x_test)[:, 1]
    else:
        y_pred_test_scores = y_pred_test
    return y_pred_test, y_pred_test_scores


def eval_model( model_params, X, y, info, folds, fold_id):

    print('fold # {}'.format(fold_id))
    train_index, test_index = folds[fold_id]
    model = get_model(model_params)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # info_train = pd.DataFrame(index=info[train_index])
    info_test = pd.DataFrame(index=info[test_index])
    # x_train, x_test = self.preprocess(x_train, x_test)
    # x_train, x_test = self.extract_features(x_train, x_test)
    model = model.fit(x_train, y_train)
    y_pred_test, y_pred_test_scores = predict(model, x_test)
    info_test['y_pred_test_scores'] = y_pred_test_scores
    info_test['y_pred_test'] = y_pred_test
    info_test['y_test'] = y_test
    return info_test

