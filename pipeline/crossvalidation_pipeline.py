import datetime
import logging
from copy import deepcopy
from os import makedirs
from os.path import join, exists
from posixpath import abspath

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from data.data_access import Data
from model.model_factory import get_model
from pipeline.one_split import OneSplitPipeline
from utils.plots import plot_box_plot
from utils.rnd import set_random_seeds

timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())


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


class CrossvalidationPipeline(OneSplitPipeline):
    def __init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params, exp_name):
        OneSplitPipeline.__init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params,
                                  exp_name)

    def run(self, n_splits=5):

        list_model_scores = []
        model_names = []

        for data_params in self.data_params:
            data_id = data_params['id']
            # logging
            logging.info('loading data....')
            data = Data(**data_params)

            x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_validate_test()

            X = np.concatenate((x_train, x_validate_), axis=0)
            y = np.concatenate((y_train, y_validate_), axis=0)
            info = np.concatenate((info_train, info_validate_), axis=0)

            # get model
            logging.info('fitting model ...')

            for model_param in self.model_params:
                if 'id' in model_param:
                    model_name = model_param['id']
                else:
                    model_name = model_param['type']

                set_random_seeds(random_seed=20080808)
                model_name = model_name + '_' + data_id
                m_param = deepcopy(model_param)
                m_param['id'] = model_name
                logging.info('fitting model ...')

                scores = self.train_predict_crossvalidation(m_param, X, y, info, cols, model_name)
                scores_df, scores_mean, scores_std = get_mean_variance(scores)
                list_model_scores.append(scores_df)
                model_names.append(model_name)
                self.save_score(data_params, m_param, scores_df, scores_mean, scores_std, model_name)
                logging.info('scores')
                logging.info(scores_df)
                logging.info('mean')
                logging.info(scores_mean)
                logging.info('std')
                logging.info(scores_std)

        df = pd.concat(list_model_scores, axis=1, keys=model_names)
        df.to_csv(join(self.directory, 'folds.csv'))
        plot_box_plot(df, self.directory)

        return scores_df

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
        n_splits = self.pipeline_params['params']['n_splits']
        skf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)
        i = 0
        scores = []
        model_list = []
        for train_index, test_index in skf.split(X, y.ravel()):
            model = get_model(model_params)
            logging.info('fold # ----------------%d---------' % i)
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            info_train = pd.DataFrame(index=info[train_index])
            info_test = pd.DataFrame(index=info[test_index])
            x_train, x_test = self.preprocess(x_train, x_test)
            # feature extraction
            logging.info('feature extraction....')
            x_train, x_test = self.extract_features(x_train, x_test)

            model = model.fit(x_train, y_train)

            y_pred_test, y_pred_test_scores = self.predict(model, x_test, y_test)
            score_test = self.evaluate(y_test, y_pred_test, y_pred_test_scores)
            logging.info('model {} -- Test score {}'.format(model_name, score_test))
            self.save_prediction(info_test, y_pred_test, y_pred_test_scores, y_test, i, model_name)

            if hasattr(model, 'save_model'):
                logging.info('saving coef')
                save_model(model, model_name + '_' + str(i), self.directory)

            if self.save_train:
                logging.info('predicting training ...')
                y_pred_train, y_pred_train_scores = self.predict(model, x_train, y_train)
                self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, i, model_name,
                                     training=True)

            scores.append(score_test)

            fs_parmas = deepcopy(model_params)
            if hasattr(fs_parmas, 'id'):
                fs_parmas['id'] = fs_parmas['id'] + '_fold_' + str(i)
            else:
                fs_parmas['id'] = fs_parmas['type'] + '_fold_' + str(i)

            model_list.append((model, fs_parmas))
            i += 1
        self.save_coef(model_list, cols)
        logging.info(scores)
        return scores

    def save_score(self, data_params, model_params, scores, scores_mean, scores_std, model_name):
        file_name = join(self.directory, model_name + '_params' + '.yml')
        logging.info("saving yml : %s" % file_name)
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump({'data': data_params, 'models': model_params, 'pre': self.pre_params,
                           'pipeline': self.pipeline_params, 'scores': scores.to_json(),
                           'scores_mean': scores_mean.to_json(), 'scores_std': scores_std.to_json()},
                          default_flow_style=False))
