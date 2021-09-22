import logging
from copy import deepcopy
from os import makedirs
from os.path import join, exists
from posixpath import abspath

import numpy as np
import pandas as pd
import scipy.sparse
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from data.data_access import Data
from model.model_factory import get_model
from pipeline.pipe_utils import get_model_id, get_coef_from_model, get_balanced
from preprocessing import pre
from utils.evaluate import evalualte_survival, evalualte_classification_binary, evalualte_regression
from utils.plots import generate_plots, plot_roc, plot_prc, save_confusion_matrix
# timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())
from utils.rnd import set_random_seeds


def save_model(model, model_name, directory_name):
    filename = join(abspath(directory_name), 'fs')
    logging.info('saving model {} coef to dir ({})'.format(model_name, filename))
    if not exists(filename.strip()):
        makedirs(filename)
    filename = join(filename, model_name + '.h5')
    logging.info('FS dir ({})'.format(filename))
    model.save_model(filename)


def get_model_name(model):
    if 'id' in model:
        model_name = model['id']
    else:
        model_name = model['type']
    return model_name


class OneSplitPipeline:
    def __init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params, exp_name):

        self.task = task
        if type(data_params) == list:
            self.data_params = data_params
        else:
            self.data_params = [data_params]
        self.pre_params = pre_params
        self.features_params = feature_params
        self.model_params = model_params
        self.exp_name = exp_name
        self.pipeline_params = pipeline_params
        print (pipeline_params)
        if 'save_train' in pipeline_params['params']:
            self.save_train = pipeline_params['params']['save_train']
        else:
            self.save_train = False
        if 'eval_dataset' in pipeline_params['params']:
            self.eval_dataset = pipeline_params['params']['eval_dataset']
        else:
            self.eval_dataset = 'validation'
        self.prapre_saving_dir()

    def prapre_saving_dir(self):
        self.directory = self.exp_name
        if not exists(self.directory):
            makedirs(self.directory)

    def save_prediction(self, info, y_pred, y_pred_scores, y_test, model_name, training=False):

        if training:
            file_name = join(self.directory, model_name + '_training.csv')
        else:
            file_name = join(self.directory, model_name + '_testing.csv')
        logging.info("saving results : %s" % file_name)
        print('info', info)
        info = pd.DataFrame(index=info)
        info['pred'] = y_pred
        info['pred_scores'] = y_pred_scores

        # survival case
        # https://docs.scipy.org/doc/numpy/user/basics.rec.html
        if y_test.dtype.fields is not None:
            fields = y_test.dtype.fields
            for f in fields:
                info['y_{}'.format(f)] = y_test[f]
        else:
            info['y'] = y_test
        info.to_csv(file_name)

    def get_list(self, x, cols):
        x_df = pd.DataFrame(x, columns=cols)
        genes = cols.get_level_values(0).unique()
        genes_list = []
        input_shapes = []
        for g in genes:
            g_df = x_df.loc[:, g].as_matrix()
            input_shapes.append(g_df.shape[1])
            genes_list.append(g_df)
        return genes_list

    def get_train_test(self, data):
        x_train, x_test, y_train, y_test, info_train, info_test, columns = data.get_train_test()
        balance_train = False
        balance_test = False
        p = self.pipeline_params['params']
        if 'balance_train' in p:
            balance_train = p['balance_train']
        if 'balance_test' in p:
            balance_test = p['balance_test']

        if balance_train:
            x_train, y_train, info_train = get_balanced(x_train, y_train, info_train)
        if balance_test:
            x_test, y_test, info_test = get_balanced(x_test, y_test, info_test)
        return x_train, x_test, y_train, y_test, info_train, info_test, columns

    def run(self):
        test_scores = []
        model_names = []
        model_list = []
        y_pred_test_list = []
        y_pred_test_scores_list = []
        y_test_list = []
        fig = plt.figure()
        fig.set_size_inches((10, 6))
        print self.data_params
        for data_params in self.data_params:
            print 'data_params', data_params
            data_id = data_params['id']
            logging.info('loading data....')
            data = Data(**data_params)
            # get data
            x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_validate_test()

            logging.info('predicting')
            if self.eval_dataset == 'validation':
                x_t = x_validate_
                y_t = y_validate_
                info_t = info_validate_
            else:
                x_t = np.concatenate((x_test_, x_validate_))
                y_t = np.concatenate((y_test_, y_validate_))
                info_t = info_test_.append(info_validate_)

            logging.info('x_train {} y_train {} '.format(x_train.shape, y_train.shape))
            logging.info('x_test {} y_test {} '.format(x_t.shape, y_t.shape))

            # pre-processing
            logging.info('preprocessing....')
            x_train, x_test = self.preprocess(x_train, x_t)
            for m in self.model_params:
                # get model
                model_params_ = deepcopy(m)
                set_random_seeds(random_seed=20080808)
                model = get_model(model_params_)
                logging.info('fitting')
                logging.info(model_params_)
                if model_params_['type'] == 'nn' and not self.eval_dataset == 'validation':
                    model = model.fit(x_train, y_train, x_validate_, y_validate_)
                else:
                    model = model.fit(x_train, y_train)
                logging.info('predicting')

                model_name = get_model_name(model_params_)
                model_name = model_name + '_' + data_id
                model_params_['id'] = model_name
                logging.info('model id: {}'.format(model_name))
                model_list.append((model, model_params_))
                y_pred_test, y_pred_test_scores = self.predict(model, x_test, y_t)
                test_score = self.evaluate(y_t, y_pred_test, y_pred_test_scores)
                logging.info('model name {} -- Test score {}'.format(model_name, test_score))
                test_scores.append(test_score)
                model_names.append(model_name)
                logging.info('saving results')
                self.save_score(data_params, model_params_, test_score, model_name)
                self.save_prediction(info_t, y_pred_test, y_pred_test_scores, y_t, model_name)
                y_test_list.append(y_t)
                y_pred_test_list.append(y_pred_test)
                y_pred_test_scores_list.append(y_pred_test_scores)

                # saving coef
                self.save_coef([(model, model_params_)], cols)

                # saving confusion matrix
                cnf_matrix = confusion_matrix(y_t, y_pred_test)
                save_confusion_matrix(cnf_matrix, self.directory, model_name)

                # saving coefs
                logging.info('saving coef')
                if hasattr(model, 'save_model'):
                    logging.info('saving coef')
                    save_model(model, model_name, self.directory)

                if self.save_train:
                    y_pred_train, y_pred_train_scores = self.predict(model, x_train, y_train)
                    train_score = self.evaluate(y_train, y_pred_train, y_pred_train_scores)
                    logging.info('model {} -- Train score {}'.format(model_name, train_score))
                    self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, model_name,
                                         training=True)

        test_scores = pd.DataFrame(test_scores, index=model_names)
        generate_plots(test_scores, self.directory)
        self.save_all_scores(test_scores)

        if self.task == 'classification_binary':
            auc_fig = plt.figure()
            auc_fig.set_size_inches((10, 6))
            prc_fig = plt.figure()
            prc_fig.set_size_inches((10, 6))
            for y_test, y_pred_test, y_pred_test_scores, model_name in zip(y_test_list, y_pred_test_list,
                                                                           y_pred_test_scores_list, model_names):
                plot_roc(auc_fig, y_test, y_pred_test_scores, self.directory, label=model_name)
                plot_prc(prc_fig, y_test, y_pred_test_scores, self.directory, label=model_name)
            auc_fig.savefig(join(self.directory, 'auc_curves'))
            prc_fig.savefig(join(self.directory, 'auprc_curves'))
        return test_scores

    def evaluate(self, y_test, y_pred_test, y_pred_test_scores):
        if self.task == 'survival':
            test_score = evalualte_survival(y_test, y_pred_test)
        if self.task == 'classification_binary':
            test_score = evalualte_classification_binary(y_test, y_pred_test, y_pred_test_scores)
        if self.task == 'regression':
            test_score = evalualte_regression(y_test, y_pred_test, y_pred_test_scores)
        return test_score

    def save_coef(self, model_list, cols):
        coef_df = pd.DataFrame(index=cols)

        dir_name = join(self.directory, 'fs')
        if not exists(dir_name):
            makedirs(dir_name)

        for model, model_params in model_list:
            model_name = get_model_id(model_params)
            c_ = get_coef_from_model(model)
            logging.info('saving coef ')
            model_name_col = model_name
            if hasattr(model, 'get_named_coef') and c_ is not None:
                file_name = join(dir_name, 'coef_' + model_name)
                coef = model.get_named_coef()
                if type(coef) == list:
                    for i, c in enumerate(coef):
                        if type(c) == pd.DataFrame:
                            c.to_csv(file_name + '_layer' + str(i) + '.csv')
                elif type(coef) == dict:
                    for c in coef.keys():
                        if type(coef[c]) == pd.DataFrame:
                            coef[c].to_csv(file_name + '_layer' + str(c) + '.csv')

            if type(c_) == list or type(c_) == tuple:
                coef_df[model_name_col] = c_[0]
            else:
                coef_df[model_name_col] = c_
        file_name = join(dir_name, 'coef.csv')
        coef_df.to_csv(file_name)


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

    def save_score(self, data_params, model_params, score, model_name):
        file_name = join(self.directory, model_name + '_params.yml')
        logging.info("saving yml : %s" % file_name)
        yml_dict = {
            'task': self.task,
            'exp_name': self.exp_name,
            'data_params': data_params,
            'pre_params': self.pre_params,
            'features_params': self.features_params,
            'model_params': model_params,
            'pipeline_params': self.pipeline_params,
            'score': str(score)}

        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump(yml_dict, default_flow_style=False)
            )

    def predict(self, model, x_test, y_test):
        logging.info('predicitng ...')
        y_pred_test = model.predict(x_test)
        if hasattr(model, 'predict_proba'):
            y_pred_test_scores = model.predict_proba(x_test)[:, 1]
        else:
            y_pred_test_scores = y_pred_test

        print 'y_pred_test', y_pred_test.shape, y_pred_test_scores.shape
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
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)

            if scipy.sparse.issparse(x_train):
                x_train = x_train.todense()
                x_test = x_test.todense()
        return x_train, x_test
