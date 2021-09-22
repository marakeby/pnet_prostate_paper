import datetime
import logging
import math
import os

import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from model.callbacks_custom import GradientCheckpoint, FixedEarlyStopping
from model.model_utils import get_layers, plot_history, get_coef_importance
from utils.logs import DebugFolder


class Model(BaseEstimator):
    def __init__(self, build_fn, **sk_params):
        params = sk_params
        params['build_fn'] = build_fn
        self.set_params(params)

    def set_params(self, sk_params):
        self.params = sk_params
        self.build_fn = sk_params['build_fn']
        self.sk_params = sk_params
        self.batch_size = sk_params['fitting_params']['batch_size']
        self.model_params = sk_params['model_params']
        self.nb_epoch = sk_params['fitting_params']['epoch']
        self.verbose = sk_params['fitting_params']['verbose']
        self.select_best_model = sk_params['fitting_params']['select_best_model']
        if 'save_gradient' in sk_params['fitting_params']:
            self.save_gradient = sk_params['fitting_params']['save_gradient']
        else:
            self.save_gradient = False

        if 'prediction_output' in sk_params['fitting_params']:
            self.prediction_output = sk_params['fitting_params']['prediction_output']
        else:
            self.prediction_output = 'average'

        if 'x_to_list' in sk_params['fitting_params']:
            self.x_to_list = sk_params['fitting_params']['x_to_list']
        else:
            self.x_to_list = False

        if 'period' in sk_params['fitting_params']:
            self.period = sk_params['fitting_params']['period']
        else:
            self.period = 10

        if 'max_f1' in sk_params['fitting_params']:
            self.max_f1 = sk_params['fitting_params']['max_f1']
        else:
            self.max_f1 = False

        if 'debug' in sk_params['fitting_params']:
            self.debug = sk_params['fitting_params']['debug']
        else:
            self.debug = False

        if 'feature_importance' in sk_params:
            self.feature_importance = sk_params['feature_importance']

        if 'loss' in sk_params['model_params']:
            self.loss = sk_params['model_params']['loss']
        else:
            self.loss = 'binary_crossentropy'

        if 'reduce_lr' in sk_params['fitting_params']:
            self.reduce_lr = sk_params['fitting_params']['reduce_lr']
        else:
            self.reduce_lr = False

        if 'lr' in sk_params['fitting_params']:
            self.lr = sk_params['fitting_params']['lr']
        else:
            self.lr = 0.001

        if 'reduce_lr_after_nepochs' in sk_params['fitting_params']:
            self.reduce_lr_after_nepochs = True
            self.reduce_lr_drop = sk_params['fitting_params']['reduce_lr_after_nepochs']['drop']
            self.reduce_lr_epochs_drop = sk_params['fitting_params']['reduce_lr_after_nepochs']['epochs_drop']
        else:
            self.reduce_lr_after_nepochs = False

        pid = os.getpid()
        timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}-{0:%S}'.format(datetime.datetime.now())
        self.debug_folder = DebugFolder().get_debug_folder()
        self.save_filename = os.path.join(self.debug_folder,
                                          sk_params['fitting_params']['save_name'] + str(pid) + timeStamp)
        self.shuffle = sk_params['fitting_params']['shuffle']
        self.monitor = sk_params['fitting_params']['monitor']
        self.early_stop = sk_params['fitting_params']['early_stop']

        self.duplicate_samples = False
        self.class_weight = None
        if 'duplicate_samples' in sk_params:
            self.duplicate_samples = sk_params['duplicate_samples']

        if 'n_outputs' in sk_params['fitting_params']:
            self.n_outputs = sk_params['fitting_params']['n_outputs']
        else:
            self.n_outputs = 1

        if 'class_weight' in sk_params['fitting_params']:
            self.class_weight = sk_params['fitting_params']['class_weight']
            logging.info('class_weight {}'.format(self.class_weight))

    def get_params(self, deep=False):
        return self.params

    def get_callbacks(self, X_train, y_train):
        callbacks = []
        if self.reduce_lr:
            reduce_lr = ReduceLROnPlateau(monitor=self.monitor, factor=0.5,
                                          patience=2, min_lr=0.000001, verbose=1, mode='auto')
            logging.info("adding a reduce lr on Plateau callback%s " % reduce_lr)
            callbacks.append(reduce_lr)

        if self.select_best_model:
            saving_callback = ModelCheckpoint(self.save_filename, monitor=self.monitor, verbose=1, save_best_only=True,
                                              mode='max')
            logging.info("adding a saving_callback%s " % saving_callback)
            callbacks.append(saving_callback)

        if self.save_gradient:
            saving_gradient = GradientCheckpoint(self.save_filename, self.feature_importance, X_train, y_train,
                                                 self.nb_epoch,
                                                 self.feature_names, period=self.period)
            logging.info("adding a saving_callback%s " % saving_gradient)
            callbacks.append(saving_gradient)

        if self.early_stop:
            # early_stop = EarlyStopping(monitor=self.monitor, min_delta=0.01, patience=20, verbose=1, mode='min', baseline=0.6, restore_best_weights=False)
            early_stop = FixedEarlyStopping(monitors=[self.monitor], min_deltas=[0.0], patience=10, verbose=1,
                                            modes=['max'], baselines=[0.0])
            callbacks.append(early_stop)

        if self.reduce_lr_after_nepochs:
            # learning rate schedule
            def step_decay(epoch, init_lr, drop, epochs_drop):
                initial_lrate = init_lr
                lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                return lrate

            from functools import partial
            step_decay_part = partial(step_decay, init_lr=self.lr, drop=self.reduce_lr_drop,
                                      epochs_drop=self.reduce_lr_epochs_drop)
            lr_callback = LearningRateScheduler(step_decay_part, verbose=1)
            callbacks.append(lr_callback)
        return callbacks

    def get_validation_set(self, X_train, y_train, test_size=0.2):
        X_train1, X_validatioin, y_train_debug, y_validation_debug = train_test_split(X_train, y_train,
                                                                                      test_size=test_size,
                                                                                      stratify=y_train,
                                                                                      random_state=422342)
        training_data = [X_train1, y_train_debug]
        validation_data = [X_validatioin, y_validation_debug]
        return training_data, validation_data

    def get_th(self, y_validate, pred_scores):
        thresholds = np.arange(0.1, 0.9, 0.01)
        print thresholds
        scores = []
        for th in thresholds:
            y_pred = pred_scores > th
            f1 = metrics.f1_score(y_validate, y_pred)
            precision = metrics.precision_score(y_validate, y_pred)
            recall = metrics.recall_score(y_validate, y_pred)
            accuracy = accuracy_score(y_validate, y_pred)
            score = {}
            score['accuracy'] = accuracy
            score['precision'] = precision
            score['f1'] = f1
            score['recall'] = recall
            score['th'] = th
            scores.append(score)
        ret = pd.DataFrame(scores)
        print ret
        best = ret[ret.f1 == max(ret.f1)]
        th = best.th.values[0]
        return th

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        ret = self.build_fn(**self.model_params)
        if type(ret) == tuple:
            self.model, self.feature_names = ret
        else:
            self.model = ret
        logging.info('start fitting')

        callbacks = self.get_callbacks(X_train, y_train)

        if self.class_weight == 'auto':
            classes = np.unique(y_train)
            class_weights = class_weight.compute_class_weight('balanced', classes, y_train.ravel())
            class_weights = dict(zip(classes, class_weights))
        else:
            class_weights = self.class_weight

        logging.info('class_weight {}'.format(class_weights))

        # speical case of survival
        if y_train.dtype.fields is not None:
            y_train = y_train['time']

        if self.debug:
            # train on 80 and validate on 20, report validation and training performance over epochs
            logging.info('dividing training data into train and validation with split 80 to 20')
            training_data, validation_data = self.get_validation_set(X_train, y_train, test_size=0.2)
            X_train, y_train = training_data
            X_val, y_val = validation_data

        if self.n_outputs > 1:
            y_train = [y_train] * self.n_outputs
            y_val = [y_val] * self.n_outputs

        if not X_val is None:
            validation_data = [X_val, y_val]
        else:
            validation_data = []

        history = self.model.fit(X_train, y_train, validation_data=validation_data, epochs=self.nb_epoch,
                                 batch_size=self.batch_size,
                                 verbose=self.verbose, callbacks=callbacks,
                                 shuffle=self.shuffle, class_weight=class_weights)

        '''
        saving history
        '''
        plot_history(history.history, self.save_filename + '_validation')
        hist_df = pd.DataFrame(history.history)
        with open(self.save_filename + '_train_history.csv', mode='w') as f:
            hist_df.to_csv(f)

        # if not X_val is None:
        pred_validate_score = self.get_prediction_score(X_train)
        if self.n_outputs > 1:
            y_train = y_train[0]

        if self.max_f1:
            self.th = self.get_th(y_train, pred_validate_score)
            logging.info('prediction threshold {}'.format(self.th))

        if hasattr(self, 'feature_importance'):
            self.coef_ = self.get_coef_importance(X_train, y_train, target=-1,
                                                  feature_importance=self.feature_importance)

        return self

    def get_coef_importance(self, X_train, y_train, target=-1, feature_importance='deepexplain_grad*input'):

        coef_ = get_coef_importance(self.model, X_train, y_train, target, feature_importance, detailed=False)
        return coef_

    def predict(self, X_test):
        if self.select_best_model:
            logging.info("loading model %s" % self.save_filename)
            self.model.load_weights(self.save_filename)

        prediction_scores = self.get_prediction_score(X_test)

        std_th = .5
        if hasattr(self, 'th'):
            std_th = self.th
        elif self.loss == 'hinge':
            std_th = 1.
        elif self.loss == 'binary_crossentropy':
            std_th = .5

        if self.loss == 'mean_squared_error':
            prediction = prediction_scores
        else:
            prediction = np.where(prediction_scores >= std_th, 1., 0.)

        return prediction

    def get_prediction_score(self, X):
        prediction_scores = self.model.predict(X)
        if (type(prediction_scores) == list):
            if len(prediction_scores) > 1:
                if self.prediction_output == 'average':
                    prediction_scores = np.mean(np.array(prediction_scores), axis=0)
                else:
                    prediction_scores = prediction_scores[-1]

        print np.array(prediction_scores).shape
        return np.array(prediction_scores)

    def predict_proba(self, X_test):
        prediction_scores = self.get_prediction_score(X_test)
        if type(X_test) is list:
            n_samples = X_test[0].shape[0]
        else:
            n_samples = X_test.shape[0]
        ret = np.ones((n_samples, 2))
        ret[:, 0] = 1. - prediction_scores.ravel()
        ret[:, 1] = prediction_scores.ravel()
        print ret.shape
        return ret

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return accuracy_score(y_test, y_pred)

    def get_layer_output(self, layer_name, X):
        layer = self.model.get_layer(layer_name)
        inp = self.model.input
        functor = K.function(inputs=[inp, K.learning_phase()], outputs=[layer.output])  # evaluation function
        layer_outs = functor([X, 0.])
        return layer_outs

    def get_layer_outputs(self, X):
        inp = self.model.input
        layers = get_layers(self.model)[1:]
        layer_names = []
        for l in layers:
            layer_names.append(l.name)
        outputs = [layer.get_output_at(0) for layer in layers]  # all layer outputs
        functor = K.function(inputs=[inp, K.learning_phase()], outputs=outputs)  # evaluation function
        layer_outs = functor([X, 0.])
        ret = dict(zip(layer_names, layer_outs))
        return ret

    def save_model(self, filename):
        model_json = self.model.to_json()
        json_file_name = filename.replace('.h5', '.json')
        with open(json_file_name, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(filename)

    def load_model(self, filename):
        ret = self.build_fn(**self.model_params)
        if type(ret) == tuple:
            self.model, self.feature_names = ret
        else:
            self.model = ret

        self.model.load_weights(filename)

        return self

    def save_feature_importance(self, filename):

        coef = self.coef_
        if type(coef) != list:
            coef = [self.coef_]

        for i, c in enumerate(coef):
            df = pd.DataFrame(c)
            df.to_csv(filename + str(i) + '.csv')

    def get_named_coef(self):

        if not hasattr(self, 'feature_names'):
            return self.coef_
        coef = self.coef_
        coef_dfs = {}
        common_keys = set(coef.keys()).intersection(self.feature_names.keys())
        for k in common_keys:
            c = coef[k]
            names = self.feature_names[k]
            df = pd.DataFrame(c.ravel(), index=names, columns=['coef'])
            coef_dfs[k] = df
        return coef_dfs

    def get_coef(self):
        return self.coef_
