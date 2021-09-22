import os
from os.path import join
import numpy as np
import yaml
from data.data_access import Data
from model import model_factory


class DataModelLoader():
    def __init__(self, params_file):
        self.dir_path = os.path.dirname(os.path.realpath(params_file))
        model_parmas, data_parmas = self.load_parmas(params_file)
        data_reader = Data(**data_parmas)
        self.model = None
        x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data_reader.get_train_validate_test()

        self.x_train = x_train
        self.x_test = np.concatenate([x_validate_, x_test_], axis=0)

        self.y_train = y_train
        self.y_test = np.concatenate([y_validate_, y_test_], axis=0)

        self.info_train = info_train
        self.info_test = list(info_validate_) + list(info_test_)
        self.columns = cols

    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test, self.info_train, self.info_test, self.columns

    def get_model(self, model_name='P-net_params.yml'):
        # if self.model is None:
        self.model = self.load_model(self.dir_path, model_name)
        return self.model

    def load_model(self, model_dir_, model_name):
        # 1 - load architecture
        params_filename = join(model_dir_, model_name + '_params.yml')
        stream = file(params_filename, 'r')
        params = yaml.load(stream)
        # print params
        # fs_model = model_factory.get_model(params['model_params'][0])
        fs_model = model_factory.get_model(params['model_params'])
        # 2 -compile model and load weights (link weights)
        weights_file = join(model_dir_, 'fs/{}.h5'.format(model_name))
        model = fs_model.load_model(weights_file)
        return model

    def load_parmas(self, params_filename):
        stream = file(params_filename, 'r')
        params = yaml.load(stream, Loader=yaml.UnsafeLoader)
        model_parmas = params['model_params']
        data_parmas = params['data_params']
        return model_parmas, data_parmas
