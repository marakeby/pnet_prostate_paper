import os
import sys
from os.path import dirname

current_dir = dirname(os.path.realpath(__file__))
print current_dir

sys.path.insert(0, dirname(current_dir))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from os.path import join
from config_path import PROSTATE_LOG_PATH
from utils.loading_utils import DataModelLoader
from utils.evaluate import evalualte

base_dir = join(PROSTATE_LOG_PATH, 'pnet')
model_name = 'onsplit_average_reg_10_tanh_large_testing'

model_dir = join(base_dir, model_name)
model_file = 'P-net_ALL'
params_file = join(model_dir, model_file + '_params.yml')

# params_file = '/Users/haithamelmarakeby/PycharmProjects/pnet_prostate/run/logs/p1000/pnet/onsplit_average_reg_10_tanh_large_testing_Apr-11_11-22/P-net_ALL_params.yml'
print('loading ', params_file)
loader = DataModelLoader(params_file)
nn_model = loader.get_model(model_file)
print(nn_model.model.summary())
# feature_names= nn_model.feature_names

x_train, x_test, y_train, y_test, info_train, info_test, columns = loader.get_data()

info = list(info_train) + list(info_test)
pred_scores = nn_model.predict_proba(x_test)[:, 1]
pred = nn_model.predict(x_test)

metrics = evalualte(pred, y_test, pred_scores)
print (metrics)

# training
pred_scores = nn_model.predict_proba(x_train)[:, 1]
pred = nn_model.predict(x_train)
# print (pred)
# print(sum(pred))
metrics = evalualte(pred, y_train, pred_scores)
print (metrics)
# print (pred_scores)


# from tensorflow.python.saved_model import loader_impl
# from tensorflow.python.keras.saving.saved_model import load as saved_model_load

# model = saved_model_load.load('pnet', custom_objects={'f1':f1} )
