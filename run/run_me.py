# ----
# from pipeline.increasing_sample_size_pipeline import IncreasingSampleSizePipeline
# from pipeline.one_split_ensamble import OneSplitPipelineEnsemble
import imp
import logging
import random

import numpy as  np
import tensorflow as tf


random_seed = 234

random.seed(random_seed)
np.random.seed(random_seed)

tf.random.set_random_seed(random_seed)
import os
import sys
from os.path import join, dirname

current_dir = dirname(os.path.realpath(__file__))
print current_dir

sys.path.insert(0, dirname(current_dir))

from pipeline.crossvalidation_pipeline import CrossvalidationPipeline
from pipeline.one_split import OneSplitPipeline
from pipeline.train_validate import TrainValidatePipeline

from utils.logs import set_logging, DebugFolder

import datetime

timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#pnet
params_file = './params/p1000/pnet/onsplit_average_reg_10_tanh_large_testing'
# params_file = './params/p1000/pnet/onsplit_average_reg_10_tanh_large_testing_6layers'
# params_file = './params/p1000/pnet/crossvalidation_average_reg_10_tanh'
# params_file = './params/p1000/pnet/crossvalidation_average_reg_10_tanh_split18'

#other ML models
# params_file = './params/p1000/compare/onsplit_ML_test'
# params_file = './params/p1000/compare/crossvalidation_ML_test'

#dense
# params_file = './params/p1000/dense/onesplit_number_samples_dense_sameweights'
# params_file = './params/p1000/dense/onsplit_dense'

#number_samples
# params_file = './params/p1000/number_samples/crossvalidation_average_reg_10'
# params_file = './params/p1000/number_samples/crossvalidation_average_reg_10_tanh'
# params_file = './params/p1000/number_samples/crossvalidation_number_samples_dense_sameweights'

#external_validation
# params_file = './params/p1000/external_validation/pnet_validation'

params_file = join(current_dir, params_file)
log_dir = params_file.replace('params', 'logs')
log_dir = log_dir + timeStamp
set_logging(log_dir)

logging.info('random seed %d' % random_seed)
params_file_full = params_file + '.py'
params = imp.load_source(params_file, params_file_full)

DebugFolder(log_dir)
if params.pipeline['type'] == 'one_split':
    pipeline = OneSplitPipeline(task=params.task, data_params=params.data, model_params=params.models,
                                pre_params=params.pre, feature_params=params.features, pipeline_params=params.pipeline,
                                exp_name=log_dir)

elif params.pipeline['type'] == 'crossvalidation':
    pipeline = CrossvalidationPipeline(task=params.task, data_params=params.data, feature_params=params.features,
                                       model_params=params.models, pre_params=params.pre,
                                       pipeline_params=params.pipeline, exp_name=log_dir)
elif params.pipeline['type'] == 'Train_Validate':
    pipeline = TrainValidatePipeline (data_params = params.data,  model_params = params.models, pre_params = params.pre, feature_params = params.features, pipeline_params=params.pipeline, exp_name = log_dir)

pipeline.run()
