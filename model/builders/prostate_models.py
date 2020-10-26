import logging

import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense
# from keras.regularizers import l2, L1L2
from keras.regularizers import l2

from data.data_access import Data
# from data.pathways.pathway_loader import get_pathway_files
from model.builders.builders_utils import get_pnet
# from model.constraints_custom import ConnectionConstaints
from model.layers_custom import f1
from model.model_utils import print_model, get_layers


# assumes the first node connected to the first n nodes and so on
def build_pnet(optimizer, w_reg, add_unk_genes=True, sparse=True, dropout=0.5, use_bias=False, activation='tanh',
               loss='binary_crossentropy', data_params=None, n_hidden_layers=1, direction='root_to_leaf',
               batch_normal=False, kernel_initializer='glorot_uniform', shuffle_genes=False, reg_outcomes=False):
    print data_params
    print 'n_hidden_layers', n_hidden_layers
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print x.shape
    print y.shape
    print info.shape
    print cols.shape
    # features = cols.tolist()
    features = cols
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = []
    feature_names.append(features)

    n_features = x.shape[1]

    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols

    # n_genes = len(genes)
    # genes = list(genes)
    # layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg), use_bias=False, name='h0')
    # layer1 = SpraseLayer(n_genes, input_shape=(n_features,), activation=activation,  use_bias=False,name='h0')
    # layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation, name='h0')
    ins = Input(shape=(n_features,), dtype='float32', name='inputs')
    # outcome = layer1(ins)
    # decision_outcomes = []

    # decision_outcome = Dense(1, activation='sigmoid', name='o{}'.format(0))(ins)
    # decision_outcomes.append(decision_outcome)
    #
    # decision_outcome = Dense(1, activation='sigmoid', name='o{}'.format(1))(outcome)
    # decision_outcomes.append(decision_outcome)
    # outcome, decision_outcomes, feature_n = get_pnet(ins, features, genes, n_hidden_layers, direction, activation,
    #                                                  activation_decision, w_reg, dropout, sparse, add_unk_genes, batch_normal, use_bias= use_bias, kernel_initializer=kernel_initializer, shuffle_genes= shuffle_genes, reg_outcomes=reg_outcomes)

    outcome, decision_outcomes, feature_n = get_pnet(ins,
                                                     features,
                                                     genes,
                                                     n_hidden_layers,
                                                     direction,
                                                     activation,
                                                     activation_decision,
                                                     w_reg,
                                                     w_reg_outcomes,
                                                     dropout,
                                                     sparse,
                                                     add_unk_genes,
                                                     batch_normal,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes,
                                                     attention=attention,
                                                     dropout_testing=dropout_testing,
                                                     non_neg=non_neg
                                                     # reg_outcomes=reg_outcomes,
                                                     # adaptive_reg =adaptive_reg,
                                                     # adaptive_dropout=adaptive_dropout
                                                     )
    # outcome= outcome[0:-2]
    # decision_outcomes= decision_outcomes[0:-2]
    # feature_n= feature_n[0:-2]

    feature_names.extend(feature_n)

    print('Compiling...')

    model = Model(input=[ins], output=decision_outcomes)

    # n_outputs = n_hidden_layers + 2
    n_outputs = len(decision_outcomes)
    loss_weights = range(1, n_outputs + 1)
    # loss_weights = [l*l for l in loss_weights]
    loss_weights = [np.exp(l) for l in loss_weights]
    # loss_weights = [l*np.exp(l) for l in loss_weights]
    # loss_weights=1
    print 'loss_weights', loss_weights
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1], loss_weights=loss_weights)
    # loss=['binary_crossentropy']*(n_hidden_layers +2))
    logging.info('done compiling')

    print_model(model)
    print get_layers(model)
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names


# assumes the first node connected to the first n nodes and so on
def build_pnet2(optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True, loss_weights=1.0, dropout=0.5,
                use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None, n_hidden_layers=1,
                direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform', shuffle_genes=False,
                attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True, sparse_first_layer=True):
    print data_params
    print 'n_hidden_layers', n_hidden_layers
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print x.shape
    print y.shape
    print info.shape
    print cols.shape
    # features = cols.tolist()
    features = cols
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = []
    feature_names.append(features)

    n_features = x.shape[1]

    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols

    # n_genes = len(genes)
    # genes = list(genes)
    # layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg), use_bias=False, name='h0')
    # layer1 = SpraseLayer(n_genes, input_shape=(n_features,), activation=activation,  use_bias=False,name='h0')
    # layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation, name='h0')
    ins = Input(shape=(n_features,), dtype='float32', name='inputs')
    # outcome = layer1(ins)
    # decision_outcomes = []

    # decision_outcome = Dense(1, activation='sigmoid', name='o{}'.format(0))(ins)
    # decision_outcomes.append(decision_outcome)
    #
    # decision_outcome = Dense(1, activation='sigmoid', name='o{}'.format(1))(outcome)
    # decision_outcomes.append(decision_outcome)
    outcome, decision_outcomes, feature_n = get_pnet(ins,
                                                     features=features,
                                                     genes=genes,
                                                     n_hidden_layers=n_hidden_layers,
                                                     direction=direction,
                                                     activation=activation,
                                                     activation_decision=activation_decision,
                                                     w_reg=w_reg,
                                                     w_reg_outcomes=w_reg_outcomes,
                                                     dropout=dropout,
                                                     sparse=sparse,
                                                     add_unk_genes=add_unk_genes,
                                                     batch_normal=batch_normal,
                                                     sparse_first_layer=sparse_first_layer,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes,
                                                     attention=attention,
                                                     dropout_testing=dropout_testing,
                                                     non_neg=non_neg
                                                     # reg_outcomes=reg_outcomes,
                                                     # adaptive_reg =adaptive_reg,
                                                     # adaptive_dropout=adaptive_dropout
                                                     )
    # outcome= outcome[0:-2]
    # decision_outcomes= decision_outcomes[0:-2]
    # feature_n= feature_n[0:-2]

    feature_names.extend(feature_n)

    print('Compiling...')

    if repeated_outcomes:
        outcome = decision_outcomes
    else:
        outcome = decision_outcomes[-1]

    model = Model(input=[ins], output=outcome)

    # n_outputs = n_hidden_layers + 2
    if type(outcome) == list:
        n_outputs = len(outcome)
    else:
        n_outputs = 1
    # loss_weights= range(1,n_outputs+1)

    # loss_weights = [l*l for l in loss_weights]
    # loss_weights = [np.exp(l) for l in loss_weights]
    # loss_weights = [np.exp(l)*np.exp(l) for l in loss_weights]
    if type(loss_weights) == list:
        loss_weights = loss_weights
    else:
        loss_weights = [loss_weights] * n_outputs
    # loss_weights=[1.]*
    # loss_weights=1
    print 'loss_weights', loss_weights
    # optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1], loss_weights=loss_weights)
    # loss=['binary_crossentropy']*(n_hidden_layers +2))
    logging.info('done compiling')

    print_model(model)
    print get_layers(model)
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names


def build_dense(optimizer, n_weights, w_reg, activation='tanh', loss='binary_crossentropy', data_params=None):
    print data_params

    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print x.shape
    print y.shape
    print info.shape
    print cols.shape
    # features = cols.tolist()
    features = cols
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = []
    feature_names.append(features)

    n_features = x.shape[1]

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')
    n = np.ceil(float(n_weights) / float(n_features))
    print n
    layer1 = Dense(units=int(n), activation=activation, W_regularizer=l2(w_reg), name='h0')
    outcome = layer1(ins)
    outcome = Dense(1, activation=activation_decision, name='output')(outcome)
    model = Model(input=[ins], output=outcome)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=[f1])
    # loss=['binary_crossentropy']*(n_hidden_layers +2))
    logging.info('done compiling')

    print_model(model)
    print get_layers(model)
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names
