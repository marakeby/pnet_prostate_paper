import logging

import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense, Dropout, Lambda, Concatenate
from keras.regularizers import l2

from data.data_access import Data
from data.pathways.gmt_pathway import get_KEGG_map
from model.builders.builders_utils import get_pnet
from model.layers_custom import f1, Diagonal, SparseTF
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
    features = cols
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    n_features = x.shape[1]

    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')

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

                                                     )

    feature_names = feature_n
    feature_names['inputs'] = cols

    print('Compiling...')

    if repeated_outcomes:
        outcome = decision_outcomes
    else:
        outcome = decision_outcomes[-1]

    model = Model(input=[ins], output=outcome)

    if type(outcome) == list:
        n_outputs = len(outcome)
    else:
        n_outputs = 1

    if type(loss_weights) == list:
        loss_weights = loss_weights
    else:
        loss_weights = [loss_weights] * n_outputs

    print 'loss_weights', loss_weights
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1], loss_weights=loss_weights)

    logging.info('done compiling')

    print_model(model)
    print get_layers(model)
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names


def apply_models(models, inputs):
    output = inputs
    for m in models:
        output = m(output)

    return output


def get_clinical_netowrk(ins, n_features, n_hids, activation):
    layers = []
    for i, n in enumerate(n_hids):
        if i == 0:
            layer = Dense(n, input_shape=(n_features,), activation=activation, W_regularizer=l2(0.001),
                          name='h_clinical' + str(i))
        else:
            layer = Dense(n, activation=activation, W_regularizer=l2(0.001), name='h_clinical' + str(i))

        layers.append(layer)
        drop = 0.5
        layers.append(Dropout(drop, name='droput_clinical_{}'.format(i)))

    merged = apply_models(layers, ins)
    output_layer = Dense(1, activation='sigmoid', name='clinical_out')
    outs = output_layer(merged)

    return outs


def build_pnet2_account_for(optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True, loss_weights=1.0,
                            dropout=0.5,
                            use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None,
                            n_hidden_layers=1,
                            direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform',
                            shuffle_genes=False,
                            attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True,
                            sparse_first_layer=True):
    print data_params

    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    assert len(
        cols.levels) == 3, "expect to have pandas dataframe with 3 levels [{'clinicla, 'genomics'}, genes, features] "

    import pandas as pd
    x_df = pd.DataFrame(x, columns=cols, index=info)
    genomics_label = list(x_df.columns.levels[0]).index(u'genomics')
    genomics_ind = x_df.columns.labels[0] == genomics_label
    genomics = x_df['genomics']
    features_genomics = genomics.columns.remove_unused_levels()

    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    n_features = x_df.shape[1]
    n_features_genomics = len(features_genomics)

    if hasattr(features_genomics, 'levels'):
        genes = features_genomics.levels[0]
    else:
        genes = features_genomics

    print "n_features", n_features, "n_features_genomics", n_features_genomics
    print "genes", len(genes), genes

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')

    ins_genomics = Lambda(lambda x: x[:, 0:n_features_genomics])(ins)
    ins_clinical = Lambda(lambda x: x[:, n_features_genomics:n_features])(ins)

    clinical_outs = get_clinical_netowrk(ins_clinical, n_features, n_hids=[50, 1], activation=activation)

    outcome, decision_outcomes, feature_n = get_pnet(ins_genomics,
                                                     features=features_genomics,
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

                                                     )

    feature_names = feature_n
    feature_names['inputs'] = x_df.columns

    print('Compiling...')

    if repeated_outcomes:
        outcome = decision_outcomes
    else:
        outcome = decision_outcomes[-1]

    outcome_list = outcome + [clinical_outs]

    combined_outcome = Concatenate(axis=-1, name='combine')(outcome_list)
    output_layer = Dense(1, activation='sigmoid', name='combined_outcome')
    combined_outcome = output_layer(combined_outcome)
    outcome = outcome_list + [combined_outcome]
    model = Model(input=[ins], output=outcome)

    if type(outcome) == list:
        n_outputs = len(outcome)
    else:
        n_outputs = 1

    if type(loss_weights) == list:
        loss_weights = loss_weights
    else:
        loss_weights = [loss_weights] * n_outputs

    print 'loss_weights', loss_weights
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1], loss_weights=loss_weights)
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
    logging.info('done compiling')

    print_model(model)
    print get_layers(model)
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names


def build_pnet_KEGG(optimizer, w_reg, dropout=0.5, activation='tanh', use_bias=False,
                    kernel_initializer='glorot_uniform', data_params=None, arch=''):
    print data_params
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print x.shape
    print y.shape
    print info.shape
    print cols.shape

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = {}
    feature_names['inputs'] = cols
    # feature_names.append(cols)

    n_features = x.shape[1]
    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols

    feature_names['h0'] = genes
    # feature_names.append(genes)
    decision_outcomes = []
    n_genes = len(genes)
    genes = list(genes)

    layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg),
                      use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')
    layer1_output = layer1(ins)

    decision0 = Dense(1, activation='sigmoid', name='o0'.format(0))(ins)
    decision_outcomes.append(decision0)

    decision1 = Dense(1, activation='sigmoid', name='o{}'.format(1))(layer1_output)
    decision_outcomes.append(decision1)

    mapp, genes, pathways = get_KEGG_map(genes, arch)

    n_genes, n_pathways = mapp.shape
    logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))

    hidden_layer = SparseTF(n_pathways, mapp, activation=activation, W_regularizer=l2(w_reg),
                            name='h1', kernel_initializer=kernel_initializer,
                            use_bias=use_bias)

    # hidden_layer = Dense(n_pathways, activation=activation, W_regularizer=L1L2_with_map(mapp, w_reg, w_reg),
    #                      kernel_constraint=ConnectionConstaints(mapp), use_bias=False,
    #                      name='h1')

    layer2_output = hidden_layer(layer1_output)
    decision2 = Dense(1, activation='sigmoid', name='o2')(layer2_output)
    decision_outcomes.append(decision2)

    feature_names['h1'] = pathways
    # feature_names.append(pathways)
    print('Compiling...')

    model = Model(input=[ins], output=decision_outcomes)

    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * 3, metrics=[f1])
    # loss=['binary_crossentropy']*(n_hidden_layers +2))
    logging.info('done compiling')

    print_model(model)
    print get_layers(model)
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names
