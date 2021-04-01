import sys

import numpy as np
from keras import backend as K
from keras.engine import InputLayer
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from model.model_utils import get_layers


def predict(model, X, loss=None):
    prediction_scores = model.predict(X)

    prediction_scores = np.mean(np.array(prediction_scores), axis=0)
    if loss == 'hinge':
        prediction = np.where(prediction_scores >= 0.0, 1., 0.)
    else:
        prediction = np.where(prediction_scores >= 0.5, 1., 0.)

    return prediction


# def get_gradient_layer(model, X, y, layer):
#
#     # print 'layer', layer
#     grad = model.optimizer.get_gradients(model.total_loss, layer)
#     gradients = layer *  grad# gradient tensors
#     # gradients =  grad# gradient tensors
#     # gradients = layer * model.optimizer.get_gradients(model.output[0,0], layer) # gradient tensors
#     # gradients = model.optimizer.get_gradients(model.output[0,0], layer) # gradient tensors
#     # gradients =  model.optimizer.get_gradients(model.total_loss, layer) # gradient tensors
#
#     #special case of repeated outputs (e.g. output for each hidden layer)
#     if type(y) == list:
#         n = len(y)
#     else:
#         n = 1
#
#     # print model.inputs[0]._keras_shape, model.targets[0]._keras_shape
#     # print 'model.targets', model.targets[0:n]
#     # print 'model.inputs[0]', model.inputs[0]
#     input_tensors = [model.inputs[0],  # input data
#                      # model.sample_weights[0],  # how much to weight each sample by
#                      # model.targets[0:n],  # labels
#                      # model.targets[0],  # labels
#                      # K.learning_phase(),  # train or test mode
#                      ]
#
#     for i in range(n):
#         input_tensors.append(model.sample_weights[i])
#
#     for i in range(n):
#         input_tensors.append(model.targets[i])
#
#     input_tensors.append(K.learning_phase())
#     gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)
#
#
#     get_gradients = K.function(inputs=input_tensors, outputs=[gradients])
#
#
#     # https: // github.com / fchollet / keras / issues / 2226
#     # print 'y_train', y_train.shape
#
#     nb_sample = X.shape[0]
#
#     # if type(y ) ==list:
#     #     y= [yy.reshape((nb_sample, 1)) for yy in y]
#     #     sample_weights = [np.ones(nb_sample) for i in range(n)]
#     # else:
#     #     y = y.reshape((nb_sample, 1))
#     #     sample_weights = np.ones(nb_sample)
#
#     inputs = [X,  # X
#               # sample_weights,  # sample weights
#               # y,  # y
#               # 0  # learning phase in TEST mode
#               ]
#
#     for i in range(n):
#         inputs.append(np.ones(nb_sample))
#
#     if n>1 :
#         for i in range(n):
#             inputs.append(y[i].reshape((nb_sample, 1)))
#     else:
#         inputs.append(y.reshape(nb_sample, 1))
#
#     inputs.append(0)# learning phase in TEST mode
#     # print(X.shape)
#     # print (y.shape)
#
#     # inputs = [X,  # X
#     #           sample_weights,  # sample weights
#     #           y,  # y
#     #           0  # learning phase in TEST mode
#     #           ]
#     # print weights
#     gradients = get_gradients(inputs)[0]
#
#     return gradients


def get_gradient_layer(model, X, y, layer, normalize=True):
    grad = model.optimizer.get_gradients(model.total_loss, layer)
    gradients = layer * grad  # gradient tensors

    # special case of repeated outputs (e.g. output for each hidden layer)
    if type(y) == list:
        n = len(y)
    else:
        n = 1
    input_tensors = [model.inputs[0],  # input data
                     # model.sample_weights[0],  # how much to weight each sample by
                     # model.targets[0:n],  # labels
                     # model.targets[0],  # labels
                     # K.learning_phase(),  # train or test mode
                     ]

    # how much to weight each sample by
    for i in range(n):
        input_tensors.append(model.sample_weights[i])
    # labels
    for i in range(n):
        input_tensors.append(model.targets[i])
    # train or test mode
    input_tensors.append(K.learning_phase())
    # normalize
    if normalize:
        gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)

    get_gradients = K.function(inputs=input_tensors, outputs=[gradients])
    # https: // github.com / fchollet / keras / issues / 2226
    # print 'y_train', y_train.shape

    nb_sample = X.shape[0]

    inputs = [X,  # X
              # sample_weights,  # sample weights
              # y,  # y
              # 0  # learning phase in TEST mode
              ]

    for i in range(n):
        inputs.append(np.ones(nb_sample))

    if n > 1:
        for i in range(n):
            inputs.append(y[i].reshape((nb_sample, 1)))
    else:
        inputs.append(y.reshape(nb_sample, 1))

    inputs.append(0)  # learning phase in TEST mode
    gradients = get_gradients(inputs)[0][0]
    return gradients


def get_shap_scores_layer(model, X, layer_name, output_index=-1, method_name='deepexplainer'):
    # local_smoothing ?
    # ranked_outputs
    def map2layer(model, x, layer_name):
        fetch = model.get_layer(layer_name).output
        feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
        return K.get_session().run(fetch, feed_dict)

    import shap
    if type(output_index) == str:
        y = model.get_layer(output_index).output
    else:
        y = model.outputs[output_index]

    x = model.get_layer(layer_name).output
    if method_name == 'deepexplainer':
        explainer = shap.DeepExplainer((x, y), map2layer(model, X.copy(), layer_name))
        shap_values, indexes = explainer.shap_values(map2layer(model, X, layer_name), ranked_outputs=2)
    elif method_name == 'gradientexplainer':
        explainer = shap.GradientExplainer((x, y), map2layer(model, X.copy(), layer_name), local_smoothing=2)
        shap_values, indexes = explainer.shap_values(map2layer(model, X, layer_name), ranked_outputs=2)
    else:
        raise ('unsppuorted method')

    print (shap_values[0].shape)
    return shap_values[0]


# model, X_train, y_train, target, detailed=detailed, method_name=method
def get_shap_scores(model, X_train, y_train, target=-1, method_name='deepexplainer', detailed=False):
    gradients_list = []
    gradients_list_sample_level = []
    i = 0
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )
            if target is None:
                output = i
            else:
                output = target
            print 'layer # {}, layer name {},  output name {}'.format(i, l.name, output)
            i += 1
            # gradients = get_deep_explain_score_layer(model, X_train, l.name, output, method_name= method_name )
            gradients = get_shap_scores_layer(model, X_train, l.name, output, method_name=method_name)
            # getting average score
            if gradients.ndim > 1:
                # feature_weights = np.sum(np.abs(gradients), axis=-2)
                feature_weights = np.sum(gradients, axis=-2)
            else:
                feature_weights = np.abs(gradients)
            gradients_list.append(feature_weights)
            gradients_list_sample_level.append(gradients)
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list
    pass


def get_deep_explain_scores(model, X_train, y_train, target=-1, method_name='grad*input', detailed=False, **kwargs):
    # gradients_list = []
    # gradients_list_sample_level = []

    gradients_list = {}
    gradients_list_sample_level = {}

    i = 0
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )

            if target is None:
                output = i
            else:
                output = target

            print 'layer # {}, layer name {},  output name {}'.format(i, l.name, output)
            i += 1
            gradients = get_deep_explain_score_layer(model, X_train, l.name, output, method_name=method_name)
            if gradients.ndim > 1:
                # feature_weights = np.sum(np.abs(gradients), axis=-2)
                # feature_weights = np.sum(gradients, axis=-2)
                print 'gradients.shape', gradients.shape
                # feature_weights = np.abs(np.sum(gradients, axis=-2))
                feature_weights = np.sum(gradients, axis=-2)
                # feature_weights = np.mean(gradients, axis=-2)
                print 'feature_weights.shape', feature_weights.shape
                print 'feature_weights min max', min(feature_weights), max(feature_weights)
            else:
                # feature_weights = np.abs(gradients)
                feature_weights = gradients
                # feature_weights = np.mean(gradients)
            # gradients_list.append(feature_weights)
            # gradients_list_sample_level.append(gradients)
            gradients_list[l.name] = feature_weights
            gradients_list_sample_level[l.name] = gradients
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list
    pass


def get_deep_explain_score_layer(model, X, layer_name, output_index=-1, method_name='grad*input'):
    scores = None
    import keras
    from deepexplain.tensorflow_ import DeepExplain
    import tensorflow as tf
    ww = model.get_weights()
    with tf.Session() as sess:
        try:
            with DeepExplain(session=sess) as de:  # <-- init DeepExplain context
                # Need to reconstruct the graph in DeepExplain context, using the same weights.
                # model= nn_model.model
                print layer_name
                model = keras.models.clone_model(model)
                model.set_weights(ww)
                # if layer_name=='inputs':
                #     layer_outcomes= X
                # else:
                #     layer_outcomes = nn_model.get_layer_output(layer_name, X)[0]

                x = model.get_layer(layer_name).output
                # x = model.inputs[0]
                if type(output_index) == str:
                    y = model.get_layer(output_index).output
                else:
                    y = model.outputs[output_index]

                # y = model.get_layer('o6').output
                # x = model.inputs[0]
                print layer_name
                print 'model.inputs', model.inputs
                print 'model y', y
                print 'model x', x
                attributions = de.explain(method_name, y, x, model.inputs[0], X)
                print 'attributions', attributions.shape
                scores = attributions
                return scores
        except:
            sess.close()
            print("Unexpected error:", sys.exc_info()[0])
            raise


def get_skf_weights(model, X, y, importance_type):
    from features_processing.feature_selection import FeatureSelectionModel
    layers = get_layers(model)
    inp = model.input
    layer_weights = []
    for i, l in enumerate(layers):

        if type(l) == InputLayer:
            layer_out = X
        elif l.name.startswith('h'):
            out = l.output
            print i, l, out
            func = K.function([inp] + [K.learning_phase()], [out])
            layer_out = func([X, 0.])[0]
        else:
            continue

        if type(y) == list:
            y = y[0]

        # layer_out = StandardScaler().fit_transform(layer_out)
        p = {'type': importance_type, 'params': {}}
        fs_model = FeatureSelectionModel(p)
        fs_model = fs_model.fit(layer_out, y.ravel())
        fs_coef = fs_model.get_coef()
        fs_coef[fs_coef == np.inf] = 0
        layer_weights.append(fs_coef)
    return layer_weights


def get_gradient_weights(model, X, y, signed=False, detailed=False, normalize=True):
    gradients_list = []
    gradients_list_sample_level = []
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )
            w = l.get_output_at(0)
            gradients = get_gradient_layer(model, X, y, w, normalize)
            if gradients.ndim > 1:
                if signed:
                    feature_weights = np.sum(gradients, axis=-2)
                else:
                    feature_weights = np.sum(np.abs(gradients), axis=-2)

            else:
                feature_weights = np.abs(gradients)
            gradients_list.append(feature_weights)
            gradients_list_sample_level.append(gradients)
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list


def get_gradient_weights_with_repeated_output(model, X, y):
    gradients_list = []
    # print 'trainable weights',model.trainable_weights
    # print 'layers', get_layers (model)

    for l in get_layers(model):

        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue

        # print 'get the gradient of layer {}'.format(l.name)
        if l.name.startswith('o') and not l.name.startswith('o_'):
            print l.name
            print l.weights
            weights = l.get_weights()[0]
            # weights = l.get_weights()
            # print 'weights shape {}'.format(weights.shape)
            gradients_list.append(weights.ravel())

    return gradients_list


# get weights of each layer based on training a linear model that predicts the outcome (y) given the layer output
def get_weights_linear_model(model, X, y):
    weights = None
    layer_weights = []
    layers = get_layers(model)
    inp = model.input
    for i, l in enumerate(layers):
        if type(l) in [Sequential, Dropout]:
            continue
        print  type(l)
        if type(l) == InputLayer:
            layer_out = X
        else:
            out = l.output
            print i, l, out
            func = K.function([inp] + [K.learning_phase()], [out])
            layer_out = func([X, 0.])[0]
        # print layer_out.shape
        # layer_outs.append(layer_out)
        linear_model = LogisticRegression(penalty='l1')
        # linear_model = LinearRegression( )
        # layer_out = StandardScaler().fit_transform(layer_out)
        if type(y) == list:
            y = y[0]
        linear_model.fit(layer_out, y.ravel())
        # print 'layer coef  shape ', linear_model.coef_.T.ravel().shape
        layer_weights.append(linear_model.coef_.T.ravel())
    return layer_weights


# def get_weights_gradient_outcome(model, x_train, y_train):
#     if type(y_train) == list:
#         n = len(y_train)
#     else:
#         n = 1
#     nb_sample = x_train.shape[0]
#     sample_weights = np.ones(nb_sample)
#     print model.output
#     # output = model.output[-1]
#     # model = nn_model.model
#     input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
#     # input_tensors = model.inputs + model.targets + [K.learning_phase()]
#     layers = get_layers(model)
#     gradients_list= []
#     i=0
#     for l in layers:
#         if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )
#             print i
#             output = model.output[i]
#             i+=1
#             print i, l.name, output.name, output, l.get_output_at(0)
#             # gradients = K.gradients(K.mean(output), l.get_output_at(0))
#             gradients = K.gradients(output, l.get_output_at(0))
#             # w= l.get_output_at(0)
#             # gradients = [w*g for g in K.gradients(output, w)]
#             get_gradients = K.function(inputs=input_tensors, outputs=gradients)
#             inputs = [x_train] + [sample_weights] * n + y_train  + [0]
#             gradients = get_gradients(inputs)
#             print 'gradients',len(gradients), gradients[0].shape
#             g= np.sum(np.abs(gradients[0]), axis = 0)
#             g= np.sum(gradients[0], axis = 0)
#             g= np.abs(g)
#             print 'gradients', gradients[0].shape
#             gradients_list.append(g)
#
#     return gradients_list
#

def get_gradeint(model, x, y, x_train, y_train, multiply_by_input=False):
    n_outcomes = 1
    if type(y_train) == list:
        n_outcomes = len(y_train)
    n_sample = x_train.shape[0]
    sample_weights = np.ones(n_sample)
    input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
    if multiply_by_input:
        gradients = [x * g for g in K.gradients(y, x)]
    else:
        gradients = K.gradients(y, x)
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    inputs = [x_train] + [sample_weights] * n_outcomes + y_train + [0]
    gradients = get_gradients(inputs)
    return gradients


def get_weights_gradient_outcome(model, x_train, y_train, detailed=False, target=-1, multiply_by_input=False,
                                 signed=True):
    print model.output
    layers = get_layers(model)
    gradients_list = []
    gradients_list_sample_level = []
    i = 0
    for l in layers:
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just an ad hoc convention )

            if target is None:
                output = model.output[i]
            else:
                if type(target) == str:
                    output = model.get_layer(target).output
                else:
                    output = model.outputs[target]

            print 'layer # {}, layer name {},  output name {}'.format(i, l.name, output.name)
            i += 1
            print i, l.name, output.name, output, l.get_output_at(0)
            gradients = get_gradeint(model, l.output, output, x_train, y_train, multiply_by_input=multiply_by_input)

            print 'gradients', len(gradients), gradients[0].shape
            if signed:
                g = np.sum(gradients[0], axis=0)
            else:
                g = np.sum(np.abs(gradients[0]), axis=0)
            # g = np.abs(g)
            gradients_list_sample_level.append(gradients[0])
            print 'gradients', gradients[0].shape
            gradients_list.append(g)

    if detailed:
        return gradients_list, gradients_list_sample_level

    return gradients_list


# def get_gradient_weights(model, X, y):
#     gradients_list = []
#     print 'trainable weights',model.trainable_weights
#     print 'layers', model.layers
#
#     # for l in get_layers(model):
#     # for l in [model.inputs[0] ]+ model.trainable_weights:
#     # c = get_gradient_layer(model, X, y, model.inputs[0])
#     # gradients_list.append(np.mean(c, axis=0))
#     for l in  model.trainable_weights:
#         # print l
#         # l = l.trainable_weights
#         # layer = model.inputs[0]
#         # print  ,
#
#         # if type(l) == InputLayer:
#         #     w = model.inputs[0]
#         # # elif type(l)==Sequential:
#         # #     continue
#         # elif hasattr(l, 'kernel') and type(l) != SpraseLayer:
#         #     w= l.output
#         # else: continue
#
#         if 'kernel' in str(l):
#
#             gradients = get_gradient_layer(model, X, y, l)
#             if gradients.ndim >1:
#                 feature_weights = np.mean(gradients, axis=1)
#             else:
#                 feature_weights = gradients
#             # feature_weights= gradients
#             print 'layer {} grdaient shape {}', l, feature_weights.shape
#             gradients_list.append(feature_weights)
#     return gradients_list

def get_permutation_weights(model, X, y):
    scores = []
    prediction_scores = predict(model, X)
    # print y
    # print prediction_scores
    baseline_acc = accuracy_score(y[0], prediction_scores)
    rnd = np.random.random((X.shape[0],))
    x_original = X.copy()
    for i in range(X.shape[1]):
        # if (i%100)==0:
        print i
        # x = X.copy()
        x_vector = x_original[:, i]
        # np.random.shuffle(x[:, i])
        x_original[:, i] = rnd
        acc = accuracy_score(y[0], predict(model, x_original))
        x_original[:, i] = x_vector
        scores.append((baseline_acc - acc) / baseline_acc)
    return np.array(scores)


def get_deconstruction_weights(model):
    for layer in model.layers:
        # print layer.name
        weights = layer.get_weights()  # list of numpy arrays
        # for w in weights:
        #     print w.shape
    pass
