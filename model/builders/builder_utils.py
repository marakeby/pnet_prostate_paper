import keras.backend as K
import theano
from keras.layers import TimeDistributed, RepeatVector, Permute

from model.layers_custom import AttLayer

theano.config.openmp = False
from keras.layers.core import Flatten, Reshape, Lambda
from keras.layers import Dense, merge


def get_similarity(similarity, params=None):
    ''' Specify similarity in configuration under 'similarity' -> 'mode'
    If a parameter is needed for the model, specify it in 'similarity'

    Example configuration:

    config = {
        ... other parameters ...
        'similarity': {
            'mode': 'gesd',
            'gamma': 1,
            'c': 1,
        }
    }

    cosine: dot(a, b) / sqrt(dot(a, a) * dot(b, b))
    polynomial: (gamma * dot(a, b) + c) ^ d
    sigmoid: tanh(gamma * dot(a, b) + c)
    rbf: exp(-gamma * l2_norm(a-b) ^ 2)
    euclidean: 1 / (1 + l2_norm(a - b))
    exponential: exp(-gamma * l2_norm(a - b))
    gesd: euclidean * sigmoid
    aesd: (euclidean + sigmoid) / 2
    '''

    dot = lambda a, b: K.batch_dot(a, b, axes=1)
    l2_norm = lambda a, b: K.sqrt(K.sum(K.square(a - b), axis=1, keepdims=True))

    if similarity == 'cosine':
        return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())
    elif similarity == 'polynomial':
        return lambda x: (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
    elif similarity == 'sigmoid':
        return lambda x: K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
    elif similarity == 'rbf':
        return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2)
    elif similarity == 'euclidean':
        return lambda x: 1 / (1 + l2_norm(x[0], x[1]))
    elif similarity == 'exponential':
        return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
    elif similarity == 'gesd':
        euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
        sigmoid = lambda x: 1 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
        return lambda x: euclidean(x) * sigmoid(x)
    elif similarity == 'aesd':
        euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
        sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
        return lambda x: euclidean(x) + sigmoid(x)
    else:
        raise Exception('Invalid similarity: {}'.format(similarity))


def get_attention_model_shared(p1_encoded_f, flate=True):
    p1_encoded_f = check_shape(p1_encoded_f)
    weights1 = TimeDistributed(Dense(1, activation='sigmoid'))(p1_encoded_f)

    attention = Flatten()(weights1)
    # attention = Activation('softmax')(attention)
    in_shape = p1_encoded_f._keras_shape
    attention = RepeatVector(in_shape[-1])(attention)
    attention = Permute([2, 1])(attention)

    weighted1 = merge([p1_encoded_f, attention], mode='mul')
    # weighted1 = merge([p1_encoded_f, attention], mode='dot', dot_axes=1)
    # weighted1= K.sum(weighted1, axis=1)

    from keras.layers import Lambda
    sum_dim1 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(in_shape[-1],))
    weighted1 = sum_dim1(weighted1)
    # if flate:
    #     flat = Flatten()
    #     weighted1 = flat(weighted1)
    return weighted1


def get_attention_model_not_shared(p1_encoded_f, flate=True):
    print ('p1_encoded_f._keras_shape', p1_encoded_f._keras_shape)
    # weighted = Attention()(p1_encoded_f)
    weighted = AttLayer()(p1_encoded_f)
    return weighted


def check_shape(p1_encoded_f):
    shape = p1_encoded_f._keras_shape[1:]
    if len(shape) < 2:
        shape = shape + (1,)
        p1_encoded_f = Reshape(shape)(p1_encoded_f)
    return p1_encoded_f


def get_interaction_model(p1_encoded_f, p2_encoded_f=None, flate=True):
    if p2_encoded_f == None:
        p2_encoded_f = p1_encoded_f

    p1_encoded_f = check_shape(p1_encoded_f)
    p2_encoded_f = check_shape(p2_encoded_f)

    print p1_encoded_f._keras_shape
    print p2_encoded_f._keras_shape

    match = merge([p1_encoded_f, p2_encoded_f], mode='dot', dot_axes=[2, 2])

    # if weighted:
    #     match = Flatten()(match)
    #     shape = match._keras_shape[1:]
    #     shape = shape + (1,)
    #     print shape
    #     match = Reshape(shape)(match)
    #     weights_dot_match = TimeDistributed(Dense(1))(match)
    #     match = merge([match, weights_dot_match], mode='mul')
    if flate:
        flat = Flatten()
        match = flat(match)

    return match


def apply_on_splits(model, p1_encoded_f, n_parts=2, name=''):
    # n_samples, max_len, input_dim = p1_encoded_f._keras_shape
    print 'p1_encoded_f._keras_shape', p1_encoded_f._keras_shape
    n_samples, max_len = p1_encoded_f._keras_shape

    step = max_len / n_parts
    xs1 = []
    for i in range(0, max_len, step):
        # part = i:i + step
        print i
        # split1 = Lambda(lambda x: x[:, i:i + step, :], output_shape=[max_len / n_parts, input_dim])
        split1 = Lambda(lambda x: x[:, i:i + step], output_shape=[max_len / n_parts])
        # split1 = Lambda(lambda x: x[:, :, i:i + step], output_shape=[max_len , axis /n_parts])
        x1 = split1(p1_encoded_f)
        x1 = model(x1)

        # n_samples, n_features = x1._keras_shape
        # x1 = Reshape((1, n_features), input_shape=(n_features,))(x1)
        xs1.append(x1)

    p1_encoded_f = merge(xs1, mode='concat', concat_axis=1, name=name)
    return p1_encoded_f
