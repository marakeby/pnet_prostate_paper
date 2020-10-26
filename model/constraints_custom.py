# import theano.tensor as T
# import theano
import numpy as np
from keras import backend as K
from keras.constraints import Constraint


class OneToOne(Constraint):
    def __call__(self, p):
        # p = theano.shared(p)
        # n, m =p.shape
        # print n, m
        # p *= T.identity_like(p)
        # TODO: review the identity_like funciton
        p *= K.identity(p)
        return p


class ConnectionConstaints(Constraint):
    def __init__(self, m):
        self.connection_map = m
        # print type(self.connection_map)
        print 'building a kernel constraint '

    # def __init__(self, xx):
    #     # self.connection_map =xxw
    #     self.connection_map = np.eye(200)
    #     print type(self.connection_map)
    def __call__(self, p):
        # p = theano.shared(p)
        # n, m =p.shape
        # print n, m
        mapp = np.array(self.connection_map)
        # p *= mapp.astype(theano.config.floatX)
        p *= mapp.astype(K.floatx())
        # p = K.abs(p)
        return p

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "map": self.connection_map
        }


class ZeroWeights(Constraint):
    def __call__(self, p):
        # p = theano.shared(p)
        # n, m =p.shape
        # print n, m
        # p = T.zeros_like(p)
        p = K.zeros_like(p)
        return p


class OneWeights(Constraint):
    def __call__(self, p):
        # p = theano.shared(p)
        # n, m =p.shape
        # print n, m
        # p = T.ones_like(p)
        p = K.ones_like(p)
        return p
