import random

import numpy as  np


# from theano.tensor.shared_randomstreams import RandomStreams

def set_random_seeds(random_seed):
    # srng = RandomStreams(seed=random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
