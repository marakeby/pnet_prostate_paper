import logging
import numpy as np
import copy
from sklearn import preprocessing as p


def get_processor(args):
    print args
    proc_type = args['type']
    logging.info("Pre-processing: %s", proc_type)
    if proc_type == 'standard':  # 0 mean , 1 variance
        if 'params' in args:
            p1 = args['params']
            proc = p.StandardScaler(**p1)
        else:
            proc = p.StandardScaler()
    elif proc_type == 'normalize':  # 1 norm
        proc = p.Normalizer()

    elif proc_type == 'scale':  # 0:1 scale
        if 'params' in args:
            p1 = args['params']
            proc = p.MinMaxScaler(**p1)
        else:
            proc = p.MinMaxScaler()
    elif proc_type == 'tfidf':
        from sklearn.feature_extraction.text import TfidfTransformer

        p1 = args['params']
        proc = TfidfTransformer(**p1)
        print p1

    else:
        proc = None

    return proc


def remove_outliers(y):
    m = np.mean(y)
    s = np.std(y)
    y2 = copy.deepcopy(y)
    s = np.std(y)
    n = 4
    print n
    y2[y > m + n * s] = m + n * s
    y2[y < m - n * s] = m - n * s
    print min(y2), max(y2), np.mean(y2)
    return y2
