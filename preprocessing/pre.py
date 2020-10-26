# from preprocessing.smart import SmartPreprocesor

__author__ = 'marakeby'
import logging

import numpy as np
from sklearn import preprocessing as p


def get_processor(args):
    print args
    proc_type = args['type']
    logging.info("Pre-processing: %s", proc_type)
    # params = args['params']
    if proc_type == 'standard':  # 0 mean , 1 variance
        if 'params' in args:
            p1 = args['params']
            proc = p.StandardScaler(**p1)
        else:
            proc = p.StandardScaler()
    elif proc_type == 'normalize':  # 1 norm
        proc = p.Normalizer()

    # elif proc_type =='abs': #  1 norm
    #     proc = np.abs
    elif proc_type == 'scale':  # 0:1 scale
        if 'params' in args:
            p1 = args['params']
            proc = p.MinMaxScaler(**p1)
        else:
            proc = p.MinMaxScaler()

    elif proc_type == 'log':  # to be implemented
        proc = None  # TODO: implement log scaling
    elif proc_type == 'tissue-specific':
        from tissue_specefic import tissue_specific
        proc = tissue_specific()
    elif proc_type == 'smart':
        p1 = args['params']
        print p1
        proc = get_processor(p1)
        proc = SmartPreprocesor(proc)
    elif proc_type == 'tfidf':
        from sklearn.feature_extraction.text import TfidfTransformer

        p1 = args['params']
        proc = TfidfTransformer(**p1)
        print p1

    else:
        proc = None

    return proc


def remove_outliers(y):
    # print min(y), max(y), np.mean(y)
    m = np.mean(y)
    s = np.std(y)
    # print min(y), max(y), np.mean(y)
    # print 's', s
    # y = y-m
    import copy
    y2 = copy.deepcopy(y)

    # y2 = list(y)
    s = np.std(y)
    n = 4
    print n
    y2[y > m + n * s] = m + n * s
    y2[y < m - n * s] = m - n * s
    print min(y2), max(y2), np.mean(y2)
    return y2
