import logging
from os.path import join


def save_prediction(directory, info, y_pred, y_test, model_name, training=False):
    if training:
        file_name = join(directory, model_name + '_traing.csv')
    else:
        file_name = join(directory, model_name + '_testing.csv')
    logging.info("saving : %s" % file_name)
    info['pred'] = y_pred
    info['y'] = y_test
    info.to_csv(file_name)
