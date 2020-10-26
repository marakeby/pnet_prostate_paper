import logging
import os
import sys


def set_logging(filename):
    # base_folder ='../logs'
    # timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())

    # filename, file_extension = os.path.splitext(filename)
    # filename = os.path.join(base_folder,filename+ timeStamp+'.log')
    if not os.path.exists(filename):
        os.makedirs(filename)

    filename = os.path.join(filename, 'log.log')
    logging.basicConfig(filename=filename,
                        filemode='w',
                        format='%(asctime)s - {%(filename)s:%(lineno)d} - %(message)s',
                        datefmt='%m/%d %I:%M',
                        level=logging.INFO)  # or logging.DEBUG
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('setting logs')


debug_folder = '.'


class DebugFolder():

    def __init__(self, folder=None):
        self.set_debug_folder(folder)

    def get_debug_folder(self):
        global debug_folder
        return debug_folder

    def set_debug_folder(self, folder):
        if not folder is None:
            global debug_folder
            debug_folder = folder
