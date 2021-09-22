from os.path import join, realpath, dirname

BASE_PATH = dirname(realpath(__file__))
DATA_PATH = join(BASE_PATH, '_database')
GENE_PATH = join(DATA_PATH, 'genes')
PATHWAY_PATH = join(DATA_PATH, 'pathways')
REACTOM_PATHWAY_PATH = join(PATHWAY_PATH, 'Reactome')
PROSTATE_DATA_PATH = join(DATA_PATH, 'prostate')
RUN_PATH = join(BASE_PATH, 'train')
LOG_PATH = join(BASE_PATH, '_logs')
PROSTATE_LOG_PATH = join(LOG_PATH, 'p1000')
PARAMS_PATH = join(RUN_PATH, 'params')
POSTATE_PARAMS_PATH = join(PARAMS_PATH, 'P1000')
PLOTS_PATH = join(BASE_PATH, '_plots')
