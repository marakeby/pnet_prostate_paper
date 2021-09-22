import itertools
import networkx as nx
import numpy as np
import pandas as pd

from reactome import Reactome, ReactomeNetwork

reactome = Reactome()
names_df = reactome.pathway_names
hierarchy_df = reactome.hierarchy
genes_df = reactome.pathway_genes

print (names_df.head())
print (hierarchy_df.head())
print (genes_df.head())

reactome_net = ReactomeNetwork()
print reactome_net.info()

print '# of root nodes {} , # of terminal nodes {}'.format(len(reactome_net.get_roots()),
                                                           len(reactome_net.get_terminals()))
print nx.info(reactome_net.get_completed_tree(n_levels=5))
print nx.info(reactome_net.get_completed_network(n_levels=5))
layers = reactome_net.get_layers(n_levels=3)
print len(layers)


def get_map_from_layer(layer_dict):
    '''
    :param layer_dict: dictionary of connections (e.g {'pathway1': ['g1', 'g2', 'g3']}
    :return: dataframe map of layer (index = genes, columns = pathways, , values = 1 if connected; 0 else)
    '''
    pathways = layer_dict.keys()
    genes = list(itertools.chain.from_iterable(layer_dict.values()))
    genes = list(np.unique(genes))
    df = pd.DataFrame(index=pathways, columns=genes)
    for k, v in layer_dict.items():
        df.loc[k, v] = 1
    df = df.fillna(0)
    return df.T


for i, layer in enumerate(layers[::-1]):
    mapp = get_map_from_layer(layer)
    if i == 0:
        genes = list(mapp.index)[0:5]
    filter_df = pd.DataFrame(index=genes)
    all = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
    genes = list(mapp.columns)
    print all.shape
