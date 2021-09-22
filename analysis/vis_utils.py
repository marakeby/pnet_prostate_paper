from os.path import join

import pandas as pd

from config_path import *


def get_reactome_pathway_names():
    '''
    :return: dataframe of pathway ids and names
    '''
    reactome_pathways_df = pd.read_csv(
        join(REACTOM_PATHWAY_PATH, 'ReactomePathways.txt'), sep='	',
        header=None)
    reactome_pathways_df.columns = ['id', 'name', 'species']
    reactome_pathways_df_human = reactome_pathways_df[reactome_pathways_df['species'] == 'Homo sapiens']
    reactome_pathways_df_human.reset_index(inplace=True)
    return reactome_pathways_df_human


def get_x_y(df_encoded, layers_nodes):
    '''
    :param df_encoded: datafrme with columns (source  target  value) representing the network
    :param layers_nodes: data frame with index (nodes ) and one columns (layer) representing the layer to which this node belongs
    :return: x, y positions onf each node
    '''
    # node_id = range(len(layers_nodes))
    # node_weights = pd.DataFrame([node_id, layers_nodes], columns=['node_id', 'node_name'])
    # print node_weights
    # node weight is the max(sum of input edges, sum of output edges)
    source_weights = df_encoded.groupby(by='source')['value'].sum()
    target_weights = df_encoded.groupby(by='target')['value'].sum()

    node_weights = pd.concat([source_weights, target_weights])
    node_weights = node_weights.to_frame()
    node_weights = node_weights.groupby(node_weights.index).max()
    print layers_nodes
    print node_weights

    # print node_weights
    node_weights = node_weights.join(layers_nodes)
    node_weights.sort_values(by=['layer', 'value'], ascending=False, inplace=True)
    n_layers = layers_nodes['layer'].max() + 1
    node_weights['x'] = node_weights['layer'] / n_layers
    print node_weights

    node_weights['layer_weight'] = node_weights.groupby('layer')['value'].transform(pd.Series.sum)
    node_weights['y'] = node_weights.groupby('layer')['value'].transform(pd.Series.cumsum)

    node_weights['y'] = (node_weights['y'] - node_weights['value'] / 2) / node_weights['layer_weight'] + 0.05

    print node_weights
    node_weights.sort_index(inplace=True)
    return node_weights['x'], node_weights['y']


def get_data_trace(encoded_top_genes, all_node_labels, layers, node_colors=None):
    x, y = get_x_y(encoded_top_genes, layers)
    data_trace = dict(
        type='sankey',
        domain=dict(
            x=[0, 1],
            y=[0, 1]
        ),
        orientation="h",
        valueformat=".0f",
        node=dict(
            pad=15,
            thickness=30,
            line=dict(
                color="black",
                width=0.7
            ),
            label=all_node_labels,
            #       label =  ['all_node_labels'] *15,

            # color = '#262C46'
            x=x,
            y=y,
            color=node_colors if node_colors else None,
        ),
        link=dict(
            source=encoded_top_genes['source'],
            target=encoded_top_genes['target'],
            value=encoded_top_genes['value'],
            # color=encoded_top_genes['color']

        )
    )

    layout = dict(
        title="Neural Network Architecture",
        height=800,
        width=1400,
        font=dict(family="Arial", size=20
                  # size=16,
                  # type='bold'
                  )
        # updatemenus=[dict(
        #     y=0.6,
        #     buttons=[
        #         dict(
        #             label='Horizontal',
        #             method='restyle',
        #             args=['orientation', 'h']
        #         ),
        #         dict(
        #             label='Vertical',
        #             method='restyle',
        #             args=['orientation', 'v']
        #         )
        #     ]

        # )]
    )
    return data_trace, layout
