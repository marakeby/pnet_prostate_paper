from os.path import join

from setup import saving_dir

nodes = [27687,
         9229,
         1387,
         1066,
         447,
         147,
         26]

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def run():
    fig = plt.figure(figsize=(6, 4), dpi=200)

    df = pd.DataFrame(nodes, index=['layer 0', 'layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'layer 6'],
                      columns=['Number of nodes'])
    print  df
    ax = sns.barplot(x=df.index, y='Number of nodes', data=df)
    ax.set_yscale("log")

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        if i.get_height() > 1.0:
            ax.text(i.get_x() + 0.1 * i.get_width(), i.get_height() + .3 * i.get_height(),
                    '{:5.0f}'.format(i.get_height()), fontsize=10,
                    color='dimgrey', rotation=0)

    ax.set_ylabel('Number of nodes', fontdict=dict(family='Arial', weight='bold', fontsize=12))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.subplots_adjust(left=0.25)
    plt.savefig(join(saving_dir, 'layer_nodes.png'))
    # plt.show()


if __name__ == "__main__":
    run()
