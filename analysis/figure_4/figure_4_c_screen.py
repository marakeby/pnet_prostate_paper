from os.path import join

import matplotlib.gridspec as gridspec
import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt

# https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib
from config_path import PROSTATE_DATA_PATH
from setup import saving_dir


def run():
    base_dir = PROSTATE_DATA_PATH
    filename = join(base_dir, 'supporting_data/Z score list in all conditions.xlsx')
    df = pd.read_excel(filename)
    df = df.set_index('gene symbol')
    df.head()
    df = df.groupby(by=df.index).max().sort_values('Z-LFC AVERAGE Enzalutimide')

    print df.head()
    print df.head(-5)
    fig = plt.figure(figsize=(4, 4))

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 9, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    x = range(df.shape[0])

    ax1.plot(x, df['Z-LFC AVERAGE Enzalutimide'], '.')
    ax2.plot(x, df['Z-LFC AVERAGE Enzalutimide'], '.')
    ax3.plot(x, df['Z-LFC AVERAGE Enzalutimide'], '.')

    ax3.set_ylim(-15.5, -10)
    ax2.set_ylim(-6, 6)
    ax1.set_ylim(10, 15)

    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    #
    ax1.yaxis.tick_left()
    ax1.tick_params(labelright='off')
    ax2.tick_params(labelright='off')
    ax3.tick_params(labelright='off')

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax2.set_ylabel('Z-score (CSS+enza)', fontdict=dict(weight='bold', fontsize=12))
    interesting = ['AR', 'TP53', 'PTEN', 'RB1', 'MDM4', 'FGFR1', 'MAML3', 'PDGFA', 'NOTCH1', 'EIF3E']

    texts = []
    #
    xy_dict = dict(TP53=(30, -4),
                   PTEN=(10, 20),
                   MDM4=(-30, 4),
                   FGFR1=(-10, 20),
                   MAML3=(30, -10),
                   # PDGFA=(),
                   NOTCH1=(30, 2),
                   EIF3E=(40, -2)

                   )
    direction = [-1, 1] * 5
    x = [0, 30, -30, 0, ]
    y = [0, -2, +4, 0]
    print direction
    for i, gene in enumerate(interesting):
        if gene in df.index:
            print gene
            ind = df.index.str.contains(gene)
            x = list(ind).index(True)
            y = df['Z-LFC AVERAGE Enzalutimide'][x]
            print gene, x, y
            # ax2.plot(x, y, 'r*')
            # ax2.text(x+170, y, gene, fontdict=dict( fontsize=8))
            xytext = (direction[i] * 30, -2)
            ax2.annotate(gene, (x, y), xycoords='data', fontsize=8,
                         bbox=dict(boxstyle="round", fc="none", ec="gray"),
                         xytext=xy_dict[gene], textcoords='offset points', ha='center',
                         arrowprops=dict(arrowstyle="->"))
            # texts.append(ax2.text(x, y, gene))

    adjust_text(texts)
    ax2.grid()
    plt.subplots_adjust(left=0.15)
    filename = join(saving_dir, 'screen.png')
    plt.savefig(filename, dpi=200)


if __name__ == "__main__":
    run()
