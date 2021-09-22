import itertools
from os.path import join

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config_path import PROSTATE_LOG_PATH
from setup import saving_dir


def plot_(primary):
    percent = 100 * primary / sum(primary)
    labels = ['True prediction', 'False prediction']
    xpos = [0, 0.4]
    width = [0.3, 0.3]
    fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
    fig.set_size_inches(4, 5)
    ax = axes
    plt.bar(xpos, percent, align='center', alpha=.4, color=['green', 'red'], width=width)
    plt.bar(xpos, percent, align='center', alpha=.4, color=['green', 'red'], width=width)

    plt.xticks(xpos, labels)
    plt.ylim([0, 100])
    # for item in [fig, ax]:
    #     item.patch.set_visible(False)
    ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        if i.get_height() > 1.0:
            ax.text(i.get_x() + 0.3 * i.get_width(), i.get_height() + 5.0, '{:5.1f} %'.format(i.get_height()),
                    fontsize=15,
                    color='black', rotation=0)


# https://stackoverflow.com/questions/33371939/calculate-rgb-equivalent-of-base-colors-with-alpha-of-0-5-over-white-background
def make_rgb_transparent(rgb, bg_rgb, alpha):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]


def plot_stacked(ax, filename, correct, wrong):
    percent1 = 100. * correct / sum(correct)
    percent2 = 100. * wrong / sum(wrong)
    labels = ['Fraser et al.', 'Robinson et al.']
    #     xpos= [0,0.4]
    xpos = [0, 2]
    width = [1.0, 1.0]

    print percent1
    print percent2
    bottom = [percent1[0], percent2[0]]
    top = [percent1[1], percent2[1]]
    ORANGE = (255 / 256., 127 / 256., 0, 1.0)
    BLUE = (55 / 256., 126 / 256., 184 / 256., 1.0)
    LIGHT_BLUE = (0.65098041296005249, 0.80784314870834351, 0.89019608497619629, 1.0)
    #     colors = [BLUE, ORANGE]
    BLACK = [0, 0, 0, 1.0]
    RED = [1., 0, 0, 1.0]
    # colors = ['green', 'red']
    # colors= ["#2ecc71", "#e74c3c"]
    selected_colors = ["#34495e", "#e74c3c"]
    alpha = 0.7
    selected_colors = [colors.colorConverter.to_rgb(c) for c in selected_colors]
    selected_colors = [make_rgb_transparent(rgb, (1, 1, 1), alpha) for rgb in selected_colors]

    # colors = [BLACK, RED]

    #     colors = [GREEN, ORANGE]
    p1 = ax.bar(xpos, bottom, align='center', color=[selected_colors[0]] * 2, width=width)
    p2 = ax.bar(xpos, top, bottom=bottom, align='center', color=[selected_colors[1]] * 2, width=width)
    ax.grid()
    ax.set_ylim([0, 100])

    ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # plt.yticks([])
    # plt.ylim((0, 130))

    bottom_strs = ['%1.2f' % i + '%' for i in bottom]
    top_strs = ['%1.2f' % i + '%' for i in top]

    table = ax.table(cellText=[bottom_strs, top_strs],
                     rowLabels=['True Ra   te', 'Error Rate'],
                     rowColours=selected_colors,
                     colLabels=labels,
                     loc='bottom', fontsize=14, cellLoc='center')

    # plt.subplots_adjust(left=0.3, bottom=0.2)

    # iterate through cells of a table
    table_props = table.properties()
    table_cells = table_props['child_artists']
    for i, cell in enumerate(table_cells):
        print i
        if i in [6]:
            cell.get_text().set_fontsize(20)
            cell.get_text().set_color('w')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.3)


#     saving_filename = join(dirname(filename), 'BCR_distribution.png')
#     print saving_filename
#     plt.savefig(saving_filename)

# primary = np.array([95,35])


def plot_external_validation_all(ax):
    plot_stacked(ax, '', primary, mets)


def plot_external_validation_matrix(ax):
    normalize = True
    # labels = np.array([['TR', 'TR'], ['ER ', 'ER']])
    labels = np.array([['TN', 'FP'], ['FN ', 'TP']])
    cmap = plt.cm.Reds
    cm = np.array([primary, mets])

    if normalize:
        cm = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    classes = ['{}\n{}'.format('Fraser et al.', '(localized)'), '{}\n{}'.format('Robinson et al.', '(metastatic)')]
    cm = cm.T
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig = plt.gcf()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='10%', pad=0.1)

    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position("left")
    cb.outline.set_visible(False)
    tick_marks = np.arange(len(classes))

    if labels is None:
        fmt = '{:.2f}%' if normalize else '{:d}'
    else:
        fmt = '{}: {:.2f}%' if normalize else '{}: {:d}'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = fmt.format(labels[i, j], cm[i, j])
        ax.text(j, i, text,
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=12)

    fontproperties = {'family': 'Arial', 'weight': 'bold', 'size': 14}

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticks([])
    ax2 = divider.append_axes('bottom', size='5%', pad=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.set_yticks([])
    ax2.set_xticks([])


dir_name = join(PROSTATE_LOG_PATH, 'external_validation/pnet_validation')
primary_filename = join(dir_name, 'P-net__primary_testing.csv')
met_filename = join(dir_name, 'P-net__mets_testing.csv')
primary_df = pd.read_csv(primary_filename)
met_df = pd.read_csv(met_filename)
primary = [sum(primary_df.pred == False), sum(primary_df.pred == True)]
mets = [sum(met_df.pred == False), sum(met_df.pred == True)]
primary = np.array(primary)
mets = np.array(mets)


def run_externnal_validation():
    plot_(primary)
    filename = join(saving_dir, 'external_validation_primary.png')
    plt.savefig(filename, dpi=600)

    plot_(mets)
    filename = join(saving_dir, 'external_validation_mets.png')
    plt.savefig(filename, dpi=600)

    fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
    fig.set_size_inches(5, 7)
    ax = axes
    plot_stacked(ax, '', primary, mets)
    filename = join(saving_dir, 'external_validation_all.png')
    plt.savefig(filename, dpi=600)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.subplots(1, 1)
    plot_external_validation_matrix(ax)
    filename = join(saving_dir, 'external_validation_matrix.png')
    plt.savefig(filename, dpi=200)


if __name__ == '__main__':
    # primary = np.array([95,35])
    # mets = np.array([72,23])

    run_externnal_validation()
