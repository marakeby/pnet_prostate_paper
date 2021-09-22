from os.path import join

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from upsetplot import from_memberships
from upsetplot import plot

from config_path import PROSTATE_DATA_PATH, PLOTS_PATH
from data.data_access import Data


def plot_upset(ax):
    data = np.array([795., 27., 182., 7.])
    # plt.rcParams.update({'font.size': fontsize})
    example = from_memberships(
        [
            [' TP53 WT', ' MDM4 WT'],
            [' TP53 WT', ' MDM4 amp.'],
            [' TP53 mutant', ' MDM4 WT'],
            [' TP53 mutant', ' MDM4 amp.']],
        data=data
    )
    intersections, matrix, shading, totals = plot(example, with_lines=True, show_counts=True,
                                                  element_size=50)
    plt.ylabel('Number of patients', fontproperties)


def plot_screen(ax):
    base_dir = PROSTATE_DATA_PATH
    filename = join(base_dir, 'supporting_data/Z score list in all conditions.xlsx')
    df = pd.read_excel(filename)
    df = df.set_index('gene symbol')
    df.head()
    df = df.groupby(by=df.index).max().sort_values('Z-LFC AVERAGE Enzalutimide')

    divider = make_axes_locatable(ax)
    ax2 = ax
    ax1 = divider.append_axes('top', size='10%', pad=0.1)
    ax3 = divider.append_axes('bottom', size='10%', pad=0.1)

    x = range(df.shape[0])

    ax1.plot(x, df['Z-LFC AVERAGE Enzalutimide'], '.', markersize=0.5)
    ax2.plot(x, df['Z-LFC AVERAGE Enzalutimide'], '.', markersize=0.5)
    ax3.plot(x, df['Z-LFC AVERAGE Enzalutimide'], '.', markersize=0.5)

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

    ax2.set_ylabel('Z-score (CSS+enza)', fontproperties, labelpad=labelpad)
    interesting = ['AR', 'TP53', 'PTEN', 'RB1', 'MDM4', 'FGFR1', 'MAML3', 'PDGFA', 'NOTCH1', 'EIF3E']

    texts = []
    #
    xy_dict = dict(TP53=(30, -8),
                   PTEN=(10, 20),
                   MDM4=(-30, 4),
                   FGFR1=(0, 20),
                   MAML3=(30, -10),
                   # PDGFA=(),
                   NOTCH1=(30, -1),
                   EIF3E=(10, -15)

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
            ax2.annotate(gene, (x, y), xycoords='data', fontsize=fontsize,
                         bbox=dict(boxstyle="round", fc="none", ec="gray", linewidth=0.0),
                         xytext=xy_dict[gene], textcoords='offset points', ha='center',
                         arrowprops=dict(arrowstyle="->", linewidth=0.3))
            # texts.append(ax2.text(x, y, gene))

    adjust_text(texts)
    ax2.grid()
    ax1.yaxis.set_tick_params(labelsize=fontsize)
    ax2.yaxis.set_tick_params(labelsize=fontsize)
    ax3.yaxis.set_tick_params(labelsize=fontsize)

    ax1.tick_params(axis="y", direction="out", length=0, pad=pad)
    ax2.tick_params(axis="y", direction="out", length=0, pad=pad)
    ax3.tick_params(axis="y", direction="out", length=0, pad=pad)


def plot_stacked_hist_cnv(gene_name, ax):
    data_params = {'id': 'cnv', 'type': 'prostate_paper', 'params': {'data_type': 'cnv', 'drop_AR': False}}
    data = Data(**data_params)
    x, y, info, columns = data.get_data()
    df = pd.DataFrame(x, index=info, columns=list(columns))
    df['y'] = y

    ind = np.sort(df[gene_name].unique())
    print ind
    primary_df = df[df['y'] == 0]
    mets_df = df[df['y'] == 1]
    p = primary_df[gene_name].value_counts()
    m = mets_df[gene_name].value_counts()

    mapping = {0: 'Neutral', 1: 'Amplification', 2: 'High amplification', -1: 'Deletion', -2: 'Deep deletion'}

    index = pd.DataFrame(index=ind)
    summary = pd.concat([m, p], axis=1, keys=['Metastatic', 'Primary'])
    summary = summary.join(index, how='right')
    summary.fillna(0, inplace=True)
    summary = 100. * summary / summary.sum()

    summary = summary.rename(index=mapping)
    D_id_color = {'High amplification': 'maroon', 'Amplification': 'lightcoral', 'Neutral': 'gainsboro',
                  'Deletion': 'skyblue', 'Deep deletion': 'steelblue'}

    color = [D_id_color[i] for i in summary.index]

    print summary
    # bars = summary.T.plot.bar(stacked=True, ax=ax, color=color, width=0.25)
    # labels = summary.index
    # ax.bar(labels, men_means, 0.5,  label=labels[0],)
    # ax.bar(labels, women_means, 0.5,  bottom=men_means,label=labels[1])
    # fig, ax = plt.subplots()
    labels = summary.index
    bottoms = summary.cumsum()
    print bottoms
    x = [0.25, 0.75]
    for i, l in enumerate(labels):
        print l
        bottom = bottoms.loc[l, :] - summary.loc[l, :]
        ax.bar(x, summary.loc[l, :].values, label=l, color=D_id_color[l], width=0.3, bottom=bottom)
    print summary.columns
    ax.set_xticks(x)
    ax.set_xticklabels(list(summary.columns))
    ax.tick_params(axis='x', rotation=0, labelsize=fontsize)
    ax.tick_params(axis='y', rotation=0, labelsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=fontsize, bbox_to_anchor=(1., -0.1), framealpha=0.0)

    ax.set_ylabel('Sample percent (%)', fontproperties, labelpad=labelpad)
    ax.set_title('Copy number variations', fontproperties)


def plot_crispr(ax):
    colors_dict = {
        'sgGFP-3': '#848484',
        'sgGFP-4': '#DCDCDC',
        'sgMDM4-1': '#D58A85',
        'sgMDM4-2': '#6B1D14',

    }
    filename = join(PROSTATE_DATA_PATH,
                    'functional/revision/KD MDM4 proliferation raw data of 3 experiments in 4 cell lines in triplicates_procssed.xls')
    df = pd.read_excel(filename, sheet_name='processed2', header=[0], index_col=None)
    del df['Total cells']

    df_exp1 = df[df.Experiment == 'experiment 1']
    normalizing_frame = df_exp1[df_exp1['Sample ID'] == 'sgGFP-3'].groupby(['Cell line']).mean()
    normalized_df = df_exp1.join(normalizing_frame, on=['Cell line'], rsuffix='_base')
    normalized_df['Normalized Viable cells'] = normalized_df['Viable cells'] / normalized_df['Viable cells_base']

    x = "Cell line"
    y = "Normalized Viable cells"
    group = 'Sample ID'
    order = sorted(list(normalized_df[x].unique()))
    print order
    # Draw the bar chart
    ax = sns.barplot(
        data=normalized_df,
        x=x,
        y=y,
        order=order,
        hue=group,
        alpha=0.6,
        # ci='sd',
        ci=68,
        capsize=.1, errcolor='k', errwidth=0.5, ax=ax, palette=colors_dict
    )

    # Get the legend from just the bar chart
    handles, labels = ax.get_legend_handles_labels()

    # Draw the stripplot
    sns.stripplot(
        data=normalized_df,
        x=x,
        y=y,
        hue=group,
        order=order,
        dodge=True,
        edgecolor="black",
        linewidth=.2,
        ax=ax, size=1, palette=colors_dict
    )
    # Remove the old legend
    ax.legend_.remove()
    # Add just the bar chart legend back
    ax.legend(
        handles,
        labels,
        fontsize=fontsize,
        # loc=8,
        # bbox_to_anchor=(1.25, .5),
        bbox_to_anchor=(.5, 1.2), loc='upper center', ncol=2,
        framealpha=0.0
    )
    ax.set_xticklabels(order, fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
    ax.set_ylabel('Normalized Cell Count', fontproperties, labelpad=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_4d(ax):
    filename = join(PROSTATE_DATA_PATH,
                    'functional/revision/KD MDM4 proliferation raw data of 3 experiments in 4 cell lines in triplicates.xls')
    df = pd.read_excel(filename, sheet_name='process', header=[0], index_col=None)
    del df['Total cells']

    dd = df.groupby(['cell line', 'Sample ID']).agg([np.mean, np.std])
    dd.columns = dd.columns.droplevel(0)
    dd = dd.reset_index()

    dd = dd.set_index('cell line')
    normalizing_frame = dd[dd['Sample ID'] == 'sgGFP3']['mean'].to_frame()
    normalizing_frame.columns = ['normalize']

    normalized_df = dd.join(normalizing_frame)
    normalized_df['Normalized Cell Count'] = normalized_df['mean'] / normalized_df['normalize']
    colors = {
        'sgGFP3': '#848484',
        'sgGFP4': '#DCDCDC',
        'sgMDM4-1': '#D58A85',
        'sgMDM4-2': '#6B1D14',

    }

    def grouped_barplot(df, cat, subcat, val, err):
        u = df[cat].unique()
        x = np.arange(len(u))
        subx = df[subcat].unique()
        offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
        print df.shape
        print x
        print offsets
        width = np.diff(offsets).mean()
        for i, gr in enumerate(subx):
            print gr
            dfg = df[df[subcat] == gr]
            ax.bar(x + offsets[i], dfg[val].values, width=width,
                   label="{}".format(gr), yerr=dfg[err].values, capsize=1.5, color=colors[gr], alpha=0.7,
                   error_kw={'elinewidth': 0.5, 'capthick': 0.5})
            #     plt.xlabel(cat)

            yval = dfg[val].values
            xval = [x[i] + offsets[i]]
            xval = np.random.normal(xval, 0.04, len(yval))
            print yval
            print xval
            # xval = np.random.normal(xval, 0.04, len(yval))
            ax.scatter(xval, yval, marker='.', color='black', s=1.)

            #

            # ax.scatter(x, val, marker='.', color=colors[i], alpha=0.1)

        ax.set_ylabel(val, fontproperties, labelpad=0)
        ax.set_xticks(x)
        ax.set_xticklabels(u, rotation=0)
        # ax.set_xticks( u)
        # ax.legend(framealpha=0.0, bbox_to_anchor=(1., 1.15), ncol=4, prop={'size': fontsize})
        ax.legend(framealpha=0.0, bbox_to_anchor=(.5, 1.2), loc='upper center', ncol=2, prop={'size': fontsize})

    cat = "cell line"
    subcat = "Sample ID"
    val = "Normalized Cell Count"
    err = "std"
    print normalized_df.shape
    print normalized_df.reset_index().columns
    grouped_barplot(normalized_df.reset_index(), cat, subcat, val, err)
    # subx = df[subcat].unique()
    # for i, gr in enumerate(subx):

    # plt.legend( framealpha=0.0, loc='upper center', ncol=4, prop={'size': 6})
    # plt.ylim(0,1.5)
    # plt.show()
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)


def plot_crispr_sem(ax):
    colors_dict = {
        'sgGFP-3': '#848484',
        'sgGFP-4': '#DCDCDC',
        'sgMDM4-1': '#D58A85',
        'sgMDM4-2': '#6B1D14',

    }
    filename = join(PROSTATE_DATA_PATH,
                    'functional/revision/KD MDM4 proliferation raw data of 3 experiments in 4 cell lines in triplicates_procssed.xls')
    df = pd.read_excel(filename, sheet_name='processed2', header=[0], index_col=None)
    del df['Total cells']

    df_exp1 = df[df.Experiment == 'experiment 1']
    normalizing_frame = df_exp1[df_exp1['Sample ID'] == 'sgGFP-3'].groupby(['Cell line']).mean()
    normalized_df = df_exp1.join(normalizing_frame, on=['Cell line'], rsuffix='_base')
    normalized_df['Normalized Cell Count'] = normalized_df['Viable cells'] / normalized_df['Viable cells_base']

    # order= sorted(list(normalized_df[x].unique()))

    def grouped_barplot(df_in, cat_col, subcat_col, val_col):
        df = df_in.copy()
        cats = sorted(df[cat_col].unique())
        x = np.arange(len(cats))
        subcats = df[subcat_col].unique()
        offsets = (np.arange(len(subcats)) - np.arange(len(subcats)).mean()) / (len(subcats) + 1.)
        width = np.diff(offsets).mean()

        print  'offsets', offsets
        print  'offsets', x
        xval_list = []
        for i, c in enumerate(cats):
            for j, s in enumerate(subcats):
                xval_list.append({'cat': c, 'subcat': s, 'xval': x[i] + offsets[j]})

        jetter = np.random.uniform(low=-0.05, high=0.05, size=(len(xval_list)))
        print jetter
        xval_df = pd.DataFrame(xval_list).set_index(['cat', 'subcat'])
        xval_df.xval = xval_df.xval + jetter
        print xval_df
        df = df.join(xval_df, on=[cat_col, subcat_col], how='inner')
        print df.head()
        for i, gr in enumerate(subcats):
            inds = df[subcat_col] == gr
            group_df = df.loc[inds, :]
            means = group_df.groupby(cat)[val_col].mean()
            stds = group_df.groupby(cat)[val_col].sem()
            ax.bar(x + offsets[i], means, width=width, yerr=stds, label="{}".format(gr), capsize=1.5,
                   color=colors_dict[gr],
                   alpha=0.7, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
            xval = df['xval'].values
            yval = df[val_col].values
            ax.scatter(xval, yval, marker='o', facecolors='none', edgecolors='black', s=1.5, linewidths=0.2)

        ax.set_ylabel(val, fontproperties, labelpad=0)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=0, fontsize=fontsize)
        ax.legend(framealpha=0.0, bbox_to_anchor=(.5, 1.2), loc='upper center', ncol=2, prop={'size': fontsize})

    cat = "Cell line"
    subcat = "Sample ID"
    val = "Normalized Cell Count"
    print normalized_df.shape
    print normalized_df.reset_index().columns
    grouped_barplot(normalized_df, cat, subcat, val)

    ax.set_yticks([0, 0.5, 1.0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)


fontsize = 5  # legends, axis
fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}
pad = 2  # pad of ticks from axis
labelpad = 1


def run():
    saving_dir = join(PLOTS_PATH, 'figure4')

    fig = plt.figure(constrained_layout=False, figsize=(3.5, 3))
    spec2 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig, height_ratios=[5, 1, 5], width_ratios=[8, 1, 7])
    # ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[0, 2])
    ax3 = fig.add_subplot(spec2[2, 0])

    fig.subplots_adjust(left=0.07, bottom=0.06, right=1., top=0.98, wspace=0.1, hspace=0.1)

    # fontproperties2 = {'family': 'Arial', 'weight': 'bold', 'size': 8}

    # ax1.text(-0.1, .97,'a' , fontproperties2, transform=ax1.transAxes)
    # ax2.text(1.2, .97,'b' , fontproperties2, transform=ax1.transAxes)
    # ax3.text(2.2, .97,'b' , fontproperties2, transform=ax1.transAxes)

    # plot_stacked_hist_cnv( 'MDM4', ax1)
    # ax1.tick_params(axis="y", direction="out",length=0, pad=pad)
    # ax1.tick_params(axis="x", direction="out",length=0, pad=pad)

    plot_screen(ax2)
    plot_crispr_sem(ax3)
    ax3.tick_params(axis="y", direction="out", length=0, pad=pad)
    filename = join(saving_dir, 'figure4.png')
    plt.savefig(filename, dpi=300)

    matplotlib.rcParams['pdf.fonttype'] = 42
    filename = join(saving_dir, 'figure4.pdf')
    plt.savefig(filename)


if __name__ == "__main__":
    run()
