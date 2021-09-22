from os.path import join

import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

from config_path import PROSTATE_LOG_PATH


def get_dense_sameweights(col='f1'):
    filename = join(PROSTATE_LOG_PATH, 'number_samples/crossvalidation_number_samples_dense_sameweights')
    filename = filename + '/folds.csv'
    df = pd.read_csv(filename, index_col=0, header=[0, 1])
    # print df.head()
    # print df.head()
    dd = df.swaplevel(0, 1, axis=1)[col].head()
    df_pnet_col = [c for c in dd.columns if 'dense' in c]
    df_pnet = dd[df_pnet_col]
    return df_pnet


def get_pnet_preformance(col='f1'):
    filename = join(PROSTATE_LOG_PATH, 'number_samples/crossvalidation_average_reg_10_tanh')
    filename = filename + '/folds.csv'
    df = pd.read_csv(filename, index_col=0, header=[0, 1])
    dd = df.swaplevel(0, 1, axis=1)[col].head()
    df_pnet_col = [c for c in dd.columns if 'P-net' in c]
    df_pnet = dd[df_pnet_col]
    return df_pnet


def get_stats(df_pnet, df_dense):
    print df_pnet.shape, df_dense.shape
    pvalues = []
    for c1, c2 in zip(df_pnet.columns, df_dense.columns):
        # print c
        x = df_pnet.loc[:, c1]
        y = df_dense.loc[:, c2]

        twosample_results = stats.ttest_ind(x, y)
        pvalue = twosample_results[1] / 2
        print pvalue
        pvalues.append(pvalue)
    return pvalues


def plot_compaison(ax1, label, df_pnet, df_dense, sizes, linewidth):
    y1 = df_pnet.mean()
    dy = df_pnet.std()
    x = sizes
    sns.set_color_codes('muted')
    current_palette = sns.color_palette()
    colors = current_palette[0:2]
    ax1.plot(x, y1, linestyle='-', marker='o', color=colors[0], linewidth=linewidth, markersize=2.)
    ax1.fill_between(x, y1 - dy, y1 + dy, color=colors[0], alpha=0.2)
    y2 = df_dense.mean()
    dy = df_dense.std()
    ax1.plot(x, y2, linestyle='-', marker='o', color=colors[1], linewidth=linewidth, markersize=2.)
    ax1.fill_between(x, y2 - dy, y2 + dy, color=colors[1], alpha=0.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


def add_at_risk_counts_CUSTOM(*fitters, **kwargs):
    '''
    Add counts showing how many individuals were at risk at each time point in
    survival/hazard plots.
    Arguments:
      One or several fitters, for example KaplanMeierFitter,
      NelsonAalenFitter, etc...
    Keyword arguments (all optional):
      ax: The axes to add the labels to. Default is the current axes.
      fig: The figure of the axes. Default is the current figure.
      labels: The labels to use for the fitters. Default is whatever was
              specified in the fitters' fit-function. Giving 'None' will
              hide fitter labels.
    Returns:
      ax: The axes which was used.
    Examples:
        # First train some fitters and plot them
        fig = plt.figure()
        ax = plt.subplot(111)
        f1 = KaplanMeierFitter()
        f1.fit(data)
        f1.plot(ax=ax)
        f2 = KaplanMeierFitter()
        f2.fit(data)
        f2.plot(ax=ax)
        # There are equivalent
        add_at_risk_counts(f1, f2)
        add_at_risk_counts(f1, f2, ax=ax, fig=fig)
        # This overrides the labels
        add_at_risk_counts(f1, f2, labels=['fitter one', 'fitter two'])
        # This hides the labels
        add_at_risk_counts(f1, f2, labels=None)
    '''
    from matplotlib import pyplot as plt

    # Axes and Figure can't be None
    ax = kwargs.get('ax', None)
    if ax is None:
        ax = plt.gca()

    fig = kwargs.get('fig', None)
    if fig is None:
        fig = plt.gcf()

    fontsize = kwargs.get('fontsize', None)
    if fontsize is None:
        fontsize = 15

    if 'labels' not in kwargs:
        labels = [f._label for f in fitters]
    else:
        # Allow None, in which case no labels should be used
        labels = kwargs['labels']
        if labels is None:
            labels = [None] * len(fitters)
    # Create another axes where we can put size ticks
    # ax2 = plt.twiny(ax=ax)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes('bottom', size='8%', pad=0.2)

    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    # ax2_ypos = -0.20 * 6.0 / fig.get_figheight()

    # move_spines(ax2, ['bottom'], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ['top', 'right', 'bottom', 'left'])
    # remove_spines(ax2, [ 'right', 'bottom', 'left'])
    # Set ticks and labels on bottom
    ax2.xaxis.tick_bottom()
    # Match tick numbers and locations
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())

    # Remove ticks, need to do this AFTER moving the ticks
    remove_ticks(ax2, x=True, y=True)
    # Add population size at times
    ticklabels = []
    for tick in ax2.get_xticks():
        lbl = ""
        for f, l in zip(fitters, labels):
            # First tick is prepended with the label
            if tick == ax2.get_xticks()[0] and l is not None:
                if is_latex_enabled():
                    # s = "\n{}\\quad".format(l) + "{}"
                    s = "{}\\quad".format(l) + "{}"
                else:
                    s = "\n{}   ".format(l) + "{}"
            else:
                s = "\n{}"
            lbl += s.format(f.durations[f.durations >= tick].shape[0])
        ticklabels.append(lbl.strip())
    # Align labels to the right so numbers can be compared easily
    # print ticklabels
    ax2.set_xticklabels(ticklabels, ha='right', fontsize=fontsize)
    ax2.set_yticks([])

    # Add a descriptive headline.
    ax2.xaxis.set_label_coords(0, 0.0)
    ax2.set_xlabel('At risk', fontsize=fontsize + 1)

    # plt.tight_layout()
    return ax2


def remove_ticks(ax, x=False, y=False):
    '''
    Remove ticks from axis.
    Parameters:
      ax: axes to work on
      x: if True, remove xticks. Default False.
      y: if True, remove yticks. Default False.
    Examples:
    removeticks(ax, x=True)
    removeticks(ax, x=True, y=True)
    '''
    if x:
        ax.xaxis.set_ticks_position('none')
    if y:
        ax.yaxis.set_ticks_position('none')
    return ax


# SURVIVAL ANALYSIS PLOTTING FUNCTIONS: copied from github/lifelines and edited for customization
def is_latex_enabled():
    '''
    Returns True if LaTeX is enabled in matplotlib's rcParams,
    False otherwise
    '''
    import matplotlib as mpl

    return mpl.rcParams['text.usetex']


def remove_spines(ax, sides):
    '''
    Remove spines of axis.
    Parameters:
      ax: axes to operate on
      sides: list of sides: top, left, bottom, right
    Examples:
    removespines(ax, ['top'])
    removespines(ax, ['top', 'bottom', 'right', 'left'])
    '''
    for side in sides:
        ax.spines[side].set_visible(False)
    return ax
