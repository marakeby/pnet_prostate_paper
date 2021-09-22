from os.path import join

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from config_path import DATA_PATH, LOG_PATH
from setup import saving_dir

base_dir = DATA_PATH


def plot_(primary):
    percent = 100 * primary / sum(primary)
    labels = ['True prediction', 'False prediction']
    xpos = [0, 0.4]
    width = [0.3, 0.3]
    fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
    fig.set_size_inches(3, 5)
    ax = axes
    plt.bar(xpos, percent, align='center', alpha=1.0, color=['black', 'red'], width=width)
    plt.ylim([0, 100])

    ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for i in ax.patches:
        if i.get_height() > 1.0:
            ax.text(i.get_x() + 0.3 * i.get_width(), i.get_height() + 5.0, '{:5.1f}%'.format(i.get_height()),
                    fontsize=15,
                    color='black', rotation=0)


def get_predictions(filename, correct_prediction=True):
    print filename
    df = pd.read_csv(filename, index_col=0)
    if correct_prediction:
        df.pred = df.pred_scores >= 0.5
    ind = (df.pred == 0) & (df.y == 0)
    df_correct = df[ind].copy()
    ind = (df.pred == 1) & (df.y == 0)
    df_wrong = df[ind].copy()
    return df, df_correct, df_wrong


def get_clinical():
    filename = join(base_dir, 'prostate/supporting_data/prad_p1000_clinical_final.txt')
    clinical_df = pd.read_csv(filename, sep='\t')
    return clinical_df


def plot_score_vs_pfs_time(filename, correct_prediction):
    clinical_df = get_clinical()
    df, correct, wrong = get_predictions(filename, correct_prediction=correct_prediction)
    print df.shape, correct.shape, wrong.shape

    def plot_data(dd, label):
        plt.figure()
        dd_merged = clinical_df.merge(dd, how='inner', left_on='Patient.ID', right_index=True)
        data = dd_merged[['PFS.time', 'pred_scores']].copy()
        data.dropna(inplace=True)
        # print data.head()
        data['PFS.time'] = data['PFS.time'] / 30
        linear_regressor = LinearRegression()
        X = data['pred_scores'].values.reshape(-1, 1)
        Y = data['PFS.time'].values.reshape(-1, 1)
        linear_regressor.fit(X, Y)
        Y_pred = linear_regressor.predict(X)
        # sns.lmplot(x='pred_scores', y='PFS.time', data=dd_test, fit_reg=True, ci=None)
        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        x = np.mean(X)
        y = np.mean(Y)
        plt.text(x, y, '$R^2$ = %0.2f' % r2_score(Y, Y_pred))
        # plt.text('Coefficient of determination: %.2f'% )
        # plt.legend(['Prediction score','PFS time'])
        plt.xlabel('Prediction score')
        plt.ylabel('PFS time')
        maxx = 1.1 * data['PFS.time'].max()
        plt.ylim((0, maxx))
        plt.title(label)

    plot_data(df, label='all')
    saving_filename = join(saving_dir, 'prediction_vs_pfs_time_all.png')
    print saving_filename
    plt.savefig(saving_filename)
    plot_data(correct, label='correct')
    saving_filename = join(saving_dir, 'prediction_vs_pfs_time_correct.png')
    plt.savefig(saving_filename)
    plot_data(wrong, label='wrong')
    saving_filename = join(saving_dir, 'prediction_vs_pfs_time_wrong.png')
    plt.savefig(saving_filename)


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


def move_spines(ax, sides, dists):
    '''
    Move the entire spine relative to the figure.
    Parameters:
      ax: axes to operate on
      sides: list of sides to move. Sides: top, left, bottom, right
      dists: list of float distances to move. Should match sides in length.
    Example:
    move_spines(ax, sides=['left', 'bottom'], dists=[-0.02, 0.1])
    '''
    for side, dist in zip(sides, dists):
        ax.spines[side].set_position(('axes', dist))
    return ax


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
    ax2 = divider.append_axes('bottom', size='10%', pad=0.6)

    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    # ax2_ypos = -0.20 * 6.0 / fig.get_figheight()

    # move_spines(ax2, ['bottom'], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ['top', 'right', 'bottom', 'left'])
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
                    s = "\n{}\\quad".format(l) + "{}"
                else:
                    s = "\n{}   ".format(l) + "{}"
            else:
                s = "\n{}"
            lbl += s.format(f.durations[f.durations >= tick].shape[0])
        ticklabels.append(lbl.strip())
    # Align labels to the right so numbers can be compared easily
    print ticklabels
    ax2.set_xticklabels(ticklabels, ha='right', fontsize=fontsize)
    ax2.set_yticks([])

    # Add a descriptive headline.
    ax2.xaxis.set_label_coords(0, 0.2)
    ax2.set_xlabel('At risk', fontsize=fontsize)

    # plt.tight_layout()
    return ax2


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


def move_spines(ax, sides, dists):
    '''
    Move the entire spine relative to the figure.
    Parameters:
      ax: axes to operate on
      sides: list of sides to move. Sides: top, left, bottom, right
      dists: list of float distances to move. Should match sides in length.
    Example:
    move_spines(ax, sides=['left', 'bottom'], dists=[-0.02, 0.1])
    '''
    for side, dist in zip(sides, dists):
        ax.spines[side].set_position(('axes', dist))
    return ax


def plot_surv(ax, filename, correct_prediction):
    labels = ['Low Model Score', 'High Model Score']
    # labels = [ 'Correct prediction', 'Wrong prediction']
    # sns.set_context("paper", font_scale=1)
    clinical_df = get_clinical()
    df, correct, wrong = get_predictions(filename, correct_prediction)
    print correct.shape, wrong.shape
    correct_full = clinical_df.merge(correct, how='inner', left_on='Patient.ID', right_index=True)
    wrong_full = clinical_df.merge(wrong, how='inner', left_on='Patient.ID', right_index=True)

    wrong_full = wrong_full.dropna(subset=['PFS.time', 'PFS'])
    correct_full = correct_full.dropna(subset=['PFS.time', 'PFS'])
    print correct_full.shape
    print wrong_full.shape

    # fig = plt.figure(figsize=(8, 6))
    # ax = plt.subplot()

    data = correct_full
    T1 = data['PFS.time'] / 30
    E1 = data['PFS']
    kmf1 = KaplanMeierFitter()
    kmf1.fit(T1, event_observed=E1, label=labels[0])  # or, more succinctly, kmf.fit(T, E)
    kmf1.plot(ax=ax, ci_show=ci_show, linewidth=3.0)

    data = wrong_full
    T2 = data['PFS.time'] / 30
    E2 = data['PFS']
    kmf2 = KaplanMeierFitter()
    kmf2.fit(T2, event_observed=E2, label=labels[1])  # or, more succinctly, kmf.fit(T, E)
    kmf2.plot(ax=ax, ci_show=ci_show, linewidth=3.0)

    newxticks = []
    for x in ax.get_xticks():
        if x >= 0:
            newxticks += [x]
        print newxticks
        ax.set_xticks(newxticks)

    # move_spines(ax, sides=[ 'bottom'], dists=[ 0.1])

    plot_margin = 10.

    # add_at_risk_counts_CUSTOM(kmf1, kmf2, ax=ax, fontsize=10, labels=['Low Model Score', 'High Model Score'])
    add_at_risk_counts_CUSTOM(kmf1, kmf2, ax=ax, fontsize=10)

    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    results.print_summary()

    ax.text(60, 0.4, "log-rank\np = %0.1g" % results.summary['p'], fontsize=10)
    ax.set_ylim((0, 1.05))
    ax.set_xlabel("Months", fontdict=dict(family='Arial', weight='bold', fontsize=14))

    ax.set_ylabel("Prop. Survival", fontdict=dict(family='Arial', weight='bold', fontsize=14))
    ax.legend(prop={'size': 10})
    # ax.set_aspect(1.)


filename = join(LOG_PATH, 'p1000/pnet/onsplit_average_reg_10_tanh_large_testing')
ci_show = False


def plot_surv_all(ax):
    correct_prediction = True
    full_filename = join(filename, 'P-net_ALL_testing.csv')
    plot_surv(ax, full_filename, correct_prediction)


def run():
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(6, 5))
    plot_surv_all(ax)
    plt.subplots_adjust(bottom=0.15, left=0.25)
    # saving_filename = join(dirname(filename), 'BCR.png')
    saving_filename = join(saving_dir, 'survival.png')
    plt.savefig(saving_filename, dpi=200)


if __name__ == '__main__':
    run()
