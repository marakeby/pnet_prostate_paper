import itertools
from os.path import join

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import NullFormatter, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import metrics
from sklearn.metrics import confusion_matrix, average_precision_score

from analysis.figure_2.figure2_utils import add_at_risk_counts_CUSTOM
from analysis.figure_2.figure2_utils import get_dense_sameweights, get_pnet_preformance, get_stats, plot_compaison
from config_path import DATA_PATH, LOG_PATH, PROSTATE_LOG_PATH, PROSTATE_DATA_PATH, PLOTS_PATH


def plot_prc_all(ax):
    def get_prc_data():
        all_models_dict = {}

        base_dir = PROSTATE_LOG_PATH
        models_base_dir = join(base_dir, 'compare/onsplit_ML_test')
        models = ['Linear Support Vector Machine ', 'RBF Support Vector Machine ', 'L2 Logistic Regression',
                  'Random Forest',
                  'Adaptive Boosting', 'Decision Tree']
        model_map = {'Linear Support Vector Machine ': 'Linear support vector machine ',
                     'RBF Support Vector Machine ': 'RBF support vector machine ',
                     'L2 Logistic Regression': 'L2 logistic regression',
                     'Random Forest': 'Random forest',
                     'Adaptive Boosting': 'Adaptive boosting',
                     'Decision Tree': 'Decision tree'

                     }
        for i, m in enumerate(models):
            df = pd.read_csv(join(models_base_dir, m + '_data_0_testing.csv'), sep=',', index_col=0, header=[0, 1])
            all_models_dict[model_map[m]] = df

        pnet_base_dir = join(base_dir, 'pnet/onsplit_average_reg_10_tanh_large_testing')
        df_pnet = pd.read_csv(join(pnet_base_dir, 'P-net_ALL_testing.csv'), sep=',', index_col=0, header=[0, 1])
        all_models_dict['P-NET'] = df_pnet
        return all_models_dict

    def plot_prc(ax, y_test, y_pred_score, save_dir, color, label=''):
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_score)
        roc_auc = average_precision_score(y_test, y_pred_score)
        ax.plot(recall, precision, label=label + ' (%0.2f)' % roc_auc, linewidth=linewidth, color=color)
        ax.set_xlim([0.0, 1.02])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel('Recall', fontproperties, labelpad=2)
        ax.set_ylabel('Precision', fontproperties, labelpad=2)
        # We change the fontsize of minor ticks label
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.tick_params(axis='both', which='minor', labelsize=5)

    all_models_dict = get_prc_data()
    n = len(all_models_dict.keys()) + 1
    colors = sns.color_palette(None, n)
    import collections

    # sort based on area under prc
    sorted_dict = {}
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        average_prc = average_precision_score(y_test, y_pred_score)
        sorted_dict[k] = average_prc

    sorted_dict = sorted(sorted_dict.items(), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_dict)
    print 'sorted_dict', sorted_dict

    for i, k in enumerate(sorted_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        print i, k
        plot_prc(ax, y_test, y_pred_score, None, label=k, color=colors[i])

    f_scores = np.linspace(0.2, 0.8, num=4)
    for i, f_score in enumerate(f_scores):
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.3, linewidth=0.5)
        if i == 0:
            tex = 'F1={0:.1f}'
            xy = (y[45] - 0.07, 1.02)
        else:
            tex = '{0:.1f}'
            xy = (y[45] - 0.03, 1.02)
        ax.annotate(tex.format(f_score), fontsize=fontsize, xy=xy, alpha=0.7)
    legend = ax.legend(loc="lower left", fontsize=fontsize - 0.5, framealpha=0.0, markerscale=0.1)

    for legend_handle in legend.legendHandles:
        legend_handle._legmarker.set_markersize(0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=0, width=0, grid_alpha=0.5)

    for tick in ax.get_yaxis().get_major_ticks():
        tick.set_pad(1)
    for tick in ax.get_xaxis().get_major_ticks():
        tick.set_pad(1)

    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)

    # ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=.1)


def plot_confusion_matrix_all(ax):
    base_dir = join(PROSTATE_LOG_PATH, 'pnet')
    models_base_dir = join(base_dir, 'onsplit_average_reg_10_tanh_large_testing')
    filename = join(models_base_dir, 'P-net_ALL_testing.csv')
    df = pd.read_csv(filename, index_col=0)

    # df.pred = df.pred_scores > 0.5
    df.head()
    y_t = df.y
    y_pred_test = df.pred
    cnf_matrix = confusion_matrix(y_t, y_pred_test)
    print cnf_matrix

    cm = np.array(cnf_matrix)
    classes = ['Primary', 'Metastatic']
    labels = np.array([['TN', 'FP'], ['FN ', 'TP']])

    plot_confusion_matrix(ax, cm, classes,
                          labels,
                          normalize=True,
                          cmap=plt.cm.Reds)
    ax.tick_params(axis=u'both', which=u'both', length=0)


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
    filename = join(DATA_PATH, 'prostate/supporting_data/prad_p1000_clinical_final.txt')
    clinical_df = pd.read_csv(filename, sep='\t')
    return clinical_df


def plot_surv(ax, filename, correct_prediction, ci_show):
    # labels = ['Low model score', 'High model score']
    labels = ['LPS', 'HPS']
    clinical_df = get_clinical()
    df, correct, wrong = get_predictions(filename, correct_prediction)
    print correct.shape, wrong.shape
    correct_full = clinical_df.merge(correct, how='inner', left_on='Patient.ID', right_index=True)
    wrong_full = clinical_df.merge(wrong, how='inner', left_on='Patient.ID', right_index=True)

    wrong_full = wrong_full.dropna(subset=['PFS.time', 'PFS'])
    correct_full = correct_full.dropna(subset=['PFS.time', 'PFS'])
    print correct_full.shape
    print wrong_full.shape

    data = correct_full
    T1 = data['PFS.time'] / 30
    E1 = data['PFS']
    kmf1 = KaplanMeierFitter()
    kmf1.fit(T1, event_observed=E1, label=labels[0])  # or, more succinctly, kmf.fit(T, E)
    kmf1.plot(ax=ax, ci_show=ci_show, linewidth=linewidth)

    data = wrong_full
    T2 = data['PFS.time'] / 30
    E2 = data['PFS']
    kmf2 = KaplanMeierFitter()
    kmf2.fit(T2, event_observed=E2, label=labels[1])  # or, more succinctly, kmf.fit(T, E)
    kmf2.plot(ax=ax, ci_show=ci_show, linewidth=linewidth)

    newxticks = []
    for x in ax.get_xticks():
        if x >= 0:
            newxticks += [x]
        print newxticks
        ax.set_xticks(newxticks)

    # add_at_risk_counts_CUSTOM(kmf1, kmf2, ax=ax, fontsize=fontsize, labels=['LMS','HMS'])
    add_at_risk_counts_CUSTOM(kmf1, kmf2, ax=ax, fontsize=fontsize)
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    results.print_summary()

    pval = results.summary['p'][0]
    # print pval.values, type(pval.values)
    if pval < 0.0001:
        text = "log-rank\np < 0.0001"
    else:
        text = "log-rank\np = %0.1g" % results.summary['p']
    ax.text(60, 0.4, text, fontsize=fontsize)
    ax.set_ylim((0, 1.05))
    ax.set_xlabel("Months", fontproperties, labelpad=1)
    ax.set_ylabel("Survival rate", fontproperties, labelpad=0)
    ax.legend(prop={'size': fontsize}, framealpha=0, loc='lower right')
    # ax.xaxis.set_label_position('top')
    # ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelsize=fontsize)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)


def plot_confusion_matrix(ax, cm, classes, labels=None,
                          normalize=False,
                          # title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig = plt.gcf()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.tick_params(axis=u'both', which=u'both', length=0)
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
                color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    ax.set_ylabel('True label', fontproperties)
    ax.set_xlabel('Predicted label', fontproperties)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontproperties)
    ax.set_yticks([t - 0.25 for t in tick_marks])
    ax.set_yticklabels(classes, fontproperties, rotation=90)


# def plot_external_validation_matrix(ax):
#     primary, mets = get_pimary_mets()
#     labels = np.array([['TN', 'FP'], ['FN ', 'TP']])
#     cm = np.array([primary, mets])
#     classes = ['{}\n{}'.format('Fraser et al.', '(localized)'), '{}\n{}'.format('Robinson et al.', '(metastatic)')]
#
#     plot_confusion_matrix(ax, cm, classes,
#                           labels,
#                           normalize=True,
#                           cmap=plt.cm.Reds)
#     ax.tick_params(axis=u'both', which=u'both', length=0)
#
#
def get_pimary_mets():
    dir_name = join(PROSTATE_LOG_PATH, 'external_validation/pnet_validation')
    primary_filename = join(dir_name, 'P-net__primary_testing.csv')
    met_filename = join(dir_name, 'P-net__mets_testing.csv')
    primary_df = pd.read_csv(primary_filename)
    met_df = pd.read_csv(met_filename)
    primary = [sum(primary_df.pred == False), sum(primary_df.pred == True)]
    mets = [sum(met_df.pred == False), sum(met_df.pred == True)]
    primary = np.array(primary)
    mets = np.array(mets)
    return primary, mets


# def plot_external_validation_matrix():


def plot_surv_all(ax):
    correct_prediction = True
    filename = join(LOG_PATH, 'p1000/pnet/onsplit_average_reg_10_tanh_large_testing')
    full_filename = join(filename, 'P-net_ALL_testing.csv')
    plot_surv(ax, full_filename, correct_prediction, ci_show=False)
    # ax.margins(0.1)
    # plt.gcf().subplots_adjust(top=0.9)
    # ax.tick_params(direction='in', length=.5, width=0, grid_alpha=0.5)
    for tick in ax.get_xaxis().get_major_ticks():
        tick.set_pad(4)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # ax.subplots_adjust(top=0.9)


def plot_pnet_vs_dense_with_ratio(ax, c, label, plot_ratio=False):
    sns.set_color_codes('muted')
    current_palette = sns.color_palette()
    color = current_palette[3]

    sizes = []
    for i in range(0, 20, 3):
        df_split = pd.read_csv(join(PROSTATE_DATA_PATH, 'splits/training_set_{}.csv'.format(i)), index_col=0)
        sizes.append(df_split.shape[0])
    sizes = np.array(sizes)

    df_dense_sameweights = get_dense_sameweights(c)
    df_pnet = get_pnet_preformance(col=c)
    pvalues = get_stats(df_pnet, df_dense_sameweights)
    print c, zip(pvalues, sizes)
    plot_compaison(ax, label, df_pnet, df_dense_sameweights, sizes, linewidth)
    ax.set_ylabel(label, fontproperties, labelpad=1)
    ax.legend(['P-NET', 'Dense'], fontsize=fontsize, loc='upper left', framealpha=0)

    y1 = df_pnet.mean()
    y2 = df_dense_sameweights.mean()
    height = map(max, zip(y1, y2))
    print 'height', height
    updated_values = []
    for i, (p, s) in enumerate(zip(pvalues, sizes)):
        if p >= 0.05:
            displaystring = r'NS'
        elif p < 0.0001:
            displaystring = r'***'
        elif p < 0.001:
            displaystring = r'**'
        else:
            displaystring = r'*'
        updated_values.append('{:.0f}\n{}'.format(s, displaystring))
        # ax.axvline(x=s, ymin=0.0, ymax=0.85,linestyle='--', alpha=0.3, linewidth=linewidth)
    ax.set_xscale("log")
    ax.set_xticks([], [])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(sizes)
    ax.set_xticklabels(updated_values, fontsize=fontsize)
    ax.set_xlim((min(sizes) - 5, max(sizes) + 50))

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.tick_params(axis=u'x', which=u'both', length=0)
    ax.tick_params(axis=u'y', which=u'both', length=0)

    if plot_ratio:
        ax2 = ax.twinx()
        y1 = df_pnet.mean()
        y2 = df_dense_sameweights.mean()
        ratio = (y1.values - y2.values) / y2.values
        new_x = np.linspace(min(sizes), max(sizes), num=np.size(sizes))
        coefs = np.polyfit(sizes, ratio, 3)
        new_line = np.polyval(coefs, new_x)

        ax2.plot(new_x, new_line, '-.', linewidth=linewidth, color=color)
        ax2.set_ylim((0.005, .23))
        ax.set_ylim((.5, 1.05))
        ax2.set_ylabel('Performance increase', fontproperties, labelpad=3)
        vals = ax2.get_yticks()
        # ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals], fontproperties)
        ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals], fontsize=fontsize)
        # ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax.set_yticks([], minor=True)
        ax2.spines['right'].set_color(color)
        ax2.yaxis.label.set_color(color)
        ax2.tick_params(axis='y', colors=color)
        ax2.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # ax2.spines['left'].set_visible(False)
        # ax2.spines['bottom'].set_visible(False)
        # ax2.tick_params(length=, width=1)
        ax2.tick_params(length=2, direction="in", pad=-15)
        for tick in ax2.get_yaxis().get_major_ticks():
            # tick.set_pad(-20)
            tick.set_pad(-15)

    ax.set_xlabel('Number of samples', fontproperties, labelpad=1)
    ax.tick_params(direction='out', length=.5, width=0, grid_alpha=0.5)
    # ax.legend(loc="lower left", fontsize=fontsize, framealpha=0.0)

    for tick in ax.get_yaxis().get_major_ticks():
        tick.set_pad(.7)
    for tick in ax.get_xaxis().get_major_ticks():
        tick.set_pad(.7)

    pvalues_dict = {}
    for p, s in zip(pvalues, sizes):
        pvalues_dict[s] = p

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    return pvalues_dict


def plot_external_validation_matrix(ax):
    primary, mets = get_pimary_mets()

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
    cax = divider.append_axes('right', size='10%', pad=0.05)
    # cax = divider.append_axes('left', size='10%', pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=fontsize, pad=1)
    cb.ax.tick_params(axis=u'both', which=u'both', length=0)
    cb.ax.set_ylabel('Sample percentage (%)', rotation=90, fontsize=fontsize, labelpad=1.5)
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
                color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    # ax.set_ylabel('True label', fontproperties)
    # ax.set_xlabel('Predicted label', fontproperties)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontproperties)
    ax.set_yticks([t - 0.25 for t in tick_marks])
    ax.set_yticks([])
    # ax2.set_yticks([])
    # ax2.set_xticks([])

    # ax.set_yticklabels(classes, fontproperties, rotation=90)

    # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # fig = plt.gcf()
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cb = fig.colorbar(im, cax=cax, orientation='vertical')
    # cb.ax.tick_params(labelsize=fontsize)
    # cb.ax.tick_params(axis=u'both', which=u'both', length=0)
    # cb.outline.set_visible(False)
    # tick_marks = np.arange(len(classes))
    #
    # if labels is None:
    #     fmt = '{:.2f}%' if normalize else '{:d}'
    # else:
    #     fmt = '{}: {:.2f}%' if normalize else '{}: {:d}'
    #
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     text = fmt.format(labels[i, j], cm[i, j])
    #     ax.text(j, i, text,
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)
    #
    #
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    #
    # ax.set_xticks(tick_marks )
    # ax.set_xticklabels(classes,fontproperties, rotation=0)
    # ax.set_yticks([])
    # ax2 = divider.append_axes('bottom', size='5%', pad=0.6)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # ax2.set_yticks([])
    # ax2.set_xticks([])


linewidth = 0.7
fontsize = 5  # legends, axis
fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}
saving_dir = join(PLOTS_PATH, 'figure2')


def run():
    fig = plt.figure(constrained_layout=False, figsize=(3.5, 3.))
    spec2 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig, width_ratios=[30, 0.5, 20], height_ratios=[10, 0.5, 10])

    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[0, 1:])
    ax3 = fig.add_subplot(spec2[2, 0])
    ax4 = fig.add_subplot(spec2[2, 2])

    plot_prc_all(ax1)
    # plot_confusion_matrix_all(ax2)
    plot_external_validation_matrix(ax2)

    plot_pnet_vs_dense_with_ratio(ax3, 'auc', 'AUC', plot_ratio=True)
    plot_surv_all(ax4)
    # plot_external_validation_matrix(ax3)
    fig.subplots_adjust(left=0.07, bottom=0.09, right=0.93, top=0.95, wspace=0.3, hspace=0.3)

    saving_filename = join(saving_dir, 'figure2.png')
    plt.savefig(saving_filename, dpi=300)
    matplotlib.rcParams['pdf.fonttype'] = 42

    saving_filename = join(saving_dir, 'figure2.pdf')
    plt.savefig(saving_filename)
    # fontproperties['size'] = 20
    # plt.text(0.1,0.1,'aedfcew,erw', fontproperties)

    # plt.savefig(saving_filename, dpi=300)


if __name__ == "__main__":
    run()
