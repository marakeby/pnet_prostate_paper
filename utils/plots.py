import itertools
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import average_precision_score


def plot_auc_bootstrap(all_models_dict, ax):
    n = len(all_models_dict.keys())
    import seaborn as sns
    colors = sns.color_palette(None, n)

    all_scores = []
    names = []
    xs = []
    avg_scores = []
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_score']
        score, ci_lower, ci_upper, scores = score_ci(y_test, y_pred_score, score_fun=metrics.roc_auc_score,
                                                     n_bootstraps=1000, seed=123)
        all_scores.append(scores)
        names.append(k)
        xs.append(np.random.normal(i + 1, 0.04, len(scores)))
        avg_scores.append(score)

    all_scores = [x for _, x in sorted(zip(avg_scores, all_scores))]
    names = [x for _, x in sorted(zip(avg_scores, names))]

    ax.boxplot(all_scores, labels=names)
    ngroup = len(all_scores)
    clevels = np.linspace(0., 1., ngroup)
    for i, (x, val, clevel) in enumerate(zip(xs, all_scores, clevels)):
        plt.scatter(x, val, marker='.', color=colors[i], alpha=0.1)


def plot_roc(fig, y_test, y_pred_score, save_dir, label=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(fig.number)
    plt.plot(fpr, tpr, label=label + ' (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver operating characteristic (ROC)', fontsize=12)
    plt.legend(loc="lower right")


def plot_prc(fig, y_test, y_pred_score, save_dir, label=''):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = average_precision_score(y_test, y_pred_score)
    plt.figure(fig.number)
    plt.plot(recall, precision, label=label + ' (area under precision recall curve = %0.2f)' % roc_auc)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('precision', fontsize=12)
    plt.title('Precisoin Recall Curve (PRC)', fontsize=12)
    plt.legend(loc="lower right")


def generate_plots(test_scores, saving_dir):
    for c in test_scores.columns:
        plt.figure()
        print c
        ax = test_scores[c].plot(kind='bar', )
        ax.set_ylabel(c, fontsize=12)
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(join(saving_dir, str(c)))
    print test_scores


# gets list of data frames, each representing the model scores over n folds
def plot_box_plot(df, save_dir):
    # df = pd.concat(list_model_scores, axis=1, keys=model_names)
    df.columns = df.columns.swaplevel(0, 1)
    for c in df.columns.levels[0]:
        plt.figure()
        dd = df[c]
        print c, dd
        ax = dd.boxplot(showfliers=True)
        plt.ylim([-0.05, 1.05])
        ax.set_ylabel(str(c), fontsize=12)
        plt.tight_layout()
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.savefig(join(save_dir, c + '_boxplot'))

    pass


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().subplots_adjust(bottom=0.25)


def save_confusion_matrix(cnf_matrix, base_dir, model):
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                          title='Confusion matrix, without normalization')
    file_name = join(base_dir, 'confusion_' + model + '.png')
    plt.savefig(file_name)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, normalize=True, classes=[0, 1],
                          title='Normalized confusion matrix')
    file_name = join(base_dir, 'confusion_normalized_' + model)
    plt.savefig(file_name)
