import os
from os import listdir
from os.path import join

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

from config_path import PROSTATE_LOG_PATH, PLOTS_PATH
from utils.evaluate import evalualte
from utils.plots import plot_roc, plot_prc
from utils.stats_utils import score_ci

dirname_pnet = join(PROSTATE_LOG_PATH, 'review/LOOCV_reg_10_tanh')
dirname_logistic = join(PROSTATE_LOG_PATH, 'review/LOOCV_reg_10_tanh_logistic')

saving_dir = join(PLOTS_PATH, 'reviews/11-LOOCV')
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)


def plot_bootstrap(all_models_dict, ax, score_fun=metrics.roc_auc_score):
    n = len(all_models_dict.keys())
    import seaborn as sns
    colors = sns.color_palette(None, n)

    all_scores = []
    names = []
    xs = []
    avg_scores = []
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y_test']
        y_pred_score = df['y_pred_test_scores']
        score, ci_lower, ci_upper, scores = score_ci(y_test, y_pred_score, score_fun=score_fun,
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


def get_predictions(dirname):
    onlyfiles = [f for f in listdir(dirname) if ('csv' in f)]

    n_list = []
    predictions_list = []
    for filename in onlyfiles:
        pred = pd.read_csv(join(dirname, filename), index_col=0)
        predictions_list.append(pred)

    df = pd.concat(predictions_list, axis=0)
    return df


df_pnet = get_predictions(dirname_pnet)
df_logistic = get_predictions(dirname_logistic)

df = df_pnet
ret_pnet = evalualte(y_test=df.y_test, y_pred=df.y_pred_test, y_pred_score=df.y_pred_test_scores)
df = df_logistic
ret_logistic = evalualte(y_test=df.y_test, y_pred=df.y_pred_test, y_pred_score=df.y_pred_test_scores)

print ret_pnet
print ret_logistic

fig = plt.figure()
df = df_pnet
plot_roc(fig, y_test=df.y_test, y_pred_score=df.y_pred_test_scores, save_dir=saving_dir, label='P-NET')
df = df_logistic
plot_roc(fig, y_test=df.y_test, y_pred_score=df.y_pred_test_scores, save_dir=saving_dir, label='Logistic Regression')
plt.savefig(join(saving_dir, 'auc_LOOCV.png'))

fig = plt.figure()
df = df_pnet
plot_prc(fig, y_test=df.y_test, y_pred_score=df.y_pred_test_scores, save_dir=saving_dir, label='P-NET')
df = df_logistic
plot_prc(fig, y_test=df.y_test, y_pred_score=df.y_pred_test_scores, save_dir=saving_dir, label='Logistic Regression')
plt.savefig(join(saving_dir, 'prc_LOOCV.png'))
plt.close()

models_dict = {}
models_dict['P-NET'] = df_pnet
models_dict['Logistic Regression'] = df_logistic
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
plot_bootstrap(models_dict, ax1, score_fun=metrics.roc_auc_score)
plt.savefig(join(saving_dir, 'auc_LOOCV_bootstrap.png'))
plt.close()

fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
plot_bootstrap(models_dict, ax2, score_fun=metrics.average_precision_score)
plt.savefig(join(saving_dir, 'prc_LOOCV_bootstrap.png'))
