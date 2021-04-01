from os.path import join, basename, dirname
from config_path import PROSTATE_LOG_PATH, PLOTS_PATH
from os.path import join, exists

import pandas as pd
from matplotlib import pyplot as plt
import os
from os import listdir

from utils.stats_utils import score_ci

# base_dir = os.path.dirname(__file__)
import seaborn as sns
import numpy as np
from sklearn import metrics
import collections

def read_predictions(dirs_df):
    model_dict={}
    for i, row in dirs_df.iterrows():
        dir_ = row.file
        # model = row.Model + '_' +row.Size
        model = row.Model
        dir_ = join(base_dir, dir_)
        # prediction_file = [join(dir_,f) for f in listdir(dir_) if '0_testing.csv' in f][0]
        prediction_file = join(dir_,'P-net_ALL_testing.csv')
        pred_df = pd.read_csv(prediction_file)
        print(pred_df.shape)
        print(pred_df.head())
        model_dict[model] = pred_df
    return model_dict


def read_feature_ranks(dirs_df):
    model_dict={}
    for i, row in dirs_df.iterrows():
        dir_ = row.file
        # model = row.Model + '_' +row.Size
        model = row.Model
        dir_ = join(base_dir, dir_)
        'coef_P - net_ALL_layerh0'
        # prediction_file = [join(dir_,f) for f in listdir(dir_) if '0_testing.csv' in f][0]
        prediction_file = join(dir_,'fs/P-net_ALL_testing.csv')
        pred_df = pd.read_csv(prediction_file)
        print(pred_df.shape)
        print(pred_df.head())
        model_dict[model] = pred_df
    return model_dict

def plot_auc_bootstrap(all_models_dict, ax):
    n = len(all_models_dict.keys())
    colors = sns.color_palette(None, n)

    all_scores=[]
    names=[]
    xs=[]
    avg_scores=[]
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        score, ci_lower, ci_upper, scores = score_ci(y_test, y_pred_score, score_fun=metrics.roc_auc_score,
                                                     n_bootstraps=1000, seed=123)
        all_scores.append(scores)
        names.append(k)
        xs.append(np.random.normal(i + 1, 0.04, len(scores)))
        avg_scores.append(score)

    all_scores = [x for _, x in sorted(zip(avg_scores, all_scores))]
    names = [x for _, x in sorted(zip(avg_scores, names ))]

    ax.boxplot(all_scores, labels= names)
    ngroup = len(all_scores)
    clevels = np.linspace(0., 1., ngroup)
    from matplotlib import cm
    for i, (x, val, clevel) in enumerate(zip(xs, all_scores, clevels)):
        plt.scatter(x, val,marker='.', color=colors[i], alpha=0.1)


def plot_roc(ax, y_test, y_pred_score, save_dir,color, label=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    symbol = '-'
    if  'TF-IDF' in label:
        symbol = '-'
    elif 'JAMA' in label:
        symbol = '-'
    ax.plot(fpr, tpr, symbol, label=label + ' (%0.3f)' % roc_auc, linewidth=1, color=color)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontproperties)
    ax.set_ylabel('True Positive Rate', fontproperties)

def sort_dict(all_models_dict):
    sorted_dict = {}
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
        average_auc = metrics.auc(fpr, tpr)
        # average_auc = average_precision_score(y_test, y_pred_score)
        sorted_dict[k] = average_auc
        print('model {} , auc= {}'.format(k, average_auc))

    sorted_dict = sorted(sorted_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dict = collections.OrderedDict(sorted_dict)
    return sorted_dict

def plot_auc_all(all_models_dict, ax):
    # sort based on area under prc
    n = len(all_models_dict.keys())
    colors = sns.color_palette(None, n)

    sorted_dict = sort_dict(all_models_dict)
    for i, k in enumerate(sorted_dict.keys()):
        print('model {} , auc= {}'.format(k, sorted_dict[k]))
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        plot_roc(ax, y_test, y_pred_score, None, color=colors[i], label=k)

fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 10}

base_dir = join(PROSTATE_LOG_PATH,'review/fusion')



files=[]
files.append(dict(Model ='Fusion',   file=join(base_dir,'onsplit_average_reg_10_tanh_large_testing_fusion')))
files.append(dict(Model ='no-Fusion',   file=join(base_dir,'onsplit_average_reg_10_tanh_large_testing_fusion_zero')))
files.append(dict(Model ='Fusion (genes)',   file=join(base_dir,'onsplit_average_reg_10_tanh_large_testing_inner_fusion_genes')))
dirs_df = pd.DataFrame(files)

print dirs_df
model_dict= read_predictions(dirs_df)

current_dir= basename(dirname(__file__))
saving_dir = join(PLOTS_PATH, 'reviews/{}'.format(current_dir))
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

## compare predictions
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=400)
plot_auc_bootstrap(model_dict, ax)
filename = join(saving_dir, '_auc_bootsrtap')
plt.title('AUC (bootstrap)', fontsize=10)
plt.ylim(0.5, 1.05)
plt.savefig(filename, dpi=200)
plt.close()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=400)
plot_auc_all(model_dict, ax)
plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
plt.title('AUC', fontsize=10)


filename = join(saving_dir, '_auc_')

plt.savefig(filename, dpi=200)

## compare rank


