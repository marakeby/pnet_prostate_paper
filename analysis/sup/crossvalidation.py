import pandas as pd
from matplotlib import pyplot as plt, ticker
from os.path import join
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
import seaborn as sns
import matplotlib as mpl

# set default params
from config_path import PROSTATE_LOG_PATH

custom_rcParams = {
    'figure.figsize': (8, 3),
    'font.family': 'Arial',
    'font.size': 14,
    'font.weight': 'regular',
    'axes.labelsize': 14,
    'axes.formatter.useoffset': False,
    'axes.formatter.limits': (-4, 4),
    'axes.titlesize': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'pdf.fonttype': 42
}

mpl.rcParams.update(custom_rcParams)
sns.set_context('paper', rc=custom_rcParams)
sns.set_style("white", {"grid.linestyle": u'--', "axes.grid": True, "grid.color":"0.9"})


def plot_box_plot(df, save_dir):
    # df = pd.concat(list_model_scores, axis=1, keys=model_names)

    df.columns = df.columns.swaplevel(0, 1)

    # sns.set(font_scale=1.6)

    columns = df.columns.levels[0]


    #     columns =[c.replace('_data_0','') for c in columns ]
    print 'columns',columns
    for c in columns:
        plt.figure(figsize=(7, 5))
        dd = df[c].copy()
        #         print dd.columns

        dd.columns = [mapping_dict_cols[a] for a in dd.columns]
        model_names = dd.columns
        # dd.columns = [r.replace('_data_0', '') for r in dd.columns]
        # dd.columns = [r.replace('P-net_ALL', 'P-NET') for r in dd.columns]
        # avg = dd['P-net_ALL'].median()
        avg = dd['P-NET'].median()
        sns.set_style("whitegrid")

        #         sns.boxplot(x = 'day', y = 'total_bill', data = tips)
        #         ax= dd.boxplot(showfliers=True)

        #         print dd.head()
        order = list(dd.median().sort_values().index)


        dd = dd.melt()
        #         print dd.head()


        # {"versicolor": "g", "setosa": "b", "virginica": "m"}

        ax = sns.boxplot(x="variable", y="value", data=dd, whis=np.inf, order=order,  palette=my_pal)
        #         ax = sns.swarmplot(x="variable", y="value", data=dd, color=".2")
        ax.axhline(avg, ls='--')

        plt.ylim([0.4, 1.05])
        # plt.ylabel(mapping_dict[c])
        # ax.set_ylabel(mapping_dict[c], fontsize=18)


        # ax.set_ylabel(mapping_dict[c], fontsize=12)
        ax.set_ylabel(mapping_dict[c], fontdict=dict(family='Arial', weight='bold', fontsize=14))

        ax.set_xlabel('')
        plt.tight_layout()
        # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=14)
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())

        # ax.grid(b=True, which='major', color='w', linewidth=1.5)
        # ax.grid(b=True, which='minor', color='w', linewidth=0.75)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.gcf().subplots_adjust(bottom=0.3, left=0.2)

        plt.savefig(join(save_dir, c + '_boxplot'), dpi=100)


# models= ['Random Forest', 'Support Vector Machine', 'L2 Logistic Regression', 'Adaptive Boosting','P-net']
# base_dir = '/Users/haithamelmarakeby/PycharmProjects/ml_pipeline/run/logs/prostate/prostate_paper/candidates/1/compare_ML/'
# base_dir = base_dir +'crossvalidation_5layers_cnv3_mut_binary_selu_samples_prostate_and_cancer_genes_no_debug_50_300_compare_test_Jan-04_21-24'
# models = ['Linear Support Vector Machine ', 'RBF Support Vector Machine ', 'L2 Logistic Regression', 'Random Forest',
#           'Adaptive Boosting', 'P-net']



mapping_dict = {'accuracy': 'Accuracy', 'auc': 'Area Under Curve (AUC)',
                'aupr': 'AUPRC', 'f1': 'F1', 'precision': 'Precision', 'percision': 'Precision', 'recall': 'Recall'}


# base_dir = './../../run/logs/p1000/'
base_dir = PROSTATE_LOG_PATH
models_base_dir = join(base_dir ,'compare/crossvalidation_ML_test_Apr-11_12-17')

df = pd.read_csv(join(models_base_dir, 'folds.csv'), sep=',', index_col=0, header=[0, 1])
print df.head()

pnet_base_dir = join(base_dir , 'pnet/crossvalidation_average_reg_10_tanh_Apr-11_13-20')
pnet_df = pd.read_csv(join(pnet_base_dir, 'folds.csv'), sep=',', index_col=0, header=[0, 1])
print pnet_df.head()

df = pd.concat([pnet_df, df], axis=1)
# fig = plt.figure(figsize=(8, 10))
# for m in models:
#     df = pd.read_csv(join(base_dir, m + '_data_0_testing.csv'), sep=',', index_col=0, header=[0, 1])
#     y_test = df['y']
#     y_pred_score = df['pred_scores']
#     plot_roc(fig, y_test, y_pred_score, None, label=m)


# m= 'pnet_crossvalidation_compare_Dec-13_07-07/folds.csv'

# models = ['Linear SVM', 'RBF SVM', 'L2 Logistic Regression', 'Random Forest',
#           'Ada. Boosting', 'Decision Tree', 'P-NET']

models = ['Decision Tree','L2 Logistic Regression',  'Random Forest', 'Ada. Boosting', 'Linear SVM', 'RBF SVM',
            'P-NET']

current_palette = sns.color_palette(None, len(models))

my_pal = {}
for i, m in enumerate(models):
    print current_palette[i]
    my_pal[m] = current_palette[i]

mapping_dict_cols = {'Adaptive Boosting_data_0': 'Ada. Boosting',
'Decision Tree_data_0':'Decision Tree' ,
'L2 Logistic Regression_data_0':'L2 Logistic Regression',
'Linear Support Vector Machine _data_0':'Linear SVM',
'Logistic Regression_ALL':'Logistic Regression',
'P-net_ALL':'P-NET',
'RBF Support Vector Machine _data_0':'RBF SVM',
'Random Forest_data_0':'Random Forest',
                    }

print df.columns.levels[0]
df = df.drop('dense_data_0', axis=1, level=0)
df = df.drop('Logistic Regression_ALL', axis=1, level=0)

print df.columns
plot_box_plot(df, './output/')

# plt.savefig('_auc', dpi=100)

