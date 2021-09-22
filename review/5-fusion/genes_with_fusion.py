from os.path import join
import numpy as np
import pandas as pd

from config_path import PROSTATE_LOG_PATH

base_dir = join(PROSTATE_LOG_PATH, 'review/fusion')
files = []
files.append(dict(Model='Fusion', file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_fusion')))
files.append(dict(Model='no-Fusion', file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_fusion_zero')))
files.append(
    dict(Model='Fusion (genes)', file=join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_inner_fusion_genes')))
dirs_df = pd.DataFrame(files)

layers = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']


def read_feature_ranks(dirs_df):
    model_dict = {}
    for l in layers:
        coef_df_list = []
        keys = []
        for i, row in dirs_df.iterrows():
            dir_ = row.file
            model = row.Model
            dir_ = join(dir_, 'fs')
            f = 'coef_P-net_ALL_layer{}.csv'.format(l)
            coef_file = join(dir_, f)
            coef_df = pd.read_csv(coef_file, index_col=0)
            coef_df.columns = [model]
            coef_df_list.append(coef_df)
            keys.append(model)
        coef_df = pd.concat(coef_df_list, axis=1)
        model_dict[l] = coef_df

    return model_dict


coef_df_dict = read_feature_ranks(dirs_df)

l = 'h0'
ranked = coef_df_dict[l].abs().rank(ascending=False)

genes_with_fusions = np.unique(
    ['AFDN', 'ARHGAP5', 'BIRC6', 'CLTC', 'CUX1', 'EIF4A2', 'ERG', 'GNA11', 'HMGN2P46', 'KIAA1549', 'KLK2', 'NCOA2',
     'NDRG1', 'NOTCH2', 'NSD1', 'RARA', 'SLC45A3', 'STAT3', 'SUZ12', 'TBL1XR1', 'TCF7L2', 'THRAP3', 'TMPRSS2', 'YWHAE',
     'ATF1', 'BRAF', 'CALR', 'EML4', 'ERG', 'ETV1', 'ETV4', 'ETV5', 'FGFR2', 'FGFR4', 'HSP90AA1', 'KLF4', 'MLLT3',
     'MYC', 'NF1', 'NSD3', 'PDE4DIP', 'PIK3CA', 'RPN1', 'SEPT9', 'SYK', 'TMPRSS2', 'VTI1A'])

fusions = ['EIF4A2--ETV5',
           'TMPRSS2--ERG',
           'TMPRSS2--ETV4',
           'RARA--SEPT9',
           'KIAA1549--BRAF',
           'SLC45A3--ETV1',
           'THRAP3--RPN1',
           'TMPRSS2--ETV1', 'TBL1XR1--PIK3CA', 'STAT3--ETV4',
           'NOTCH2--PDE4DIP', 'NDRG1--ERG', 'ERG--TMPRSS2', 'SUZ12--NF1',
           'TMPRSS2--BRAF', 'SLC45A3--ERG', 'TMPRSS2--KLF4',
           'HMGN2P46--TMPRSS2', 'TMPRSS2--ETV5', 'HMGN2P46--HSP90AA1',
           'KLK2--FGFR2', 'AFDN--SYK', 'CLTC--ETV4', 'NDRG1--MYC',
           'BIRC6--EML4', 'TMPRSS2--MLLT3', 'TCF7L2--VTI1A', 'KLK2--ETV1',
           'GNA11--CALR', 'CUX1--PDE4DIP', 'NSD1--FGFR4', 'YWHAE--ETV1',
           'ARHGAP5--ATF1', 'NCOA2--NSD3']

genes_with_fusions_df = ranked.loc[genes_with_fusions, :].copy()

genes_with_fusions_df['diff'] = (genes_with_fusions_df['Fusion (genes)'] - genes_with_fusions_df['no-Fusion']).abs()

print genes_with_fusions_df.sort_values('diff')

fusions_avg_rank = []
for f in fusions:
    g1, g2 = f.split('--')
    avg_rank_diff = (genes_with_fusions_df.loc[g1, 'diff'] + genes_with_fusions_df.loc[g2, 'diff']) / 2.
    fusions_avg_rank.append(dict(Fusion=f, avg_rank_diff=avg_rank_diff))
print pd.DataFrame(fusions_avg_rank).sort_values('avg_rank_diff')
